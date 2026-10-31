"""Shared LLM implementations for controlled-mode injection.

Implements the ``baselines.base.LLM`` Protocol. Model is lazily loaded
and cached as a module-level singleton.

Fairness policy (v2): controlled-mode indoor LLM MUST be Qwen2.5-14B-Instruct.
Passing a smaller model violates the fairness protocol unless strict_fairness=False
is explicit (e.g. tests, debugging).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_ROOT = r"D:\rag_work\models\Qwen2.5-14B-Instruct"
_ENV_VAR = "RAG_INDOOR_LLM_ROOT"


def _resolve_snapshot(root: Path) -> Path:
    """If root has a `snapshots/<rev>` layout, return the first snapshot dir."""
    snap = root / "snapshots"
    if snap.exists():
        candidates = [p for p in snap.iterdir() if p.is_dir()]
        if candidates:
            return sorted(candidates)[0]
    return root


class QwenLLM:
    """Qwen2.5-14B-Instruct wrapper for controlled-mode injection.

    Controlled protocol mandates Qwen2.5-14B-Instruct (see
    docs/llm_fairness_policy.md). Passing a smaller model with
    strict_fairness=True raises ValueError.
    """

    _DEFAULT_MAX_TOKENS = 256
    _DEFAULT_TEMPERATURE = 0.0

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        strict_fairness: bool = True,
        allow_download: bool = False,
    ) -> None:
        if model_path is None:
            env_path = os.environ.get(_ENV_VAR) or os.environ.get(_DEFAULT_ROOT)
            if not env_path:
                raise ValueError(
                    "QwenLLM requires explicit model_path. "
                    "Controlled protocol mandates Qwen2.5-14B-Instruct. "
                    f"Pass model_path=... or set env {_ENV_VAR}."
                )
            model_path = env_path
        path = Path(model_path)
        if strict_fairness and "14B" not in str(path):
            raise ValueError(
                f"Controlled protocol mandates Qwen2.5-14B-Instruct. "
                f"Got: {path}. Pass strict_fairness=False for tests/debugging."
            )
        self._model_path = _resolve_snapshot(path)
        self._strict_fairness = strict_fairness
        self._allow_download = allow_download
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: str = "cpu"
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "QwenLLM requires torch and transformers. "
                f"Cause: {exc}"
            ) from exc

        local_only = not self._allow_download
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._model_path), local_files_only=local_only
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                str(self._model_path), local_files_only=local_only
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Qwen from {self._model_path}. "
                f"Pass allow_download=True or check the path. Cause: {exc}"
            ) from exc

        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._torch = torch
        log.info("QwenLLM loaded from %s on %s", self._model_path, self._device)

    @property
    def name(self) -> str:
        return f"Qwen/{self._model_path.name}"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self._ensure_loaded()
        torch = self._torch
        max_tokens = int(kwargs.get("max_tokens", self._DEFAULT_MAX_TOKENS))
        temperature = float(kwargs.get("temperature", self._DEFAULT_TEMPERATURE))
        do_sample = temperature > 0.0

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(gen, skip_special_tokens=True).strip()
