"""Windows ColBERTv2 compatibility patches.

Disables all C++ extensions and provides pure-Python equivalents.
Only imported when os.name == "nt" (Windows dev). On Linux (GPU server),
C++ extensions compile normally and this module is never used.

Patched:
  - ColBERT.try_load_torch_extensions → no-op
  - ColBERT.segmented_maxsim → pure-Python
  - StridedTensor.try_load_torch_extensions → no-op
  - StridedTensor.segmented_lookup → pure-Python
  - IndexScorer.try_load_torch_extensions → no-op
"""
from __future__ import annotations

import torch


def _py_segmented_lookup(tensor, pids, lengths, offsets):
    """Pure-Python replacement for segmented_lookup_cpp."""
    B = int(pids.shape[0]) if hasattr(pids, "shape") else len(pids)
    lengths_list = lengths.tolist() if hasattr(lengths, "tolist") else list(lengths)
    offsets_list = offsets.tolist() if hasattr(offsets, "tolist") else list(offsets)

    if B == 0:
        return tensor.new_zeros((0, 0))

    max_len = max(int(x) for x in lengths_list) if lengths_list else 0
    if max_len == 0:
        if tensor.dim() == 1:
            return tensor.new_zeros((B, 0))
        return tensor.new_zeros((B, 0, tensor.shape[-1]))

    if tensor.dim() == 1:
        out = tensor.new_zeros((B, max_len))
        for i in range(B):
            off, ln = int(offsets_list[i]), int(lengths_list[i])
            if ln > 0:
                out[i, :ln] = tensor[off:off + ln]
    else:
        D = tensor.shape[-1]
        out = tensor.new_zeros((B, max_len, D))
        for i in range(B):
            off, ln = int(offsets_list[i]), int(lengths_list[i])
            if ln > 0:
                out[i, :ln] = tensor[off:off + ln]
    return out


def _py_segmented_maxsim(scores, D_lengths):
    """Pure-Python replacement for segmented_maxsim_cpp."""
    num_docs = int(D_lengths.shape[0])
    out = torch.zeros(num_docs, dtype=scores.dtype, device=scores.device)
    offset = 0
    for i in range(num_docs):
        L = int(D_lengths[i].item())
        if L > 0:
            doc_scores = scores[offset:offset + L, :]
            out[i] = doc_scores.max(dim=0).values.sum()
        offset += L
    return out


def patch_colbert_for_windows() -> None:
    """Install pure-Python fallbacks. Idempotent."""
    # --- ColBERT ---
    from colbert.modeling import colbert as _c
    if not getattr(_c.ColBERT, "_sah_patched", False):
        _c.ColBERT.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)
        _c.ColBERT.segmented_maxsim = staticmethod(_py_segmented_maxsim)
        _c.ColBERT._sah_patched = True

    # --- StridedTensor ---
    from colbert.search import strided_tensor as _st
    if not getattr(_st.StridedTensor, "_sah_patched", False):
        _st.StridedTensor.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)
        _st.StridedTensor.segmented_lookup = staticmethod(_py_segmented_lookup)
        _st.StridedTensor._sah_patched = True

    # --- IndexScorer (the crash site) ---
    from colbert.search import index_storage as _is
    if not getattr(_is.IndexScorer, "_sah_patched", False):
        # Replace the loader with a no-op
        _is.IndexScorer.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)
        # Pre-set the C++ attributes to None (colbert-ai's rank() should have
        # Python fallbacks when these are None — see index_storage.py source)
        # Set the actual attribute names used by score_pids() to None.
        # colbert-ai's score_pids has `if IndexScorer.filter_pids is None:`
        # checks that fall back to pure-Python implementations.
        _is.IndexScorer.filter_pids = None
        _is.IndexScorer.decompress_residuals = None
        _is.IndexScorer.filter_pids_cpp = None
        _is.IndexScorer.decompress_residuals_cpp = None
        _is.IndexScorer.loaded_extensions = True
        _is.IndexScorer._sah_patched = True

    print("[SAH] Windows compat: all C++ extensions disabled, pure-Python fallbacks active")
