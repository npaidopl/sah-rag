"""Auto-generated framing decision document based on ablation outcomes."""
from pathlib import Path
import json

from ablation.config import REPORTS_DIR
from ablation.reporting.tables import load_all_results, macro_average
from ablation.config import EXCLUDE_FROM_MACRO


def _macro(results, variant):
    return macro_average(results, variant, EXCLUDE_FROM_MACRO)


def decide(results) -> dict:
    """Return a dict of findings and their framing implications."""
    findings = {}
    ref = _macro(results, "A4_full_sah")
    findings["reference"] = ref

    def _diff(vname):
        m = _macro(results, vname)
        if not m["recall@10"] and not ref["recall@10"]:
            return None
        return {"recall@10": m["recall@10"], "delta": m["recall@10"] - ref["recall@10"],
                "metrics": m}

    # Decisive: A1 flat hybrid
    d_a1 = _diff("A1_flat_hybrid")
    if d_a1 is not None:
        if abs(d_a1["delta"]) < 0.01:
            findings["a1_flat_hybrid"] = {
                "verdict": "SAH-RAG ≈ flat hybrid. Structure awareness does NOT help.",
                "action": "Reframe as good hybrid pipeline, not structure-aware architecture.",
                "data": d_a1,
            }
        elif d_a1["delta"] > 0:
            findings["a1_flat_hybrid"] = {
                "verdict": f"SAH-RAG > flat hybrid by {d_a1['delta']:+.4f}. Structure awareness VALIDATED.",
                "action": "Highlight structural parser as primary contribution.",
                "data": d_a1,
            }
        else:
            findings["a1_flat_hybrid"] = {
                "verdict": f"Flat hybrid > SAH-RAG by {-d_a1['delta']:+.4f}. Structure HURTS.",
                "action": "Investigate why structural parents degrade retrieval. Consider removing parser.",
                "data": d_a1,
            }

    # A7 paragraph (universal structure)
    d_a7 = _diff("A7_paragraph_parents")
    if d_a7 is not None:
        findings["a7_paragraph"] = {
            "verdict": f"A7 (paragraph parents) delta vs A4: {d_a7['delta']:+.4f}",
            "action": "If A7 > A4, universal structure (paragraphs) beats parser-specific structure.",
            "data": d_a7,
        }

    # A6 boundaries
    d_a6 = _diff("A6_boundaries_disabled")
    if d_a6 is not None:
        findings["a6_boundaries"] = {
            "verdict": f"A6 (cross-boundary) delta: {d_a6['delta']:+.4f}",
            "action": "If A6 < A4, boundary preservation has value. If ≈ A4, boundaries don't matter.",
            "data": d_a6,
        }

    # A9 placebo
    d_a9 = _diff("A9_placebo_boundaries")
    if d_a9 is not None:
        findings["a9_placebo"] = {
            "verdict": f"A9 (random boundaries) delta: {d_a9['delta']:+.4f}",
            "action": "If A9 ≈ A4, structure shape (not semantics) is what helps.",
            "data": d_a9,
        }

    # G8 random sparse
    d_g8 = _diff("G8_random_sparse")
    if d_g8 is not None:
        findings["g8_random_sparse"] = {
            "verdict": f"G8 (random sparse) delta: {d_g8['delta']:+.4f}",
            "action": "If G8 ≈ A4, learned sparse adds only noise-like diversity.",
            "data": d_g8,
        }

    # D5 BM25
    d_bm25 = _diff("D5_bm25_only")
    if d_bm25 is not None:
        findings["d5_bm25"] = {
            "verdict": f"BM25-only delta: {d_bm25['delta']:+.4f}",
            "action": "If BM25 ≈ A4, dense/learned-sparse complexity isn't justified.",
            "data": d_bm25,
        }

    return findings


def render_markdown(findings: dict) -> str:
    lines = ["# Framing Decision Document", ""]
    lines.append(f"Reference (A4_full_sah) macro R@10: {findings['reference']['recall@10']:.4f}")
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    for key, f in findings.items():
        if key == "reference":
            continue
        lines.append(f"### {key}")
        lines.append(f"- **Verdict:** {f['verdict']}")
        lines.append(f"- **Action:** {f['action']}")
        data = f.get("data", {})
        lines.append(f"- **R@10:** {data.get('recall@10', 0):.4f}  "
                     f"(Δ = {data.get('delta', 0):+.4f})")
        lines.append("")
    return "\n".join(lines)


def main():
    results = load_all_results()
    if not results:
        print("No results yet. Run after GPU stage.")
        return 1
    findings = decide(results)
    md = render_markdown(findings)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "framing.md").write_text(md, encoding="utf-8")
    (REPORTS_DIR / "framing.json").write_text(
        json.dumps(findings, indent=2, default=float), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())