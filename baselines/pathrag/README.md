# PathRAG (Reported Baseline)

**Status:** REPORTED  
**Mode:** paper numbers only (no executable code)  
**Paper:** arXiv 2502.14902 / AAAI-26

## Classification rationale

PathRAG was attempted as EXECUTABLE via a wrapper around the official
repository (`BUPT-GAMMA/PathRAG`). It was rejected on 5 grounds:

| # | Blocker | Decision |
|---|---|---|
| B1 | Wrapper imports from external repo with no LICENSE -> non-distributable | REPORTED |
| B2 | Qwen2.5-1.5B produced 0 edges -> not actually PathRAG | REPORTED |
| B3 | Controlled mode requires injected LLM; only mock was validated | REPORTED |
| B4 | Hardcoded paths violated contract section 6.5 | FIXED, but too late |
| B5 | Registry supports only [native, controlled]; wrapper needed 3 sub-modes | REPORTED |

Reviewer test: would a reviewer who read the paper say the wrapper is the
same method? No - a 0-edge graph is flat retrieval wearing PathRAG's name.

## Files in this baseline

- `adapter.py` - `ReportedBaseline` subclass
- `reported_metrics.yaml` - structure for paper values (awaiting extraction)
- `README.md` - this file

## What is NOT here

- No executable retrieval code
- No config_native.yaml / config_controlled.yaml
- No tests
- No external repo dependency

## What happens to the wrapper

Preserved at `workspace/pathrag_wrapper_future/`. It is:

- NOT a graduation candidate
- NOT registered in configs/baselines_registry.yaml
- Blocked on: GPU (for 7B+ LLM) + license resolution

## Usage

Usage example (Python):

    from pathlib import Path
    from baselines.pathrag.adapter import PathRAGReportedBaseline
    from baselines.base import BaselineConfig, RunMode

    baseline = PathRAGReportedBaseline(
        config=BaselineConfig(name="pathrag", mode=RunMode.NATIVE, top_k=10),
        work_dir=Path("outputs/pathrag_reported"),
    )
    metrics = baseline.get_reported_metrics("2wikimultihopqa")  # empty until filled

    from pathlib import Path
    from baselines.pathrag.adapter import PathRAGReportedBaseline
    from baselines.base import BaselineConfig, RunMode

    baseline = PathRAGReportedBaseline(
        config=BaselineConfig(name="pathrag", mode=RunMode.NATIVE, top_k=10),
        work_dir=Path("outputs/pathrag_reported"),
    )
    metrics = baseline.get_reported_metrics("2wikimultihopqa")  # empty until filled


## Caveats

1. Win-rate is not a retrieval metric. Cannot appear on the same
   leaderboard row as executable baselines that produce IR metrics.

2. Comparison baselines matter. A win-rate without naming the opponents
   is meaningless.

3. Data extraction is pending. Supervisor must fill reported_metrics.yaml
   from the paper.

## License

- Paper: arXiv non-exclusive distribution license.
- Official code: no LICENSE file -> all rights reserved.
- This baseline: no code copied from the official repo. Original shell only.
