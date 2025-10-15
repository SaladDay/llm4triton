"""
Minimal example exercising the evaluation harness with the example workload.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch

from harness.cases import make_case_forward
from harness.runner import execute_variant


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward = make_case_forward("example_linear", device=device)

    metrics_baseline = execute_variant(
        "baseline",
        forward,
    )
    metrics_candidate = execute_variant(
        "candidate",
        forward,
    )

    output_path = Path("examples") / "last_run_metrics.json"
    output_path.write_text(
        json.dumps(
            {
                "baseline": metrics_baseline,
                "candidate": metrics_candidate,
            },
            indent=2,
        )
    )
    print(f"Wrote metrics to {output_path.resolve()}")


if __name__ == "__main__":
    main()
