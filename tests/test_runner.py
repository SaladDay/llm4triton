from __future__ import annotations

import torch

from harness.cases import make_case_forward
from harness.runner import execute_variant


def test_execute_variant_generates_metrics() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    forward = make_case_forward("example_linear", device=device)

    baseline_metrics = execute_variant(
        "baseline",
        forward,
    )
    candidate_metrics = execute_variant(
        "candidate",
        forward,
    )

    assert baseline_metrics["correctness"] is True
    assert candidate_metrics["correctness"] is True
    assert "latency_ms" in candidate_metrics
    assert "throughput_per_s" in candidate_metrics
