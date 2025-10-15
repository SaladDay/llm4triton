"""
┌─────────────────────────────────────────────┐
│       cases/example_linear/model.py         │
└────────────────┬────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │  runner.execute_variant()  │ ← 执行入口
    └────────────┬───────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
   seed_rng()      forward_impl()
        │                 │
        │          ┌──────┴──────┐
        │          ▼             ▼
        │  agent_provide_  agent_provide_
        │    inputs()       weights()
        │          │             │
        │          └──────┬──────┘
        │                 ▼
        │           缓存系统
        │      (artifacts/_shared_cache)
        │
        └─────► 计时 + 验证 + 分析
                      │
                      ▼
              artifacts/{label}/
                ├── metrics.json
                ├── reference_outputs.pt
                └── profiler/trace.json
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

import torch

from .data import seed_rng

ARTIFACTS_ROOT = Path("artifacts")
REFERENCE_SNAPSHOT = "reference_outputs.pt"
DEFAULT_WARMUP_ITERS = 1
DEFAULT_MEASURE_ITERS = 10

__all__ = ["execute_variant"]


def execute_variant(
    label: str,                     
    forward_impl: Callable[[], Any],
    *,
    use_profiler: bool = False,
    warmup_iters: int = DEFAULT_WARMUP_ITERS,
    measure_iters: int = DEFAULT_MEASURE_ITERS,
    seed: Optional[int] = None,
    baseline_label: str = "baseline",
    compare_outputs: bool = True,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """
    评估入口

    Parameters
    ----------
    label:
        Unique identifier for the variant (e.g. ``"baseline"`` or ``"triton"``).
    forward_impl:
        零参数可调用对象，使用通过 `agent_provide_*` 辅助器提供的共享输入/权重执行模型前向传递。
    use_profiler:
        是否启用性能分析
    warmup_iters:
        预热迭代次数（默认1）
    measure_iters:
        测量迭代次数（默认10）
    seed:
        Optional override for the RNG seed. Defaults to the module-level seed.
    baseline_label:
        Variant label that establishes the reference outputs for correctness
        comparisons.
    compare_outputs:
        是否对比输出
    rtol / atol:
        数值容差

    Returns
    -------
    dict
        Captured metrics (latency, throughput, memory, correctness, etc.).



    1. 设置随机种子 (seed_rng)
    2. 重置CUDA内存统计
    3. 预热阶段 (warmup_iters 次)
    4. 计时测量阶段 (measure_iters 次)
    ├─ 记录开始时间
    ├─ 执行 forward_impl()
    ├─ GPU同步 (如果使用CUDA)
    └─ 计算耗时
    5. 收集性能指标：
    ├─ latency_ms: 平均延迟(毫秒)
    ├─ throughput_per_s: 吞吐量(次/秒)
    └─ max_memory_bytes: 峰值显存
    6. 正确性验证：
    ├─ 如果是baseline: 保存输出作为参考
    └─ 如果是candidate: 与baseline对比
    7. 保存结果到 artifacts/{label}/metrics.json
    """
    if measure_iters <= 0:
        raise ValueError("measure_iters must be a positive integer")

    seed_used = seed_rng(seed)

    variant_dir = ARTIFACTS_ROOT / label
    variant_dir.mkdir(parents=True, exist_ok=True)
    profiler_dir = variant_dir / "profiler"

    _maybe_reset_cuda_stats()

    with torch.inference_mode():
        for _ in range(max(0, warmup_iters)):
            forward_impl()
            _synchronize_if_needed()

    start_time = time.perf_counter()
    outputs: Any = None
    with torch.inference_mode():
        for _ in range(measure_iters):
            outputs = forward_impl()
            _synchronize_if_needed()
    elapsed = time.perf_counter() - start_time

    latency_ms = (elapsed / measure_iters) * 1000.0 if elapsed > 0 else float("inf")
    throughput = measure_iters / elapsed if elapsed > 0 else float("inf")
    memory_bytes = _get_max_memory_allocated()

    correctness = None
    correctness_details: dict[str, float] | str | None = None
    if compare_outputs and outputs is not None:
        correctness, correctness_details = _handle_correctness(
            label=label,
            baseline_label=baseline_label,
            outputs=outputs,
            rtol=rtol,
            atol=atol,
        )

    metrics: dict[str, Any] = {
        "label": label,
        "seed": seed_used,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "latency_ms": latency_ms,
        "throughput_per_s": throughput,
        "max_memory_bytes": memory_bytes,
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "correctness": correctness,
    }
    if isinstance(correctness_details, dict):
        metrics.update({f"correctness_{k}": v for k, v in correctness_details.items()})
    elif isinstance(correctness_details, str):
        metrics["correctness_note"] = correctness_details

    metrics_path = variant_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    if use_profiler:
        profiler_dir.mkdir(parents=True, exist_ok=True)
        _capture_profiler_trace(forward_impl, profiler_dir)

    return metrics


def _handle_correctness(
    *,
    label: str,
    baseline_label: str,
    outputs: Any,
    rtol: float,
    atol: float,
) -> tuple[bool, dict[str, float] | str]:
    actual_snapshot = _detach_to_cpu(outputs)
    reference_path = ARTIFACTS_ROOT / baseline_label / REFERENCE_SNAPSHOT

    if label == baseline_label:
        _atomic_save(actual_snapshot, reference_path)
        return True, {"max_abs_diff": 0.0, "max_rel_diff": 0.0}

    if not reference_path.exists():
        return (
            False,
            f"Reference outputs missing at {reference_path}. Run the baseline first.",
        )

    expected_snapshot = torch.load(reference_path, map_location="cpu")
    comparison = _compare_structures(
        actual_snapshot, expected_snapshot, rtol=rtol, atol=atol
    )
    return comparison["match"], {
        "max_abs_diff": comparison["max_abs_diff"],
        "max_rel_diff": comparison["max_rel_diff"],
    }


def _compare_structures(
    actual: Any,
    expected: Any,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    state = {
        "match": True,
        "max_abs_diff": 0.0,
        "max_rel_diff": 0.0,
    }

    def _update_diffs(diff_tensor: torch.Tensor, expected_tensor: torch.Tensor) -> None:
        if diff_tensor.numel() == 0:
            return
        abs_diff = diff_tensor.abs().max().item()
        state["max_abs_diff"] = max(state["max_abs_diff"], float(abs_diff))

        expected_abs = expected_tensor.abs().max().item()
        denom = expected_abs if expected_abs > 0 else 1.0
        rel_diff = abs_diff / denom
        state["max_rel_diff"] = max(state["max_rel_diff"], float(rel_diff))

    def _recurse(a: Any, b: Any) -> None:
        if not state["match"]:
            return
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape != b.shape or a.dtype != b.dtype:
                state["match"] = False
                return
            diff = a - b
            _update_diffs(diff, b)
            if not torch.allclose(a, b, rtol=rtol, atol=atol):
                state["match"] = False
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            if set(a.keys()) != set(b.keys()):
                state["match"] = False
                return
            for key in a:
                _recurse(a[key], b[key])
            return
        if isinstance(a, tuple) and hasattr(a, "_fields") and isinstance(b, tuple):
            if type(a) != type(b) or len(a) != len(b):
                state["match"] = False
                return
            for item_a, item_b in zip(a, b):
                _recurse(item_a, item_b)
            return
        if isinstance(a, Sequence) and not isinstance(a, (str, bytes)):
            if type(a) != type(b) or len(a) != len(b):
                state["match"] = False
                return
            for item_a, item_b in zip(a, b):
                _recurse(item_a, item_b)
            return
        state["match"] = bool(a == b)

    _recurse(actual, expected)
    return state


def _capture_profiler_trace(
    forward_impl: Callable[[], Any],
    output_dir: Path,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    wait, warmup, active, repeat = 0, 1, 3, 1
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat)
    trace_path = output_dir / "trace.json"

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=lambda prof: prof.export_chrome_trace(str(trace_path)),
    ) as prof:
        steps = (wait + warmup + active) * repeat
        with torch.inference_mode():
            for _ in range(steps):
                forward_impl()
                _synchronize_if_needed()
                prof.step()


def _maybe_reset_cuda_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _get_max_memory_allocated() -> Optional[int]:
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        return int(torch.cuda.max_memory_allocated(device_index))
    return None


def _detach_to_cpu(data: Any) -> Any:
    return _tree_apply(
        data,
        lambda tensor: tensor.detach().cpu(),
    )


def _tree_apply(data: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    if torch.is_tensor(data):
        return fn(data)
    if isinstance(data, Mapping):
        return type(data)(
            (key, _tree_apply(value, fn)) for key, value in data.items()
        )
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(_tree_apply(value, fn) for value in data))
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return type(data)(_tree_apply(item, fn) for item in data)
    return data


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
