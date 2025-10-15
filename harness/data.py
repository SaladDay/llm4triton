"""
提供确定性的输入和权重数据生成,带缓存机制
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

__all__ = [
    # 生成或加载输入数据，自动转换到指定设备和数据类型
    "agent_provide_inputs", 
    # 生成或加载模型权重
    "agent_provide_weights",
    # 统一设置所有随机种子
    "seed_rng",
    "set_default_seed",
]


DEFAULT_CACHE_ROOT = Path("artifacts") / "_shared_cache"
_DEFAULT_SEED = 20251013


def set_default_seed(seed: int) -> None:
    global _DEFAULT_SEED
    _DEFAULT_SEED = seed


def seed_rng(seed: Optional[int] = None) -> int:
    """
    Seed all supported RNGs
    """
    actual_seed = _DEFAULT_SEED if seed is None else seed
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(actual_seed)
    return actual_seed


TensorFactory = Callable[[], Any]


def agent_provide_inputs(
    case_name: str,
    factory: TensorFactory,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    cache_root: Optional[Path | str] = None,
    force_rebuild: bool = False,
) -> Any:
    """
    1. 检查缓存文件 artifacts/_shared_cache/{case_name}/inputs.pt 或 weights.pt
    2. 如果存在 → 加载缓存（确保可复现）
    3. 如果不存在 → 调用factory生成 → 保存到缓存
    4. 自动转换到指定的device和dtype
    """
    return _load_or_create(
        case_name=case_name,
        kind="inputs",
        factory=factory,
        device=device,
        dtype=dtype,
        cache_root=cache_root,
        force_rebuild=force_rebuild,
    )


def agent_provide_weights(
    case_name: str,
    factory: TensorFactory,
    *,
    device: Optional[torch.device | str] = None,
    dtype: Optional[torch.dtype] = None,
    cache_root: Optional[Path | str] = None,
    force_rebuild: bool = False,
) -> Any:

    return _load_or_create(
        case_name=case_name,
        kind="weights",
        factory=factory,
        device=device,
        dtype=dtype,
        cache_root=cache_root,
        force_rebuild=force_rebuild,
    )


def _load_or_create(
    *,
    case_name: str,
    kind: str,
    factory: TensorFactory,
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
    cache_root: Optional[Path | str],
    force_rebuild: bool,
) -> Any:
    cache_dir = _materialize_cache_dir(case_name, cache_root)
    cache_file = cache_dir / f"{kind}.pt"

    snapshot: Any
    if cache_file.exists() and not force_rebuild:
        snapshot = torch.load(cache_file, map_location="cpu")
    else:
        built = factory()
        snapshot = _detach_to_cpu(built)
        _atomic_save(snapshot, cache_file)
    return _prepare_for_consumption(snapshot, device=device, dtype=dtype)


def _materialize_cache_dir(case_name: str, cache_root: Optional[Path | str]) -> Path:
    root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    path = root / case_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_save(obj: Any, path: Path) -> None:
    """
    原子保存机制
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


def _detach_to_cpu(data: Any) -> Any:
    return _apply_to_tensors(
        data,
        lambda tensor: tensor.detach().cpu(),
    )


def _prepare_for_consumption(
    data: Any,
    *,
    device: Optional[torch.device | str],
    dtype: Optional[torch.dtype],
) -> Any:
    if device is None and dtype is None:
        return _apply_to_tensors(
            data,
            lambda tensor: tensor.detach(),
        )

    def _converter(tensor: torch.Tensor) -> torch.Tensor:
        out = tensor.detach()
        kwargs: dict[str, Any] = {}
        if device is not None:
            kwargs["device"] = device
        if dtype is not None:
            kwargs["dtype"] = dtype
        if kwargs:
            out = out.to(**kwargs)
        return out

    return _apply_to_tensors(data, _converter)


def _apply_to_tensors(data: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    if torch.is_tensor(data):
        return fn(data)
    if isinstance(data, Mapping):
        return type(data)(
            (key, _apply_to_tensors(value, fn)) for key, value in data.items()
        )
    if isinstance(data, tuple) and hasattr(data, "_fields"):
        return type(data)(*(_apply_to_tensors(value, fn) for value in data))
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        return type(data)(_apply_to_tensors(item, fn) for item in data)
    return data
