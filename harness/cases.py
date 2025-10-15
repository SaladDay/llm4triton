"""
解析案例配置文件（manifest.yaml），实例化PyTorch模块，生成测试数据
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml
from torch import nn

from .data import agent_provide_inputs, agent_provide_weights

"""
load_case_manifest:
读取并解析 cases/{case_name}/manifest.yaml：
返回 CaseManifest 对象，包含：
- case_name: 案例名称
- model_class: 模型类名（如 "SimpleLinear"）
- model_kwargs: 模型构造参数
- inputs: 输入数据规格
- weights: 权重配置
- device: 目标设备

make_case_forward
返回一个零参数的可调用对象.
调用其会自动完成：
# 1. 从缓存加载输入数据
# 2. 实例化模型
# 3. 从缓存加载权重
# 4. 执行前向传播
# 5. 返回输出结果

resolve_case_model:
得到模型类

"""

__all__ = [
    "load_case_manifest",
    "make_case_forward",
    "resolve_case_model",
]


CASES_ROOT = Path("cases")
DEFAULT_MODEL_MODULE = "model"


@dataclass(frozen=True)
class CaseManifest:
    case_name: str
    display_name: str
    description: str | None
    device: str | None
    model_module: str
    model_class: str
    model_kwargs: dict[str, Any]
    inputs: Mapping[str, Any]
    weights: Mapping[str, Any]
    raw: Mapping[str, Any]


def load_case_manifest(case_name: str) -> CaseManifest:
    data = _load_manifest_dict(case_name)
    model_cfg = data.get("model", {})
    model_module = model_cfg.get("module", data.get("model_module", DEFAULT_MODEL_MODULE))
    model_class = model_cfg.get("class_name", data.get("model_class"))
    if model_class is None:
        raise ValueError(
            f"Manifest for case '{case_name}' must specify model.class_name (got: {data})"
        )
    model_kwargs = model_cfg.get("init_kwargs", {})
    inputs = data.get("inputs", {})
    weights = data.get("weights", {})
    return CaseManifest(
        case_name=case_name,
        display_name=data.get("name", case_name),
        description=data.get("description"),
        device=data.get("device"),
        model_module=model_module,
        model_class=model_class,
        model_kwargs=dict(model_kwargs),
        inputs=dict(inputs),
        weights=dict(weights),
        raw=data,
    )


def resolve_case_model(manifest: CaseManifest) -> tuple[type[nn.Module], Any]:
    module = _import_case_module(manifest.case_name, manifest.model_module)
    try:
        model_cls = getattr(module, manifest.model_class)
    except AttributeError as exc:
        raise AttributeError(
            f"Model class '{manifest.model_class}' not found in "
            f"cases.{manifest.case_name}.{manifest.model_module}"
        ) from exc
    if not issubclass(model_cls, nn.Module):
        raise TypeError(
            f"Resolved model '{manifest.model_class}' is not a torch.nn.Module subclass"
        )
    return model_cls, module


def make_case_forward(
    case_name: str,
    *,
    device: torch.device | str | None = None,
) -> Callable[[], Any]:
    manifest = load_case_manifest(case_name)
    model_cls, module = resolve_case_model(manifest)
    model_kwargs = dict(manifest.model_kwargs)
    target_device = _normalize_device(device or manifest.device)

    input_factory = _build_input_factory(case_name, manifest.inputs)
    weight_factory = _build_weight_factory(case_name, manifest.weights, model_cls, module, model_kwargs)

    def _forward() -> Any:
        payload = agent_provide_inputs(
            case_name,
            input_factory,
            device=target_device,
        )
        args = tuple(payload.get("args", ()))
        kwargs = dict(payload.get("kwargs", {}))

        model = model_cls(**model_kwargs).to(target_device)
        weights = agent_provide_weights(
            case_name,
            weight_factory,
            device=target_device,
        )
        model.load_state_dict(weights)
        return model(*args, **kwargs)

    return _forward


def _build_input_factory(
    case_name: str,
    inputs_spec: Mapping[str, Any],
) -> Callable[[], dict[str, Any]]:
    if not inputs_spec:
        raise ValueError(f"Case '{case_name}' must define an 'inputs' section in its manifest.")

    args_spec = inputs_spec.get("args", [])
    kwargs_spec = inputs_spec.get("kwargs", {})

    def _factory() -> dict[str, Any]:
        args = tuple(_materialize_value(entry) for entry in args_spec)
        kwargs = {name: _materialize_value(spec) for name, spec in kwargs_spec.items()}
        return {"args": args, "kwargs": kwargs}

    return _factory


def _build_weight_factory(
    case_name: str,
    weights_spec: Mapping[str, Any],
    model_cls: type[nn.Module],
    module: Any,
    model_kwargs: Mapping[str, Any],
) -> Callable[[], Mapping[str, torch.Tensor]]:
    provider_name = weights_spec.get("function")
    provider_kwargs = weights_spec.get("call_kwargs", {})

    if provider_name:
        try:
            provider = getattr(module, provider_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Weight provider '{provider_name}' not found in "
                f"cases.{case_name}.{module.__name__.split('.')[-1]}"
            ) from exc

        def _factory() -> Mapping[str, torch.Tensor]:
            weights = provider(**provider_kwargs)
            if not isinstance(weights, Mapping):
                raise TypeError(
                    f"Weight provider '{provider_name}' for case '{case_name}' "
                    f"must return a Mapping compatible with state_dict()."
                )
            return weights

        return _factory

    def _default_factory() -> Mapping[str, torch.Tensor]:
        model = model_cls(**model_kwargs)
        return model.state_dict()

    return _default_factory


def _materialize_value(spec: Any) -> Any:
    if isinstance(spec, Mapping):
        if "value" in spec:
            return spec["value"]
        if "shape" in spec:
            return _build_tensor(spec)
        if "items" in spec:
            seq = [_materialize_value(item) for item in spec.get("items", [])]
            container = spec.get("container", "list")
            if container == "tuple":
                return tuple(seq)
            if container == "list":
                return list(seq)
            raise ValueError(f"Unsupported container '{container}' in spec {spec}")
        if "kwargs" in spec:
            return {key: _materialize_value(value) for key, value in spec["kwargs"].items()}
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return type(spec)(_materialize_value(item) for item in spec)
    return spec


def _build_tensor(spec: Mapping[str, Any]) -> torch.Tensor:
    shape = spec.get("shape")
    if shape is None:
        raise ValueError(f"Tensor spec requires 'shape': {spec}")
    dtype = _resolve_dtype(spec.get("dtype", "float32"))
    distribution = spec.get("distribution", "normal").lower()

    if distribution == "normal":
        mean = float(spec.get("mean", 0.0))
        std = float(spec.get("std", 1.0))
        tensor = torch.randn(*shape, dtype=dtype)
        if std != 1.0:
            tensor = tensor * std
        if mean != 0.0:
            tensor = tensor + mean
    elif distribution == "uniform":
        low = float(spec.get("low", 0.0))
        high = float(spec.get("high", 1.0))
        tensor = torch.empty(*shape, dtype=dtype).uniform_(low, high)
    elif distribution == "ones":
        tensor = torch.ones(*shape, dtype=dtype)
    elif distribution == "zeros":
        tensor = torch.zeros(*shape, dtype=dtype)
    elif distribution == "full":
        value = float(spec.get("value", 0.0))
        tensor = torch.full(shape, value, dtype=dtype)
    else:
        raise ValueError(f"Unsupported distribution '{distribution}' in spec {spec}")

    if spec.get("requires_grad", False):
        tensor.requires_grad_(True)
    return tensor


def _resolve_dtype(name: str) -> torch.dtype:
    lookup = {
        "float16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "int64": torch.int64,
        "long": torch.int64,
        "int32": torch.int32,
        "int": torch.int32,
        "int16": torch.int16,
        "short": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    key = name.lower()
    try:
        return lookup[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{name}' in manifest.") from exc


def _normalize_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        default = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(default)
    return torch.device(device)


@lru_cache(maxsize=None)
def _load_manifest_dict(case_name: str) -> Mapping[str, Any]:
    manifest_path = CASES_ROOT / case_name / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found for case '{case_name}' at {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Manifest for case '{case_name}' must be a mapping (got {type(data)})")
    return data


@lru_cache(maxsize=None)
def _import_case_module(case_name: str, module_name: str):
    full_name = f"cases.{case_name}.{module_name}"
    return importlib.import_module(full_name)
