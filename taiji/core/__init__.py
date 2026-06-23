"""Lightweight taiji.core package exports.

Core submodules are imported lazily so status and configuration code can load
without requiring model inference dependencies such as torch.
"""

_LAZY_EXPORTS = {
    "NativeInferenceEngine": ("taiji.core.inference", "NativeInferenceEngine"),
    "CudaInferenceEngine": ("taiji.core.cuda_inference", "CudaInferenceEngine"),
    "TritonInferenceEngine": ("taiji.core.cuda_inference", "TritonInferenceEngine"),
    "InferenceRouter": ("taiji.core.cuda_inference", "InferenceRouter"),
    "is_cuda_engine_available": ("taiji.core.cuda_inference", "is_cuda_engine_available"),
    "HybridEngine": ("taiji.core.hybrid_engine", "HybridEngine"),
    "sanitize_pii": ("taiji.core.hybrid_engine", "sanitize_pii"),
    "TaijiQuantizer": ("taiji.core.quantization", "TaijiQuantizer"),
    "quantize_model": ("taiji.core.quantization", "quantize_model"),
    "NativeAgentEngine": ("taiji.core.native_agent", "NativeAgentEngine"),
    "AgentStep": ("taiji.core.native_agent", "AgentStep"),
    "AgentResult": ("taiji.core.native_agent", "AgentResult"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
