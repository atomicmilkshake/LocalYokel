from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from .utils import KERNEL_PATH, KernelConfig, load_jit, make_cpp_args

_JIT_INCLUDE = [str(KERNEL_PATH / "jit")]

if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

DEFAULT_QUANT_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_quant_store_module(
    head_dim: int,
    bits: int,
    *,
    config: KernelConfig = DEFAULT_QUANT_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(head_dim, bits, *config)
    return load_jit(
        "quant_store",
        *args,
        cuda_files=["quant_store.cu"],
        cuda_wrappers=[("launch", f"QuantStoreKernel<{args}>::run")],
        extra_include_paths=_JIT_INCLUDE,
    )


@functools.cache
def _jit_quant_dequant_module(
    head_dim: int,
    bits: int,
    *,
    config: KernelConfig = DEFAULT_QUANT_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(head_dim, bits, *config)
    return load_jit(
        "quant_dequant",
        *args,
        cuda_files=["quant_dequant.cu"],
        cuda_wrappers=[("launch", f"QuantDequantKernel<{args}>::run")],
        extra_include_paths=_JIT_INCLUDE,
    )


def _codes_width(head_dim: int, bits: int) -> int:
    """Resident last-dim of the int8 codes tensor. tq4 = hd//2 nibble pack."""
    return head_dim // 2 if int(bits) == 4 else head_dim


def quant_store_side(
    codes: torch.Tensor,
    norms: torch.Tensor,
    indices: torch.Tensor,
    src: torch.Tensor,
    rot: torch.Tensor,
    bits: int,
) -> None:
    """Encode one side (K or V). src (T, H, hd) fp16; codes (R, H, codes_width) int8.

    ``codes_width`` is ``hd // 2`` when bits==4 (nibble packed), else ``hd``.
    Kernel strides (head_stride/slot_stride) are in that packed unit.
    """
    import torch

    src = src.contiguous()
    if src.dtype != torch.float16:
        src = src.to(torch.float16)
    codes = codes.contiguous()
    norms = norms.contiguous()
    rot = rot.contiguous().to(torch.float16)
    if indices.dtype not in (torch.int32, torch.int64):
        indices = indices.to(torch.int64)
    indices = indices.contiguous()
    head_dim = int(src.shape[-1])
    cw = _codes_width(head_dim, int(bits))
    if int(codes.shape[-1]) != cw:
        raise ValueError(
            f"codes last dim {int(codes.shape[-1])} != codes_width {cw} (bits={bits})"
        )
    module = _jit_quant_store_module(head_dim, int(bits))
    module.launch(codes, norms, indices, src, rot)


def quant_dequant_pages(
    scratch: torch.Tensor,
    codes: torch.Tensor,
    norms: torch.Tensor,
    rot: torch.Tensor,
    page_ids: torch.Tensor,
    bits: int,
) -> None:
    """Dequant referenced pages into scratch (P, ps, H, hd) fp16.

    codes is (P, ps, H, codes_width) int8; codes_width = hd//2 when bits==4.
    Kernel page/slot/head strides are in packed code-bytes, not fp16 elements.
    """
    import torch

    scratch = scratch.contiguous()
    codes = codes.contiguous()
    norms = norms.contiguous()
    rot = rot.contiguous().to(torch.float16)
    page_ids = page_ids.reshape(-1).contiguous()
    if page_ids.dtype not in (torch.int32, torch.int64):
        page_ids = page_ids.to(torch.int64)
    if page_ids.numel() == 0:
        return
    head_dim = int(scratch.shape[-1])
    cw = _codes_width(head_dim, int(bits))
    if int(codes.shape[-1]) != cw:
        raise ValueError(
            f"codes last dim {int(codes.shape[-1])} != codes_width {cw} (bits={bits})"
        )
    module = _jit_quant_dequant_module(head_dim, int(bits))
    module.launch(scratch, codes, norms, rot, page_ids)
