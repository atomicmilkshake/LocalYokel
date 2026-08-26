from __future__ import annotations
"""KV cache compression codec — pure torch, GPU-agnostic, CPU-testable.

Reference implementation of the quantize-on-store / dequantize-pages scheme that the
Grok review endorsed (see freetoken-kv-compression-integrated.md). Torch path is used
for unit tests (round-trip, parity) and as the fallback when the CUDA JIT codec
kernels are not compiled. The JIT kernels (`kernel/csrc/jit/quant_store.cu` +
`quant_dequant.cu`) mirror these exact math ops.

Storage (per layer, per K and per V):
  codes : (pages, page_size, kv_heads, head_dim)  int8/uint8 -- Lloyd-Max index
  norm  : (pages, page_size, kv_heads)            float16    -- per-block L2 norm
  rot   : (head_dim,)                             float16    -- precomputed ±1 sign flips
Read: dequant_pages(page_table) scatters referenced (page, slot) into a ONE fp16
scratch kept at pool paged geometry, so fused attention always sees a normal fp16
cache tensor. Compute dtype and storage dtype are split (pool.dtype stays fp16/bf16).
"""

import math

import torch


def qbits(v: str) -> int:
    return {"f16": 16, "tq2": 2, "tq3": 3, "tq4": 4}[v]


def is_quant(v: str) -> bool:
    return v in ("tq2", "tq3", "tq4")


def compression_ratio(v: str) -> float:
    return 16.0 / qbits(v)


def per_byte_val(v: str) -> float:
    """Bytes per head-dim element under quant vs fp16's 2.0 (used for sizing)."""
    return 2.0 / compression_ratio(v)


def codes_width(head_dim: int, bits: int) -> int:
    """Resident last-dim of the int8 codes tensor (bytes per (token, head) row).

    tq4 is nibble-packed (2 codes/byte) so width is ``head_dim // 2``. tq2/tq3
    stay one int8 per code — same packing scheme as before this perf commit.
    """
    if bits == 4:
        if head_dim % 2:
            raise ValueError(f"head_dim {head_dim} must be even to nibble-pack tq4")
        return head_dim // 2
    return head_dim


def pack_code_nibble(codes: torch.Tensor) -> torch.Tensor:
    """Pack ``(..., head_dim)`` 4-bit indices into ``(..., head_dim // 2)`` int8.

    Even index -> low nibble, odd index -> high nibble. Bit-matches
    ``kernel/csrc/jit/quant_codec.cuh::pack_code_nibble``.
    """
    hd = codes.shape[-1]
    if hd % 2:
        raise ValueError(f"head_dim {hd} must be even to nibble-pack")
    even = codes[..., 0::2].to(torch.int32) & 0x0F
    odd = codes[..., 1::2].to(torch.int32) & 0x0F
    return (even | (odd << 4)).to(torch.int8)


def unpack_code_nibble(packed: torch.Tensor, head_dim: int | None = None) -> torch.Tensor:
    """Unpack ``(..., head_dim // 2)`` nibble-packed int8 to ``(..., head_dim)`` indices.

    Bit-matches ``kernel/csrc/jit/quant_codec.cuh::unpack_code_nibble``.
    """
    b = packed.view(torch.uint8).to(torch.int32)
    even = b & 0x0F
    odd = (b >> 4) & 0x0F
    unpacked = torch.stack((even, odd), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )
    if head_dim is not None and unpacked.shape[-1] != head_dim:
        raise ValueError(
            f"unpacked last dim {unpacked.shape[-1]} != head_dim {head_dim}"
        )
    return unpacked


def require_pow2(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"head_dim {n} must be a power of two for WHT codec")
    return n


def random_sign_flip(dim: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return ((torch.rand(dim, generator=g) < 0.5).to(torch.float16) * 2 - 1)


def wht(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal fast Walsh-Hadamard along last dim (n pow2). out = (1/sqrt n)*WHT."""
    x = x.clone()
    n = x.shape[-1]
    require_pow2(n)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            a = x[..., i : i + h]
            b = x[..., i + h : i + 2 * h]
            t = a.clone()
            a.copy_(t + b)
            b.copy_(t - b)
        h *= 2
    return x / math.sqrt(n)


def iwht(x: torch.Tensor) -> torch.Tensor:
    # WHT is its own inverse up to the 1/sqrt(n) normalizer; applying wht twice
    # reconstructs x exactly (‖‖: wht(wht(x)) == x).
    return wht(x)


_LLOYD = {
    2: [-1.45, 0.0, 0.0, 1.45],
    3: [-1.8271, -0.6904, 0.0, 0.6904, 1.8271],
    # True 16-level Lloyd-Max (Max) for N(0,1), symmetric, spans +/-2.74.
    4: [-2.7393, -2.0719, -1.6213, -1.2595, -0.9453, -0.6591, -0.3893, -0.129,
        0.129, 0.3892, 0.6586, 0.9442, 1.2585, 1.622, 2.0747, 2.7359],
}


def _centroids(bits: int, head_dim: int | None = None) -> torch.Tensor:
    """Lloyd-Max centroids for the ROTATED coordinate distribution. After L2-normalize + WHT,
    coords are approx N(0, 1/sqrt(head_dim)) (WHT preserves unit energy across d coords), so
    the N(0,1) table is scaled by 1/sqrt(d) to match the real coordinate magnitude."""
    if bits not in (2, 3, 4):
        raise ValueError(bits)
    c = torch.tensor(_LLOYD[bits], dtype=torch.float32)
    if head_dim is not None:
        c = c * (1.0 / math.sqrt(head_dim))
    return c


def nearest_code(q: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """nearest Lloyd-Max centroid index per coordinate. q (..., hd), c (ncent,). -> (..., hd)."""
    # q[..., :, None] (..., hd, 1) - c (ncent,) broadcast -> (..., hd, ncent); argmin over last
    d = (q[..., :, None] - c).pow(2)
    return d.argmin(-1)


_CENTROID_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}


def _centroids_on(bits: int, head_dim: int, device: torch.device) -> torch.Tensor:
    key = (bits, head_dim, str(device))
    t = _CENTROID_CACHE.get(key)
    if t is None:
        t = _centroids(bits, head_dim).to(device=device, dtype=torch.float32)
        _CENTROID_CACHE[key] = t
    return t


def encode_block(
    x: torch.Tensor,
    bits: int,
    rot: torch.Tensor,
    codes_out: torch.Tensor,
    norm_out: torch.Tensor,
    seed: int = 0,
) -> None:
    """x (...,head_dim) fp16 -> L2-normalize + WHT-rotate + scalar-quantize.
    codes_out (...,head_dim) u8; norm_out (...) fp16 (per-block L2, used to restore scale)."""
    norms = x.norm(dim=-1).clamp_min(1e-6)  # per-block L2 before rotation
    xn = x / norms[..., None]                # unit-norm
    xr = xn * rot.to(x.dtype)              # sign flips bring mean to ~0
    W = wht(xr)                            # coords now ~ N(0, 1/sqrt(d)) with unit energy
    c = _centroids_on(bits, x.shape[-1], x.device)
    idx = nearest_code(W, c)
    codes_out.copy_(idx.to(codes_out.dtype))
    norm_out.copy_(norms.to(norm_out.dtype))


def dequant_codes(
    codes: torch.Tensor, norm: torch.Tensor, rot: torch.Tensor, bits: int
) -> torch.Tensor:
    """codes (...,head_dim) + per-block norm -> reconstructed fp16 (inverse of encode_block)."""
    hd = codes.shape[-1]
    c = _centroids_on(bits, hd, codes.device)
    q = c[codes.to(torch.int32).clamp(0, len(c) - 1)]  # (...,hd) in rotated unit-norm space
    q = iwht(q)                    # inverse rotation (WHT self-inverse)
    q = q * rot.to(q.dtype)        # inverse sign flips
    return (q * norm[..., None]).to(torch.float16)   # restore per-block scale


def align_dim(hd: int) -> int:
    return require_pow2(hd)