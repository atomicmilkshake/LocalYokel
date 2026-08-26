from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even, init_logger

logger = init_logger(__name__)

from .base import BaseKVCachePool
from .quant_codec import (
    codes_width,
    dequant_codes,
    encode_block,
    is_quant,
    pack_code_nibble,
    qbits,
    random_sign_flip,
    require_pow2,
    unpack_code_nibble,
)


def _per_int_bytes(q: str) -> float:
    # True packed storage (per-code bytes): tq4=0.5 (nibble, 2/byte), tq2=0.25 (4/byte),
    # tq3=0.375 (8 codes/3 bytes). int8 was 1.0 (the old ~2x). This is now the real 4-bit.
    return {2: 0.25, 3: 0.375, 4: 0.5}.get({"tq2": 2, "tq3": 3, "tq4": 4}.get(q, 0), 1.0)


def _per_norm_bytes(_q: str) -> float:
    return 2.0  # fp16 per-(token, head) L2 norm


def _kv_quant_value(config) -> str:
    v = "f16"
    if hasattr(config, "kv_quant"):
        v = getattr(config, "kv_quant") or v
    elif hasattr(config, "engine") and hasattr(config.engine, "kv_quant"):
        v = getattr(config.engine, "kv_quant") or v
    return v if v in ("tq2", "tq3", "tq4") else "f16"


def _qv(kv_quant: str) -> str:
    return kv_quant if kv_quant in ("tq2", "tq3", "tq4") else "f16"


def _local_kv_heads(spec, config) -> int:
    tp = getattr(getattr(config, "tp_info", None), "size", 1) or 1
    return div_even(spec.num_kv_heads, tp, allow_replicate=True)


class MHAKVCache(BaseKVCachePool):
    """Dense/full-attention key-value cache.

    Advanced KV compression (Grok-reviewed design). When kv_quant is a codec
    (tq2/tq3/tq4), the pool stores Lloyd-Max WHT codes + per-head L2 norms.
    ``materialize(layer, page_table)`` dequantizes ONLY the referenced pages into a
    one-layer fp16 scratch at pool paged geometry ``(P, ps, H, hd)``. ``k_cache`` /
    ``v_cache`` return that scratch (not a layer-indexed slice — scratch has no
    layer axis). Fused kernels always see plain fp16 of pool geometry.

    Default f16 (safe). Codec is MHA/full only; subclasses may set ``_SUPPORT_CODEC``.
    """

    _SUPPORT_CODEC = True

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
        kv_quant: str = "f16",
    ) -> None:
        self._num_layers = num_layers
        self._local_kv_heads = div_even(num_kv_heads, get_tp_info().size, allow_replicate=True)
        self._head_dim = head_dim
        self._page_size = page_size if page_size else 1
        self._num_pages = num_pages
        self._device = device
        self._compute_dtype = dtype
        if not getattr(type(self), "_SUPPORT_CODEC", False):
            kv_quant = "f16"
        self._kv_quant = _qv(kv_quant)
        self._codec = self._kv_quant != "f16"

        if layer_ids is None:
            self._num_storage_layers = num_layers
            self._layer_map = None
        else:
            self._num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, g in enumerate(layer_ids):
                if g < 0 or g >= num_layers:
                    raise ValueError(f"KV layer id {g} outside [0, {num_layers})")
                layer_map[g] = dense
            self._layer_map = layer_map

        self._storage_shape = (num_pages * self._page_size, self._local_kv_heads, head_dim)
        self._codes = self._norms = self._scratch = self._rot = None
        self._kv_buffer = self._k_buffer = self._v_buffer = None
        if not self._codec:
            self._kv_buffer = torch.empty(
                (2, self._num_storage_layers, num_pages, self._page_size,
                 self._local_kv_heads, head_dim),
                device=device, dtype=dtype,
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
        else:
            require_pow2(head_dim)
            self._codes_width = codes_width(head_dim, qbits(self._kv_quant))
            self._codes = torch.zeros(
                (2, self._num_storage_layers, num_pages * self._page_size,
                 self._local_kv_heads, self._codes_width),
                device=device, dtype=torch.int8,
            )
            self._norms = torch.zeros(
                (2, self._num_storage_layers, num_pages * self._page_size, self._local_kv_heads),
                device=device, dtype=torch.float16,
            )
            # One-layer scratch: (side, P, ps, H, hd) — NO layer axis.
            self._scratch = torch.zeros(
                (2, num_pages, self._page_size, self._local_kv_heads, head_dim),
                device=device, dtype=torch.float16,
            )
            self._rot = random_sign_flip(head_dim, 0).to(device)
        self._cuda_codec_ok: bool | None = None
        self._cuda_codec_err: str | None = None

    def _snap_dtype(self) -> torch.dtype:
        return self._compute_dtype

    def _rebuild_geometry(self, num_pages: int, dtype: torch.dtype) -> None:
        self._num_pages = num_pages
        self._storage_shape = (num_pages * self._page_size, self._local_kv_heads, self._head_dim)
        if not self._codec:
            self._kv_buffer = torch.empty(
                (2, self._num_storage_layers, num_pages, self._page_size,
                 self._local_kv_heads, self._head_dim),
                device=self._device, dtype=dtype,
            )
            self._k_buffer = self._kv_buffer[0]
            self._v_buffer = self._kv_buffer[1]
            self._codes = self._norms = self._scratch = None
        else:
            self._kv_buffer = self._k_buffer = self._v_buffer = None
            self._codes = torch.zeros(
                (2, self._num_storage_layers, num_pages * self._page_size,
                 self._local_kv_heads, self._codes_width),
                device=self._device, dtype=torch.int8,
            )
            self._norms = torch.zeros(
                (2, self._num_storage_layers, num_pages * self._page_size, self._local_kv_heads),
                device=self._device, dtype=torch.float16,
            )
            self._scratch = torch.zeros(
                (2, num_pages, self._page_size, self._local_kv_heads, self._head_dim),
                device=self._device, dtype=torch.float16,
            )

    def rebuild(self, num_pages: int) -> None:
        dtype = self._snap_dtype()
        rot = self._rot
        self._k_buffer = self._v_buffer = self._kv_buffer = None
        self._codes = self._norms = self._scratch = None
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch.cuda.empty_cache()
        self._rebuild_geometry(num_pages, dtype)
        if self._codec:
            self._rot = rot if rot is not None else random_sign_flip(self._head_dim, 0).to(self._device)

    def rebuild_from_config(self, config, num_pages: int, *, num_swa_pages: int | None = None) -> None:
        self.rebuild(num_pages + 1)

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        q = _kv_quant_value(config)
        if not is_quant(q):
            per_token = sum(
                spec_kv_bytes_per_token(spec, config)
                for spec in config.model_config.kv_cache_group_specs() if not spec.is_swa)
            return per_token * config.page_size, 0, config.page_size, 0

        page = config.page_size
        cache_per_page = 0
        scratch_hd = scratch_h = 0
        for spec in config.model_config.kv_cache_group_specs():
            if spec.is_swa:
                continue
            hd = spec.head_dim
            kw = _local_kv_heads(spec, config)
            scratch_hd, scratch_h = hd, kw
            # resident codes (int8) + norms (fp16) for K and V, all layers — per page
            cache_per_page += int(
                2 * (kw * hd * _per_int_bytes(q) + kw * _per_norm_bytes(q)) * spec.num_layers * page
            )
        # one-layer fp16 K+V scratch scales with P → cache_per_page, not fixed
        cache_per_page += int(2 * scratch_h * scratch_hd * 2 * page)
        return cache_per_page, 0, page, 0

    def unit_bytes(self) -> tuple[int, int]:
        tok = max(1, self._num_pages * self._page_size)
        if not self._codec:
            buf = self._kv_buffer
            return int(buf.numel() * buf.element_size()) // tok, 0
        codes_b = int(self._codes.numel() * self._codes.element_size())
        norms_b = int(self._norms.numel() * self._norms.element_size())
        scratch_b = int(self._scratch.numel() * self._scratch.element_size())
        return (codes_b + norms_b + scratch_b) // tok, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        dense = self._dense(index)
        if not self._codec:
            return self._k_buffer[dense]
        # scratch[side] is (P, ps, H, hd) — do not bind layer idx
        return self._scratch[0]

    def v_cache(self, index: int) -> torch.Tensor:
        dense = self._dense(index)
        if not self._codec:
            return self._v_buffer[dense]
        return self._scratch[1]

    def store_kv(self, k, v, out_loc, layer_id):
        dense = self._dense(layer_id)
        if not self._codec:
            from freetoken.kernel import store_cache
            store_cache(
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                indices=out_loc, k=k, v=v,
            )
            return
        self._encode(self._codes[0][dense], self._norms[0][dense], k, out_loc)
        self._encode(self._codes[1][dense], self._norms[1][dense], v, out_loc)

    def _encode(self, codes, norms, x, out_loc):
        """x (T,H,hd) or (T,H*hd) -> resident codes at token offsets.

        Resident last-dim is ``codes_width`` (hd//2 nibble-packed for tq4, hd
        int8 for tq2/tq3). One (token, head) row per store.cu warp: scatter by
        ``out_loc[t]``.
        """
        nh, hd = self._local_kv_heads, self._head_dim
        x = x.reshape(-1, nh, hd).contiguous()
        if x.numel() == 0:
            return
        loc = out_loc.to(device=self._device, dtype=torch.int64).reshape(-1)
        if self._try_quant_store(codes, norms, x, loc):
            return
        bits = qbits(self._kv_quant)
        code = torch.empty(x.shape, dtype=torch.int8, device=self._device)
        nm = torch.empty(x.shape[:-1], dtype=torch.float16, device=self._device)
        encode_block(x, bits, self._rot, code, nm)
        if bits == 4:
            code = pack_code_nibble(code)
        codes[loc] = code
        norms[loc] = nm

    def _try_quant_store(self, codes, norms, x, loc) -> bool:
        if self._device.type != "cuda" or self._cuda_codec_ok is False:
            return False
        try:
            from freetoken.kernel.quant_store import quant_store_side
            quant_store_side(
                codes=codes, norms=norms, indices=loc, src=x,
                rot=self._rot, bits=qbits(self._kv_quant),
            )
            self._cuda_codec_ok = True
            return True
        except Exception as e:
            self._cuda_codec_ok = False
            self._cuda_codec_err = repr(e)
            logger.error("CUDA tq codec store JIT/launch failed: %s", e)
            return False

    def materialize(self, layer_id: int, page_table: torch.Tensor) -> None:
        if not self._codec:
            return
        with torch.no_grad():
            dense = self._dense(layer_id)
            pages = page_table.reshape(-1).to(device=self._device, dtype=torch.int64)
            capturing = False
            if self._device.type == "cuda":
                try:
                    capturing = bool(torch.cuda.is_current_stream_capturing())
                except Exception:
                    capturing = False
            # Graph capture forbids boolean index, unique(), and GPU→CPU .max().
            # CUDA kernel already skips page_id < 0; map OOB/dummy to -1, keep shape.
            if capturing or self._cuda_codec_ok:
                loc = pages
                if self._page_size > 1:
                    loc = torch.where(
                        loc >= 0,
                        torch.div(loc, self._page_size, rounding_mode="floor"),
                        loc,
                    )
                loc = torch.where(
                    loc >= self._num_pages, loc.new_full(loc.shape, -1), loc
                )
                if self._cuda_materialize(dense, loc):
                    return
                if capturing:
                    raise RuntimeError(
                        "CUDA tq materialize failed during CUDA graph capture: "
                        f"{self._cuda_codec_err}"
                    )
            pages = pages[pages >= 0]
            if pages.numel() == 0:
                return
            # token-level indices (FI/triton page_size=1, or ps>1 loc) → page ids
            if int(pages.max()) >= self._num_pages:
                pages = torch.div(pages, self._page_size, rounding_mode="floor")
            pages = torch.unique(pages.clamp(0, self._num_pages - 1))
            if self._cuda_materialize(dense, pages):
                return
            bits = qbits(self._kv_quant)
            cw = self._codes_width
            for side in (0, 1):
                cd = self._codes[side][dense].view(
                    self._num_pages, self._page_size, self._local_kv_heads, cw)
                nm = self._norms[side][dense].view(
                    self._num_pages, self._page_size, self._local_kv_heads)
                gathered_c = cd.index_select(0, pages)
                gathered_n = nm.index_select(0, pages)
                if bits == 4:
                    gathered_c = unpack_code_nibble(gathered_c, self._head_dim)
                dq = dequant_codes(
                    gathered_c.reshape(-1, self._head_dim),
                    gathered_n.reshape(-1),
                    self._rot,
                    bits,
                ).reshape(-1, self._page_size, self._local_kv_heads, self._head_dim)
                self._scratch[side].index_copy_(0, pages, dq)

    def _cuda_materialize(self, dense: int, pages: torch.Tensor) -> bool:
        if self._device.type != "cuda" or self._cuda_codec_ok is False:
            return False
        try:
            from freetoken.kernel.quant_store import quant_dequant_pages
            bits = qbits(self._kv_quant)
            cw = self._codes_width
            for side in (0, 1):
                cd = self._codes[side][dense].view(
                    self._num_pages, self._page_size, self._local_kv_heads, cw)
                nm = self._norms[side][dense].view(
                    self._num_pages, self._page_size, self._local_kv_heads)
                quant_dequant_pages(
                    scratch=self._scratch[side],
                    codes=cd,
                    norms=nm,
                    rot=self._rot,
                    page_ids=pages,
                    bits=bits,
                )
            self._cuda_codec_ok = True
            return True
        except Exception as e:
            self._cuda_codec_ok = False
            self._cuda_codec_err = repr(e)
            logger.error("CUDA tq codec dequant JIT/launch failed: %s", e)
            return False

    def warmup_cuda_codec(self) -> bool:
        """Compile/run store+dequant outside CUDA graph capture. Returns True if CUDA path is live."""
        if not self._codec:
            return True
        if self._device.type != "cuda":
            return False
        T = min(self._page_size, 8)
        dummy = torch.zeros(T, self._local_kv_heads, self._head_dim, device=self._device, dtype=torch.float16)
        loc = torch.arange(T, device=self._device, dtype=torch.int64)
        self._encode(self._codes[0][0], self._norms[0][0], dummy, loc)
        pages = torch.zeros(1, dtype=torch.int64, device=self._device)
        if not self._cuda_materialize(0, pages):
            gid = 0 if self._layer_map is None else self._layer_map.index(0)
            self.materialize(gid, pages)
        ok = bool(self._cuda_codec_ok)
        if ok:
            logger.info("CUDA KV codec live kv_quant=%s head_dim=%s", self._kv_quant, self._head_dim)
        else:
            logger.error("CUDA KV codec unavailable: %s", self._cuda_codec_err)
        return ok

    def require_cuda_codec(self) -> None:
        import os

        if not self._codec:
            return
        if self.warmup_cuda_codec():
            return
        allow = os.getenv("FREETOKEN_TQ4_ALLOW_TORCH", "").strip().lower() in {"1", "true", "yes", "on"}
        msg = (
            f"kv_quant={self._kv_quant} requested CUDA kernels but they did not load "
            f"({self._cuda_codec_err}). Set FREETOKEN_TQ4_ALLOW_TORCH=1 to permit the "
            "torch encode_block fallback."
        )
        if allow:
            logger.warning(msg)
            return
        raise RuntimeError(msg)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._compute_dtype if not self._codec else torch.float16

    @property
    def num_layers(self) -> int:
        return self._num_layers
