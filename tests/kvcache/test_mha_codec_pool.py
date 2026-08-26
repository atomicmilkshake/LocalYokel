"""MHAKVCache codec path: scratch geometry, vectorized materialize, rebuild, MHA parity."""

from __future__ import annotations

import torch

from freetoken.distributed import set_tp_info, try_get_tp_info


def _init_tp() -> None:
    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)


def _pool(kv_quant="tq4", *, device="cpu", pages=4, page_size=16, layers=2, heads=2, hd=64):
    from freetoken.kvcache.mha_pool import MHAKVCache

    _init_tp()
    return MHAKVCache(
        num_kv_heads=heads, num_layers=layers, head_dim=hd, num_pages=pages,
        page_size=page_size, dtype=torch.float16, device=torch.device(device),
        kv_quant=kv_quant,
    )


def test_kv_quant_cli_default_is_f16():
    from freetoken.engine.config import EngineConfig
    from freetoken.server.args import ServerArgs

    assert EngineConfig.__dataclass_fields__["kv_quant"].default == "f16"
    assert ServerArgs.__dataclass_fields__["kv_quant"].default == "f16"


def test_codec_scratch_is_paged_not_layer_indexed():
    pool = _pool()
    k = pool.k_cache(0)
    v = pool.v_cache(1)
    assert k.shape == (4, 16, 2, 64)
    assert v.shape == (4, 16, 2, 64)
    assert k.data_ptr() == pool._scratch[0].data_ptr()
    assert v.data_ptr() == pool._scratch[1].data_ptr()
    assert k.data_ptr() != v.data_ptr()


def test_tq4_resident_codes_are_nibble_packed_half_width():
    from freetoken.kvcache.mha_pool import _per_int_bytes

    pool = _pool("tq4", hd=64)
    assert pool._codes.shape[-1] == 32
    assert pool._codes_width == 32
    assert _per_int_bytes("tq4") == 0.5


def test_tq2_tq3_codes_stay_int8_full_width():
    for q in ("tq2", "tq3"):
        pool = _pool(q, hd=64)
        assert pool._codes.shape[-1] == 64
        assert pool._codes_width == 64


def test_pack_unpack_nibble_roundtrip():
    from freetoken.kvcache.quant_codec import pack_code_nibble, unpack_code_nibble

    idx = torch.arange(16, dtype=torch.int32).repeat(4)  # 64 codes, 0..15
    idx = idx.view(1, 1, 64)
    packed = pack_code_nibble(idx)
    assert packed.shape == (1, 1, 32)
    assert packed.dtype == torch.int8
    got = unpack_code_nibble(packed, 64)
    assert torch.equal(got, idx)


def test_codec_store_materialize_roundtrip_matches_encode_block():
    from freetoken.kvcache.quant_codec import (
        dequant_codes,
        encode_block,
        qbits,
        unpack_code_nibble,
    )

    pool = _pool(page_size=8, pages=3)
    T, H, hd = 8, 2, 64
    k = torch.randn(T, H, hd, dtype=torch.float16)
    v = torch.randn(T, H, hd, dtype=torch.float16)
    loc = torch.arange(T, dtype=torch.int64)
    pool.store_kv(k, v, loc, layer_id=0)
    page_table = torch.tensor([0], dtype=torch.int64)
    pool.materialize(0, page_table)

    idx = unpack_code_nibble(pool._codes[0][0][:T], hd)
    recon_k = dequant_codes(idx, pool._norms[0][0][:T], pool._rot, qbits("tq4"))
    got = pool.k_cache(0).reshape(-1, H, hd)[:T]
    assert torch.allclose(got.float(), recon_k.float(), atol=1e-3, rtol=1e-3)

    code = torch.empty(k.shape, dtype=torch.int8)
    nm = torch.empty(k.shape[:-1], dtype=torch.float16)
    encode_block(k, qbits("tq4"), pool._rot, code, nm)
    oracle = dequant_codes(code, nm, pool._rot, qbits("tq4"))
    assert torch.allclose(got.float(), oracle.float(), atol=1e-3, rtol=1e-3)


def test_codec_rebuild_snapshots_dtype_preserves_identity():
    pool = _pool(kv_quant="f16")
    pool._compute_dtype = torch.bfloat16
    obj = id(pool)
    pool.rebuild(7)
    assert id(pool) == obj
    assert pool._kv_buffer.dtype == torch.bfloat16
    assert pool._kv_buffer.shape[2] == 7

    cpool = _pool(kv_quant="tq4")
    cid = id(cpool)
    rot = cpool._rot
    cpool.rebuild(5)
    assert id(cpool) == cid
    assert cpool._codes.shape[2] == 5 * 16
    assert cpool._codes.shape[-1] == 32
    assert cpool._scratch.shape[1] == 5
    assert cpool._rot is rot


def _sdpa(q, k, v):
    # q (Tq, Hq, d), k/v (Tk, Hkv, d) — GQA repeat if needed
    Tq, Hq, d = q.shape
    Tk, Hkv, _ = k.shape
    if Hq != Hkv:
        rep = Hq // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scale = d ** -0.5
    att = torch.softmax((q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2)) * scale, dim=-1)
    return (att @ v.transpose(0, 1)).transpose(0, 1)


def test_codec_mha_attention_mae_within_0_1_of_fp16():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pages, ps, layers, heads, hd = 2, 16, 1, 2, 64
    qnt = _pool("tq4", device=str(device), pages=pages, page_size=ps, layers=layers, heads=heads, hd=hd)
    T, Hq = 16, 4
    g = torch.Generator().manual_seed(0)
    k = torch.randn(T, heads, hd, dtype=torch.float16, generator=g)
    v = torch.randn(T, heads, hd, dtype=torch.float16, generator=g)
    q = torch.randn(T, Hq, hd, dtype=torch.float16, generator=g)
    k, v, q = k.to(device), v.to(device), q.to(device)
    loc = torch.arange(T, dtype=torch.int64, device=device)
    qnt.store_kv(k, v, loc, 0)
    qnt.materialize(0, torch.tensor([0], dtype=torch.int64, device=device))
    k_q = qnt.k_cache(0).reshape(-1, heads, hd)[:T]
    v_q = qnt.v_cache(0).reshape(-1, heads, hd)[:T]
    o_ref = _sdpa(q.float(), k.float(), v.float())
    o_q = _sdpa(q.float(), k_q.float(), v_q.float())
    mae = (o_q - o_ref).abs().mean().item()
    assert mae < 0.1, f"mean |o_q-o_ref|={mae} not within 0.1 of fp16"
