"""Copy repaired TurboQuant sources into the merged tree and wire materialize + CLI."""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")
PKG = Path(r"C:\Users\owenm\AppData\Local\FreeToken\venv\Lib\site-packages\freetoken")

COPIES = [
    ("kvcache/quant_codec.py", "python/freetoken/kvcache/quant_codec.py"),
    ("kvcache/mha_pool.py", "python/freetoken/kvcache/mha_pool.py"),
    ("kernel/quant_store.py", "python/freetoken/kernel/quant_store.py"),
    ("kernel/csrc/jit/quant_codec.cuh", "python/freetoken/kernel/csrc/jit/quant_codec.cuh"),
    ("kernel/csrc/jit/quant_store.cu", "python/freetoken/kernel/csrc/jit/quant_store.cu"),
    ("kernel/csrc/jit/quant_dequant.cu", "python/freetoken/kernel/csrc/jit/quant_dequant.cu"),
]


def _insert_after(text: str, needle: str, insert: str) -> str:
    if needle not in text:
        raise SystemExit(f"needle not found: {needle[:80]!r}")
    if insert.strip() in text:
        return text
    return text.replace(needle, needle + insert, 1)


def _replace_once(text: str, old: str, new: str) -> str:
    if old not in text:
        raise SystemExit(f"pattern not found: {old[:120]!r}")
    return text.replace(old, new, 1)


def main() -> None:
    for src_rel, dst_rel in COPIES:
        src = PKG / src_rel
        dst = ROOT / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print("copied", src_rel, "->", dst_rel)

    # EngineConfig.kv_quant (default f16 — do not default-on)
    cfg = ROOT / "python/freetoken/engine/config.py"
    t = cfg.read_text(encoding="utf-8")
    if "kv_quant:" not in t:
        t = _insert_after(
            t,
            "    page_size: int = 1\n",
            "    # KV storage codec for dense/full-attention pools only. f16 (default) is the\n"
            "    # safe path. tq4 is TurboQuant nibble-packed 4-bit KV (WHT + Lloyd-Max).\n"
            "    # DSV4/MLA/DSA/BSA ignore this.\n"
            "    kv_quant: str = \"f16\"\n",
        )
        cfg.write_text(t, encoding="utf-8")
        print("patched engine/config.py kv_quant")

    # Factory: pass kv_quant into MHAKVCache
    init = ROOT / "python/freetoken/kvcache/__init__.py"
    t = init.read_text(encoding="utf-8")
    old = """    return create_kvcache_pool(
        model_config=model_config,
        num_pages=num_pages + 1,  # +1 for dummy page
        page_size=config.page_size,
        num_swa_tokens=num_swa_tokens,
        device=device,
        dtype=dtype,
    )
"""
    new = """    return create_kvcache_pool(
        model_config=model_config,
        num_pages=num_pages + 1,  # +1 for dummy page
        page_size=config.page_size,
        num_swa_tokens=num_swa_tokens,
        device=device,
        dtype=dtype,
        kv_quant=getattr(config, "kv_quant", "f16") or "f16",
    )
"""
    if "kv_quant=getattr(config" not in t:
        t = _replace_once(t, old, new)
    old_sig = """def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_swa_tokens: int | None = None,
) -> BaseKVCachePool:
"""
    new_sig = """def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
    num_swa_tokens: int | None = None,
    kv_quant: str = "f16",
) -> BaseKVCachePool:
"""
    if "kv_quant: str = \"f16\"" not in t.split("def create_kvcache_pool")[1][:500]:
        t = _replace_once(t, old_sig, new_sig)
    old_mha = """    return MHAKVCache(
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        layer_ids=layer_ids,
    )
"""
    new_mha = """    return MHAKVCache(
        num_kv_heads=num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=head_dim,
        device=device,
        dtype=dtype,
        layer_ids=layer_ids,
        kv_quant=kv_quant,
    )
"""
    if "kv_quant=kv_quant" not in t:
        t = _replace_once(t, old_mha, new_mha)
    init.write_text(t, encoding="utf-8")
    print("patched kvcache/__init__.py factory")

    # CLI
    args = ROOT / "python/freetoken/server/args.py"
    t = args.read_text(encoding="utf-8")
    if "--kv-quant" not in t:
        t = _insert_after(
            t,
            """    parser.add_argument(
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )
""",
            """
    parser.add_argument(
        "--kv-quant",
        dest="kv_quant",
        choices=["f16", "tq2", "tq3", "tq4"],
        default=getattr(ServerArgs, "kv_quant", "f16"),
        help="Dense/full-attention KV storage: f16 (default) or TurboQuant tq2/tq3/tq4. "
        "tq4 is packed 4-bit (WHT + Lloyd-Max). Ignored for DSV4/MLA/DSA/BSA.",
    )
""",
        )
        args.write_text(t, encoding="utf-8")
        print("patched server/args.py --kv-quant")

    # Attention backends: store then materialize
    patches = {
        "python/freetoken/attention/fa.py": (
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n"
            "        return _fa_sgl_impl(\n",
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n"
            "        mat = getattr(self.kvcache, \"materialize\", None)\n"
            "        if callable(mat):\n"
            "            mat(layer_id, metadata.page_table)\n"
            "        return _fa_sgl_impl(\n",
        ),
        "python/freetoken/attention/fi.py": (
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n"
            "        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))\n",
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n"
            "        mat = getattr(self.kvcache, \"materialize\", None)\n"
            "        if callable(mat):\n"
            "            mat(layer_id, metadata.page_table)\n"
            "        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))\n",
        ),
        "python/freetoken/attention/triton.py": (
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n\n"
            "        k_raw = self.kvcache.k_cache(layer_id)\n",
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n"
            "        mat = getattr(self.kvcache, \"materialize\", None)\n"
            "        if callable(mat):\n"
            "            pt = getattr(metadata, \"page_table\", None)\n"
            "            if pt is None:\n"
            "                pt = getattr(metadata, \"indices\", None)\n"
            "            if pt is not None:\n"
            "                mat(layer_id, pt)\n"
            "        k_raw = self.kvcache.k_cache(layer_id)\n",
        ),
        "python/freetoken/attention/trtllm.py": (
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n"
            "        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))\n",
            "        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)\n"
            "        mat = getattr(self.kvcache, \"materialize\", None)\n"
            "        if callable(mat):\n"
            "            mat(layer_id, metadata.page_table)\n"
            "        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))\n",
        ),
    }
    for rel, (old, new) in patches.items():
        p = ROOT / rel
        t = p.read_text(encoding="utf-8")
        if "materialize" in t and 'mat = getattr(self.kvcache, "materialize"' in t:
            print("already wired", rel)
            continue
        p.write_text(_replace_once(t, old, new), encoding="utf-8")
        print("wired materialize", rel)


if __name__ == "__main__":
    main()
