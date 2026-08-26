"""Isolated JIT of CUDA tq4 quant_store + quant_dequant (Phase 3)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")
sys.path.insert(0, str(ROOT / "scripts"))
from _vcvars_env import apply_vcvars  # noqa: E402

apply_vcvars()
os.environ["PYTHONPATH"] = str(ROOT / "python")
sys.path.insert(0, str(ROOT / "python"))
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("WDK_ROOT", r"S:\WADK102")

import torch  # noqa: E402

from freetoken.kernel.quant_store import quant_dequant_pages, quant_store_side  # noqa: E402

dev = torch.device("cuda")
hd, nh, t = 256, 4, 8
src = torch.zeros(t, nh, hd, device=dev, dtype=torch.float16)
codes = torch.zeros(t, nh, hd // 2, device=dev, dtype=torch.int8)
norms = torch.zeros(t, nh, device=dev, dtype=torch.float16)
rot = torch.ones(hd, device=dev, dtype=torch.float16)
idx = torch.arange(t, device=dev, dtype=torch.int64)
quant_store_side(codes=codes, norms=norms, indices=idx, src=src, rot=rot, bits=4)
scratch = torch.zeros(t, 1, nh, hd, device=dev, dtype=torch.float16)
cd = codes.view(t, 1, nh, hd // 2)
nm = norms.view(t, 1, nh)
pages = torch.zeros(1, device=dev, dtype=torch.int64)
quant_dequant_pages(scratch=scratch, codes=cd, norms=nm, rot=rot, page_ids=pages, bits=4)
print("TQ4_CUDA_KERNELS_OK", "store+dequant", src.shape, flush=True)
