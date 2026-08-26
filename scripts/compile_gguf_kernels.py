"""Isolated compile of freetoken_gguf_kernels (Phase 1). Requires VS18 vcvars."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")
sys.path.insert(0, str(ROOT / "scripts"))
from _vcvars_env import apply_vcvars  # noqa: E402

apply_vcvars()
# PYTHONPATH is only read at interpreter start — also insert for in-process import.
os.environ["PYTHONPATH"] = str(ROOT / "python")
sys.path.insert(0, str(ROOT / "python"))
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
os.environ.setdefault("WDK_ROOT", r"S:\WADK102")

PY = Path(r"C:\Users\owenm\AppData\Local\FreeToken\venv\Scripts\python.exe")
if Path(sys.executable).resolve() != PY.resolve():
    print("re-exec under", PY, flush=True)
    raise SystemExit(subprocess.call([str(PY), str(Path(__file__).resolve())], env=os.environ.copy()))

cache = Path(os.environ.get("LOCALAPPDATA", "")) / "torch_extensions" / "torch_extensions" / "Cache"
for p in cache.glob("**/freetoken_gguf_kernels"):
    shutil.rmtree(p, ignore_errors=True)
    print("cleared", p)

print("sys.executable", sys.executable, flush=True)
from freetoken.kernel import gguf as g  # noqa: E402

print("gguf.__file__", getattr(g, "__file__", None), flush=True)
print("gguf._CSRC", g._CSRC, flush=True)

mod = g._module()
print("module", mod)
print("ops", [x for x in dir(mod) if "ggml" in x.lower() or "dequant" in x.lower()])
print("GGUF_KERNELS_OK")
