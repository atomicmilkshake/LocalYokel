import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")
sys.path.insert(0, str(ROOT / "scripts"))
from _vcvars_env import apply_vcvars  # noqa: E402

apply_vcvars()
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT / "python")
os.environ["FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29764"
os.environ["WDK_ROOT"] = r"S:\WADK102"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")

cmd = [
    r"C:\Users\owenm\AppData\Local\FreeToken\venv\Scripts\python.exe",
    "-m",
    "freetoken.cli",
    "serve",
    "--model",
    r"V:\LLM-tmp\lyf-modelopt",
    "--kv-quant",
    "tq4",
    "--moe-backend",
    "offload",
    "--moe-cache-auto",
    "--max-prefill-length",
    "2048",
    "--port",
    "1963",
    "--cuda-graph-max-bs",
    "1",
    "--served-model-name",
    "Qwen3.6-35B-A3B-Uncensored-lyf-tq4",
]
print("running", " ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
