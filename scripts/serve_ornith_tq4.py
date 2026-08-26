"""Launch merged FreeToken with Ornith-1.5-35B heretic IQ3_S + TurboQuant tq4."""
from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")
PY = Path(r"C:\Users\owenm\AppData\Local\FreeToken\venv\Scripts\python.exe")
MODEL = Path(r"U:\LLM\models\Ornith-1.5-35B-A3B-heretic-ja.i1-IQ3_S.gguf")
PORT = 1963
LOG = ROOT / "scripts" / "serve_ornith_tq4.log"


def _port_free(port: int) -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        s.connect(("127.0.0.1", port))
        s.close()
        return False
    except OSError:
        return True


def main() -> None:
    if not MODEL.exists():
        raise SystemExit(f"missing model {MODEL}")
    if not _port_free(PORT):
        print(f"port {PORT} already in use — not launching another server")
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "python")
    env["FREETOKEN_DISABLE_KERNEL_CACHE_VERSION_CHECK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    cmd = [
        str(PY),
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        str(MODEL),
        "--kv-quant",
        "tq4",
        "--moe-backend",
        "offload",
        "--moe-cache-auto",
        "--max-prefill-length",
        "2048",
        "--port",
        str(PORT),
        "--cuda-graph-max-bs",
        "1",
        "--served-model-name",
        "Ornith-1.5-35B-heretic-tq4",
    ]
    print("launch", " ".join(cmd), flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logf = open(LOG, "w", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
    )
    print("log", LOG)


if __name__ == "__main__":
    main()
