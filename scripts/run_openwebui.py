"""Launch local Open WebUI against the FreeToken serve on :1963."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"J:\LLM\workspaces\FreeToken-vcruz")
PY = Path(r"C:\Users\owenm\open-webui-venv\Scripts\python.exe")
BIN = Path(r"C:\Users\owenm\open-webui-venv\Scripts\open-webui.exe")
DATA = ROOT / "scripts" / "openwebui-data"
PORT = 8080
FT = "http://127.0.0.1:1963/v1"
MODEL = "Qwen3.6-35B-A3B-Uncensored-lyf-tq4"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "DATA_DIR": str(DATA),
            "WEBUI_AUTH": "False",
            "ENABLE_SIGNUP": "False",
            "ENABLE_PERSISTENT_CONFIG": "False",
            "OPENAI_API_BASE_URL": FT,
            "OPENAI_API_KEY": "sk-local",
            "DEFAULT_MODELS": MODEL,
            "WEBUI_NAME": "FreeToken Ornith",
            "PYTHONUNBUFFERED": "1",
        }
    )
    cmd = [str(BIN), "serve", "--host", "127.0.0.1", "--port", str(PORT)]
    print("open-webui", " ".join(cmd), flush=True)
    print("ui http://127.0.0.1:8080", "backend", FT, "model", MODEL, flush=True)
    raise SystemExit(subprocess.call(cmd, env=env, cwd=str(ROOT)))


if __name__ == "__main__":
    if not BIN.exists() and not PY.exists():
        raise SystemExit("open-webui venv missing at C:\\Users\\owenm\\open-webui-venv")
    main()
