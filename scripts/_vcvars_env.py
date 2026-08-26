"""Load VS18 vcvars64 into os.environ (and prepend UCRT) then exec a command."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

VCVARS = Path(r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat")
UCRT_BIN = Path(r"S:\WADK102\bin\10.0.26100.0\x64")
UCRT_LIB = Path(r"S:\WADK102\Lib\10.0.26100.0\ucrt\x64")
UCRT_INC = Path(r"S:\WADK102\Include\10.0.26100.0\ucrt")


def apply_vcvars() -> None:
    bat = Path(r"J:\LLM\workspaces\FreeToken-vcruz\scripts\_vcvars_dump.cmd")
    bat.write_text(
        "\r\n".join(
            [
                "@echo off",
                f'call "{VCVARS}" >nul',
                "if errorlevel 1 exit /b 1",
                "set",
                "",
            ]
        ),
        encoding="ascii",
    )
    r = subprocess.run(
        ["cmd.exe", "/c", str(bat)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
    )
    if r.returncode != 0:
        raise SystemExit(f"vcvars failed rc={r.returncode} err={r.stderr[-800:]} out={r.stdout[-400:]}")
    for line in r.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k and k not in ("PROMPT",):
            os.environ[k] = v
    # UCRT prepend (lab pin)
    path = os.environ.get("PATH", "")
    lib = os.environ.get("LIB", "")
    inc = os.environ.get("INCLUDE", "")
    if UCRT_BIN.exists():
        os.environ["PATH"] = str(UCRT_BIN) + ";" + path
    if UCRT_LIB.exists():
        os.environ["LIB"] = str(UCRT_LIB) + ";" + lib
    if UCRT_INC.exists():
        os.environ["INCLUDE"] = str(UCRT_INC) + ";" + inc
    os.environ.setdefault("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2")


def main() -> None:
    apply_vcvars()
    cl = subprocess.run(["where", "cl"], capture_output=True, text=True, errors="replace")
    print("cl:", cl.stdout.strip() or cl.stderr.strip(), flush=True)
    if len(sys.argv) < 2:
        return
    raise SystemExit(subprocess.call(sys.argv[1:]))


if __name__ == "__main__":
    main()
