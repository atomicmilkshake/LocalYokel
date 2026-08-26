"""HumanEval subset (10 tasks) against a live FreeToken OpenAI-compatible server.

Not a smoke test: generates completions, extracts the function, runs the official
check() harness, records pass/fail per task.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

TASKS = [
    "HumanEval/0",
    "HumanEval/1",
    "HumanEval/2",
    "HumanEval/3",
    "HumanEval/4",
    "HumanEval/5",
    "HumanEval/6",
    "HumanEval/8",
    "HumanEval/11",
    "HumanEval/21",
]


def load_problems() -> dict:
    try:
        from human_eval.data import read_problems

        return read_problems()
    except Exception:
        pass
    # fallback: local jsonl next to this script or HF-style cache
    for cand in (
        Path(__file__).with_name("HumanEval.jsonl"),
        Path(r"U:\LLM\models\HumanEval.jsonl"),
        Path(r"J:\LLM\workspaces\FreeToken-vcruz\scripts\HumanEval.jsonl"),
    ):
        if cand.exists():
            out = {}
            with cand.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    out[row["task_id"]] = row
            return out
    raise SystemExit("human_eval package or HumanEval.jsonl not found")


def chat_complete(base: str, model: str, prompt: str, max_tokens: int, temperature: float) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Python coding assistant. Complete the given function. "
                        "Output only Python code: the function definition and body. "
                        "No markdown fences, no explanation."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read().decode())
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    return (msg.get("content") or "") + (msg.get("reasoning_content") or "")


def extract_code(text: str, entry_point: str) -> str:
    t = text.replace("```python", "```").replace("```py", "```")
    if "```" in t:
        parts = t.split("```")
        # first fenced block
        if len(parts) >= 2:
            t = parts[1]
    # keep from first def matching entry_point if present
    marker = f"def {entry_point}"
    idx = t.find(marker)
    if idx >= 0:
        t = t[idx:]
    # drop trailing chatter after a likely end
    lines = []
    started = False
    for line in t.splitlines():
        if line.startswith("def ") or line.startswith("from ") or line.startswith("import ") or started:
            started = True
            if started and line.startswith("# ") and "example" in line.lower():
                break
            if started and line.startswith("print("):
                break
            lines.append(line)
    return "\n".join(lines).strip() + "\n"


def wait_ready(base: str, timeout_s: float = 1800) -> None:
    deadline = time.time() + timeout_s
    health = base.rstrip("/") + "/v1/models"
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as resp:
                if resp.status == 200:
                    print("server ready", resp.read()[:200], flush=True)
                    return
        except Exception as e:
            last = repr(e)
        time.sleep(3)
    raise SystemExit(f"server not ready after {timeout_s}s last={last}")


def exec_check(problem: dict, completion: str) -> tuple[bool, str]:
    # Official human-eval check: prompt + completion + test + check(entry_point)
    program = (
        problem["prompt"]
        + completion
        + "\n"
        + problem["test"]
        + "\n"
        + f"check({problem['entry_point']})\n"
    )
    ns: dict = {}
    try:
        exec(compile(program, "<humaneval>", "exec"), ns, ns)
        return True, "pass"
    except Exception:
        return False, traceback.format_exc()[-1500:]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:1963")
    p.add_argument("--model", default="local")
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default=str(Path(__file__).with_name("humaneval_subset_results.json")))
    args = p.parse_args()
    wait_ready(args.base)
    problems = load_problems()
    rows = []
    passed = 0
    for tid in TASKS:
        prob = problems[tid]
        print("=" * 60, flush=True)
        print("TASK", tid, prob["entry_point"], flush=True)
        t0 = time.time()
        try:
            raw = chat_complete(args.base, args.model, prob["prompt"], args.max_tokens, args.temperature)
        except Exception as e:
            raw = ""
            err = f"request failed: {e}"
            ok = False
            code = ""
        else:
            code = extract_code(raw, prob["entry_point"])
            ok, err = exec_check(prob, code)
        dt = time.time() - t0
        passed += int(ok)
        print("PASS" if ok else "FAIL", f"{dt:.1f}s", flush=True)
        if not ok:
            print(err[-800:], flush=True)
        rows.append(
            {
                "task_id": tid,
                "entry_point": prob["entry_point"],
                "ok": ok,
                "seconds": round(dt, 2),
                "completion": code,
                "raw": raw[:4000],
                "error": err if not ok else "",
            }
        )
    summary = {
        "n": len(TASKS),
        "passed": passed,
        "pass_at_1": passed / len(TASKS),
        "model": args.model,
        "base": args.base,
        "tasks": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "passed", "pass_at_1", "out") if k in summary} | {"out": args.out}, indent=2))
    print(f"HumanEval subset pass@1 = {passed}/{len(TASKS)} = {passed/len(TASKS):.0%}")


if __name__ == "__main__":
    main()
