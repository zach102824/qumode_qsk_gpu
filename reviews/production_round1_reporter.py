#!/usr/bin/env python3
"""Read-only round-1 reporter. No CUDA, job dispatch, source edits, or retries.

Posts bounded, non-sensitive progress to a designated GitHub issue using gh CLI.
Keep raw scientific data and full logs local. The repository is public.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = "zach102824/qumode_qsk_gpu"
ISSUE = 3
LOG = ROOT / "logs/production_round1.log"
MANIFEST = ROOT / "runs/round_01_manifest.json"
SUMMARY = ROOT / "summaries/round_01.json"
RUNS = ROOT / "runs/main"
STATE = ROOT / ".local-gpu-runner/round1-reporter.json"
EXPECTED = {"entropy": 60, "spsa": 720, "bfgs_pack": 720, "bfgs_start": 11520}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def snapshot():
    counts = dict.fromkeys(EXPECTED, 0)
    failures = 0
    if RUNS.is_dir():
        for p in RUNS.iterdir():
            if not p.is_file() or p.suffix != ".json":
                continue
            name = p.name
            if "_FAIL.json" in name:
                failures += 1
            elif name.startswith("entropy_N"):
                counts["entropy"] += 1
            elif name.startswith("spsa_N"):
                counts["spsa"] += 1
            elif name.startswith("bfgs_N") and name.endswith("_pack.json"):
                counts["bfgs_pack"] += 1
            elif name.startswith("bfgs_N") and "_start" in name:
                counts["bfgs_start"] += 1
    manifest = read_json(MANIFEST)
    summary = read_json(SUMMARY)
    log_tail = ""
    try:
        with LOG.open("rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 40000))
            log_tail = f.read().decode("utf-8", errors="replace")
    except OSError:
        pass
    errors = []
    for line in log_tail.splitlines():
        if any(term in line.lower() for term in
               ("traceback", "pipeline aborted", "binding changed", "error:", "exception:", "fail:")):
            # Never copy arbitrary local paths or complete traceback text to a PUBLIC issue.
            safe = re.sub(r"/home/[^\s'\"),]+", "[local-path]", line.strip())
            safe = re.sub(r"(?i)(gh[opusr]_[a-z0-9_]+|github_pat_[a-z0-9_]+)", "[redacted]", safe)
            errors.append(safe[:180])
    return {
        "counts": counts, "failures": failures,
        "manifest_status": manifest.get("status", "unknown"),
        "manifest_ok": manifest.get("n_ok"),
        "manifest_fail": manifest.get("n_fail"),
        "summary_complete": summary.get("official_complete"),
        "recent_errors": errors[-4:],
        "log_present": LOG.exists(),
    }


def message(s):
    lines = [
        "### Local production round 1 status",
        f"Checked (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Files observed (not yet verified for official provenance):",
    ]
    lines += [f"- {k}: **{s['counts'][k]} / {EXPECTED[k]}**"
              for k in EXPECTED]
    lines += [
        f"- FAIL files: **{s['failures']}**",
        f"- Manifest status: **{s['manifest_status']}**",
        f"- Manifest success/failure: **{s['manifest_ok']} / {s['manifest_fail']}**",
        f"- Official completion flag: **{s['summary_complete']}**",
    ]
    if s["recent_errors"]:
        lines += ["", "Recent log diagnostics (sanitized):", "~~~text"]
        lines += s["recent_errors"]
        lines += ["~~~"]
    lines += ["", "No production control or code modification was performed.",
              "Raw data, full logs, and numerical diagnostics remain on the local host."]
    return "\n".join(lines)


def publish(body):
    # gh CLI uses the authenticated GitHub credential; never log a token.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
        f.write(body)
        tmp = f.name
    try:
        subprocess.run(["gh", "issue", "comment", str(ISSUE), "--repo", REPO,
                        "--body-file", tmp], check=True, timeout=45,
                       stdout=subprocess.DEVNULL)
    finally:
        os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--once", action="store_true", help="Publish one report then exit")
    ap.add_argument("--interval", type=int, default=600, help="Polling seconds (minimum 60)")
    args = ap.parse_args()
    if args.interval < 60:
        ap.error("interval must be >= 60")
    STATE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    while True:
        s = snapshot()
        last = read_json(STATE)
        now = time.time()
        changed = {k: v for k, v in s.items() if k != "recent_errors"} != {
            k: v for k, v in last.get("snapshot", {}).items() if k != "recent_errors"}
        due = now - float(last.get("published_at", 0)) >= 3600
        completed = s["manifest_status"] == "finished"
        if args.once or changed or due:
            try:
                publish(message(s))
                STATE.write_text(json.dumps({"snapshot": s, "published_at": now}, indent=2))
                print("Published round-1 status to GitHub issue #3", flush=True)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"Publication failed (will retry): {type(exc).__name__}: {exc}", flush=True)
        if args.once or completed:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
