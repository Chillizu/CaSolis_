#!/usr/bin/env python3
"""Run three independent CaSolis_ captures for comparison with live output."""
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
VENV_PYTHON = ROOT / ".venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200

results = []
for run_idx in range(1, 4):
    out_dir = ROOT / "run_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}")
    print(f"RUN {run_idx}/3 -> {out_dir}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [PYTHON, "-u", str(ROOT / "scripts" / "capture_full_run.py"), str(N_STEPS), str(out_dir)],
        cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            print(line, end="", flush=True)
    rc = proc.wait()
    elapsed = time.time() - t0
    results.append({
        "run": run_idx,
        "output_dir": str(out_dir),
        "elapsed": elapsed,
        "returncode": rc,
    })

print(f"\n{'='*60}")
print("ALL THREE CAPTURES COMPLETE")
print(f"{'='*60}")
for r in results:
    print(f"Run {r['run']}: {r['output_dir']} ({r['elapsed']:.1f}s, rc={r['returncode']})")

summary_path = ROOT / "run_outputs" / f"three_captures_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(summary_path, "w") as f:
    import json
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"Summary saved: {summary_path}")
