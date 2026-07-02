#!/usr/bin/env python3
"""
Capture a full CaSolis_ run and all artifacts.

- Runs OnlineAgent for N steps with full JSONL logging
- Exports all files created in the sandbox (/workspace, /tmp, /persistent)
- Copies persistent data and logs to a timestamped output directory
- Generates an inventory.json summary

Usage:
    python3 scripts/capture_full_run.py [steps] [output_dir]
"""
import sys
import os
import json
import shutil
import subprocess
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent.resolve()
N_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
OUTPUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "run_outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"


def run(cmd, **kw):
    print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False, **kw)


def reset_state():
    """Remove old sandbox and persistent data."""
    run(["docker", "rm", "-f", "casolis-sandbox"], capture_output=True, timeout=30)
    persistent = ROOT / "data" / "persistent"
    if persistent.exists():
        shutil.rmtree(persistent)
    persistent.mkdir(parents=True, exist_ok=True)
    (persistent / "tools").mkdir(exist_ok=True)
    (persistent / "scripts").mkdir(exist_ok=True)
    (persistent / "metadata").mkdir(exist_ok=True)
    # Also reset code_archive so each capture only contains current-run artifacts
    code_archive = ROOT / "data" / "code_archive"
    if code_archive.exists():
        shutil.rmtree(code_archive)
    code_archive.mkdir(parents=True, exist_ok=True)


def run_agent(n_steps: int):
    """Run the agent and return summary stats."""
    from agent.online_agent import OnlineAgent

    agent = OnlineAgent(conductor_gate=0.7, mode="auto")
    print(f"\n{'='*60}")
    print(f"Starting capture run: {n_steps} steps")
    print(f"{'='*60}\n")

    t0 = time.time()
    for i in range(n_steps):
        try:
            agent.step()
        except Exception as e:
            print(f"[Step {i+1}] error: {e}")
            break
    elapsed = time.time() - t0

    stats = {
        "steps": agent.step_count,
        "success_count": getattr(agent, "success_count", 0),
        "success_rate": getattr(agent, "success_count", 0) / max(agent.step_count, 1),
        "total_reward": getattr(agent, "total_reward", 0.0),
        "elapsed_seconds": elapsed,
    }
    if hasattr(agent, "workbench") and hasattr(agent.workbench, "graph"):
        gs = agent.workbench.graph.stats()
        stats["fact_graph_nodes"] = gs.get("n_nodes", 0)
        stats["fact_graph_edges"] = gs.get("n_edges", 0)
    if hasattr(agent, "pstore"):
        agent.pstore.save_all(agent, stats)
        agent.pstore.close()
    return stats


def export_sandbox_files(dest: Path):
    """Copy /workspace, /tmp, /persistent from the sandbox container.

    /tmp is tmpfs in the container, so docker cp silently fails for it.
    We use `docker exec tar` for /tmp and docker cp for the rest.
    """
    dest.mkdir(parents=True, exist_ok=True)

    # /tmp via tar stream to avoid tmpfs docker cp limitation
    tmp_dest = dest / "tmp"
    tmp_dest.mkdir(parents=True, exist_ok=True)
    print("[EXPORT] extracting /tmp via docker exec tar (tmpfs workaround)...")
    with open(tmp_dest / "tmp.tar.gz", "wb") as tar_out:
        res = subprocess.run(
            ["docker", "exec", "casolis-sandbox", "tar", "-C", "/tmp", "-czf", "-", "."],
            stdout=tar_out, stderr=subprocess.PIPE, timeout=60,
        )
    if res.returncode == 0 and (tmp_dest / "tmp.tar.gz").stat().st_size > 0:
        extract = subprocess.run(
            ["tar", "-xzf", str(tmp_dest / "tmp.tar.gz"), "-C", str(tmp_dest)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30,
        )
        if extract.returncode == 0:
            (tmp_dest / "tmp.tar.gz").unlink()
            print(f"[EXPORT] /tmp extracted: {len([p for p in tmp_dest.rglob('*') if p.is_file()])} files")
        else:
            print(f"[WARN] failed to extract /tmp tar: {extract.stderr.decode()}")
    else:
        print(f"[WARN] failed to extract /tmp via tar: {res.stderr.decode()}")

    # /workspace and /persistent via docker cp
    for src in ["/workspace", "/persistent"]:
        d = dest / src.lstrip("/")
        d.mkdir(parents=True, exist_ok=True)
        res = run(
            ["docker", "exec", "casolis-sandbox", "find", src, "-type", "f"],
            capture_output=True, text=True, timeout=30,
        )
        if res.returncode != 0:
            continue
        paths = [p for p in res.stdout.strip().split("\n") if p]
        for p in paths:
            rel = Path(p).relative_to(src)
            out = d / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            cp = run(
                ["docker", "cp", f"casolis-sandbox:{p}", str(out)],
                capture_output=True, timeout=30,
            )
            if cp.returncode != 0:
                print(f"  [WARN] failed to copy {p}: {cp.stderr.decode()}")
    return dest


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def collect_logs_and_persistent(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    copy_tree(ROOT / "run_logs", dest / "run_logs")
    copy_tree(ROOT / "data" / "persistent", dest / "persistent")
    copy_tree(ROOT / "data" / "code_archive", dest / "code_archive")


def build_inventory(output_dir: Path, stats: dict) -> dict:
    inventory = {
        "capture_time": datetime.now().isoformat(),
        "root": str(ROOT),
        "output_dir": str(output_dir),
        "run_stats": stats,
        "files": {},
    }

    def count_and_size(path: Path) -> dict:
        if not path.exists():
            return {"count": 0, "size_bytes": 0}
        files = [p for p in path.rglob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        return {"count": len(files), "size_bytes": total}

    for name in ["sandbox_workspace", "sandbox_tmp", "sandbox_persistent", "run_logs", "persistent", "code_archive"]:
        inventory["files"][name] = count_and_size(output_dir / name)

    # List notable generated files
    notable = []
    for base in ["sandbox_workspace", "sandbox_tmp", "sandbox_persistent"]:
        bp = output_dir / base
        if not bp.exists():
            continue
        for p in sorted(bp.rglob("*")):
            if p.is_file() and p.stat().st_size > 0:
                rel = f"{base}/{p.relative_to(bp)}"
                notable.append({
                    "path": rel,
                    "size_bytes": p.stat().st_size,
                })
    inventory["notable_files"] = sorted(notable, key=lambda x: x["path"])[:500]

    # Code archive files
    cap = output_dir / "code_archive"
    if cap.exists():
        inventory["code_archive_files"] = sorted(
            [{"name": p.name, "size_bytes": p.stat().st_size} for p in cap.iterdir() if p.is_file()],
            key=lambda x: x["name"],
        )[:500]

    # Log summary
    logs = sorted((output_dir / "run_logs").glob("run_*.jsonl")) if (output_dir / "run_logs").exists() else []
    if logs:
        log = logs[-1]
        step_count = 0
        plan_steps = 0
        with open(log) as f:
            for line in f:
                r = json.loads(line)
                if r.get("event") == "step":
                    step_count += 1
                    if r.get("plan_id"):
                        plan_steps += 1
        inventory["log_summary"] = {"log_file": log.name, "step_events": step_count, "plan_steps": plan_steps}

    inv_path = output_dir / "inventory.json"
    with open(inv_path, "w") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False, default=str)
    print(f"[INVENTORY] {inv_path}")
    return inventory


def main():
    print(f"[CAPTURE] output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[RESET] cleaning sandbox and persistent data...")
    reset_state()

    print("[RUN] starting agent...")
    stats = run_agent(N_STEPS)
    print("\n[RUN STATS]")
    print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))

    print("\n[EXPORT] copying sandbox files...")
    export_sandbox_files(OUTPUT_DIR / "sandbox")
    # Flatten sandbox dirs for convenience
    for src in ["workspace", "tmp", "persistent"]:
        s = OUTPUT_DIR / "sandbox" / src
        d = OUTPUT_DIR / f"sandbox_{src}"
        if s.exists():
            shutil.move(str(s), str(d))
    shutil.rmtree(OUTPUT_DIR / "sandbox", ignore_errors=True)

    print("[COPY] collecting logs and persistent data...")
    collect_logs_and_persistent(OUTPUT_DIR / "collected")
    # Flatten collected dirs for convenience
    for src in ["run_logs", "persistent", "code_archive"]:
        s = OUTPUT_DIR / "collected" / src
        d = OUTPUT_DIR / src
        if s.exists():
            shutil.move(str(s), str(d))
    shutil.rmtree(OUTPUT_DIR / "collected", ignore_errors=True)

    print("[INVENTORY] building summary...")
    inventory = build_inventory(OUTPUT_DIR, stats)

    print(f"\n{'='*60}")
    print(f"Capture complete: {OUTPUT_DIR}")
    print(f"Files: {inventory['files']}")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
