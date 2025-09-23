# run_all_libero.py
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean
from tqdm import tqdm
from time import sleep  # added

BENCHMARKS = {
    # "libero_spatial": list(range(10)),
    # "libero_object":  list(range(10)),
    # "libero_goal":    list(range(10)),
    "libero_10":    list(range(10)),  # the "Long" column in MolmoAct
}

SUCCESS_PATTERNS = [
    r"Success(?: Rate)?:\s*([0-9]*\.?[0-9]+)",         # e.g., "Success: 0.72" or "Success Rate: 72.0"
    r"success[_\s]rate\s*[:=]\s*([0-9]*\.?[0-9]+)",    # common variation
]

def run_cmd(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        ret = proc.wait()
    return ret

# new: async launcher that keeps the logfile open while the process runs
def run_cmd_async(cmd, logfile):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    f = logfile.open("w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    return proc, f

def parse_success(log_text):
    # try to find the last mentioned success number in the log
    found_vals = []
    for pat in SUCCESS_PATTERNS:
        for m in re.finditer(pat, log_text, flags=re.IGNORECASE):
            try:
                found_vals.append(float(m.group(1)))
            except Exception:
                pass
    if not found_vals:
        return None
    return found_vals[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="external_api_policy")
    parser.add_argument("--algo", default="base")
    parser.add_argument("--seed", default="100")
    parser.add_argument("--load_task", type=int, default=0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--eval_script", default="libero/lifelong/evaluate.py")
    parser.add_argument("--outdir", default="runs_libero_external_api")
    # new: limit concurrency
    parser.add_argument("--jobs", type=int, default=1,
                        help="Max number of concurrent evaluation processes")
    args = parser.parse_args()

    seed = int(args.seed.strip())

    outdir = Path(args.outdir)
    logs_dir = outdir / "logs"
    csv_path = outdir / "results.csv"
    summary_path = outdir / "summary_by_suite.csv"

    rows = []
    # 1) run evaluations in parallel
    benches_items = list(BENCHMARKS.items())
    tasks = []
    for bench, task_ids in benches_items:
        for task_id in task_ids:
            log_file = logs_dir / bench / f"task_{task_id}" / f"seed_{seed}.log"
            cmd = [
                args.python_bin, args.eval_script,
                "--benchmark", bench,
                "--task_id", str(task_id),
                "--algo", args.algo,
                "--policy", args.policy,
                "--seed", str(seed),
                "--load_task", str(args.load_task),
                "--device_id", str(args.device_id),
            ]
            if args.save_videos:
                cmd += ["--save-videos"]
            tasks.append({
                "bench": bench,
                "task_id": task_id,
                "logfile": log_file,
                "cmd": cmd,
            })

    tqdm.write(f"Submitting {len(tasks)} jobs with up to {args.jobs} concurrent processes.")
    tqdm.write(f"Tail logs with: tail -f {logs_dir}/<suite>/task_<id>/seed_{seed}.log")

    total = len(tasks)
    active = {}  # proc -> (task, file_handle)
    started = 0
    pbar = tqdm(total=total, desc="Running jobs", unit="job")
    try:
        while started < total or active:
            # launch new ones up to limit
            while started < total and len(active) < args.jobs:
                t = tasks[started]
                started += 1
                tqdm.write("Starting: " + " ".join(t["cmd"]))
                proc, fh = run_cmd_async(t["cmd"], t["logfile"])
                active[proc] = (t, fh)

            # check for finished
            finished = [p for p in list(active.keys()) if p.poll() is not None]
            for p in finished:
                t, fh = active.pop(p)
                try:
                    fh.close()
                except Exception:
                    pass
                if p.returncode != 0:
                    tqdm.write(f"[warn] nonzero exit code {p.returncode} for {t['bench']} task {t['task_id']} seed {seed}")
                pbar.update(1)

            if active or started < total:
                sleep(0.2)
    except KeyboardInterrupt:
        tqdm.write("KeyboardInterrupt received, terminating running jobs...")
        for p in list(active.keys()):
            try:
                p.terminate()
            except Exception:
                pass
        for p in list(active.keys()):
            try:
                p.wait(timeout=5)
            except Exception:
                pass
        # ensure files are closed
        for p, (_, fh) in list(active.items()):
            try:
                fh.close()
            except Exception:
                pass
        raise
    finally:
        pbar.close()
        # best-effort to close any remaining file handles
        for _, (_, fh) in list(active.items()):
            try:
                fh.close()
            except Exception:
                pass

    # 2) parse logs (optional progress)
    for bench, task_ids in tqdm(benches_items, desc="Parsing benchmarks", unit="bench"):
        for task_id in tqdm(task_ids, desc=f"Parsing {bench} tasks", unit="task", leave=False):
            log_file = logs_dir / bench / f"task_{task_id}" / f"seed_{seed}.log"
            success = None
            if log_file.exists():
                text = log_file.read_text(errors="ignore")
                success = parse_success(text)
                # Some code prints percentages, some fractions; normalize if looks like percentage
                if success is not None and success > 1.0:
                    success = success / 100.0
            rows.append((bench, task_id, seed, success))

    # 3) write CSV of raw results
    outdir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w") as f:
        f.write("benchmark,task_id,seed,success\n")
        for bench, task_id, seed, success in rows:
            s = "" if success is None else f"{success:.6f}"
            f.write(f"{bench},{task_id},{seed},{s}\n")

    # 4) write per suite means (averaged over tasks and seeds)
    by_suite = {}
    for bench in BENCHMARKS:
        vals = [r[3] for r in rows if r[0] == bench and r[3] is not None]
        by_suite[bench] = (mean(vals) if vals else None)

    with summary_path.open("w") as f:
        f.write("suite,mean_success\n")
        for bench in BENCHMARKS:
            ms = by_suite[bench]
            s = "" if ms is None else f"{ms:.6f}"
            f.write(f"{bench},{s}\n")

    print(f"\nWrote raw results to: {csv_path}")
    print(f"Wrote per suite means to: {summary_path}")
    print("Tip: map suites to MolmoAct columns as Spatial=libero_spatial, Object=libero_object, Goal=libero_goal, Long=libero_long")

if __name__ == "__main__":
    main()
