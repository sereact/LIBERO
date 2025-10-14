# run_all_libero_mgpu_simple.py
import argparse, os, re, subprocess, sys, threading, queue
from pathlib import Path
from statistics import mean
from tqdm import tqdm

# Edit suites/tasks here
BENCHMARKS = {
    # "libero_spatial": list(range(10)),
    # "libero_object":  list(range(10)),
    "libero_goal":    list(range(4)),
    # "libero_10":      list(range(10)),
}

SUCCESS_PATTERNS = [
    r"Success(?: Rate)?:\s*([0-9]*\.?[0-9]+)",
    r"success[_\s]rate\s*[:=]\s*([0-9]*\.?[0-9]+)",
]

def parse_success(text: str):
    vals = []
    for pat in SUCCESS_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            try:
                vals.append(float(m.group(1)))
            except Exception:
                pass
    return vals[-1] if vals else None

def run_one(cmd, logfile: Path, timeout_s: int, extra_env: dict):
    logfile.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(extra_env or {})
    with logfile.open("w") as f:
        try:
            res = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                check=False, timeout=timeout_s if timeout_s and timeout_s > 0 else None,
                env=env
            )
            return res.returncode, False

        except subprocess.TimeoutExpired:
            try: f.write(f"\n[TIMEOUT] Exceeded {timeout_s}s. Process was killed.\n")
            except Exception: pass
            return -9, True

def worker(gpu_id: int, task_q: "queue.Queue[tuple]", results: list, res_lock: threading.Lock,
           pbar: tqdm, args):
    while True:
        try:
            bench, task_id, seed, log_file = task_q.get_nowait()
        except queue.Empty:
            return

        # Mask this child to one GPU; use device_id=0 inside the masked view.
        env = {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "MUJOCO_GL": "egl",
            "PYOPENGL_PLATFORM": "egl",
            "MUJOCO_EGL_DEVICE_ID": str(gpu_id),
            "EGL_DEVICE_ID": str(gpu_id),
            "TOKENIZERS_PARALLELISM": "false",
        }

        cmd = [
            args.python_bin, args.eval_script,
            "--benchmark", bench,
            "--task_id", str(task_id),
            "--algo", args.algo,
            "--policy", args.policy,
            "--seed", str(seed),
            "--load_task", str(args.load_task),
            "--device_id", "0",             # 0 within the masked GPU
            "--envs", str(args.envs),
        ]
        if args.save_videos:
            cmd += ["--save-videos"]

        if args.verbose:
            tqdm.write(f"[GPU {gpu_id}] " + " ".join(cmd))

        rc, timed_out = run_one(cmd, log_file, args.timeout, env)

        with res_lock:
            results.append((bench, task_id, seed, rc, timed_out, gpu_id))
            if rc != 0:
                msg = "timeout" if timed_out else f"nonzero exit {rc}"
                tqdm.write(f"[warn][GPU {gpu_id}] {bench} task {task_id} {msg}")
            pbar.update(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="external_api_policy")
    ap.add_argument("--algo", default="base")
    ap.add_argument("--seed", default="100")
    ap.add_argument("--load_task", type=int, default=0)
    ap.add_argument("--save-videos", action="store_true")
    ap.add_argument("--python_bin", default=sys.executable)
    ap.add_argument("--eval_script", default="libero/lifelong/evaluate.py")
    ap.add_argument("--outdir", default="runs_libero_external_api")

    # Multi-GPU scheduling
    ap.add_argument("--gpus", type=str, required=True, help="e.g. 0,1,2,3")
    ap.add_argument("--jobs-per-gpu", type=int, default=1, help="slots per GPU (start with 1)")

    # Evaluate.py pass-through
    ap.add_argument("--envs", type=int, default=1, help="MuJoCo envs per eval (1 is safest)")

    # Reliability / UX
    ap.add_argument("--timeout", type=int, default=0, help="Per-task timeout (0 = no timeout)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    seed = int(str(args.seed).strip())
    outdir = Path(args.outdir)
    logs_dir = outdir / "logs"
    csv_path = outdir / "results.csv"
    summary_path = outdir / "summary_by_suite.csv"

    gpus = [int(x) for x in args.gpus.replace(" ", "").split(",") if x != ""]
    if not gpus:
        print("[error] No GPUs parsed from --gpus"); sys.exit(1)

    # Queue up all tasks
    task_q: "queue.Queue[tuple]" = queue.Queue()
    benches_items = list(BENCHMARKS.items())
    for bench, task_ids in benches_items:
        for task_id in task_ids:
            log_file = logs_dir / bench / f"task_{task_id}" / f"seed_{seed}.log"
            task_q.put((bench, task_id, seed, log_file))

    total = task_q.qsize()
    tqdm.write(f"Scheduling {total} jobs across GPUs {gpus} with {args.jobs_per_gpu} slot(s) / GPU.")
    tqdm.write(f"Tail logs with: tail -f {logs_dir}/<suite>/task_<id>/seed_{seed}.log")

    # Start workers: one thread per slot, each pinned to a specific GPU id
    results = []  # (bench, task_id, seed, returncode, timed_out, gpu_id)
    res_lock = threading.Lock()
    threads = []
    with tqdm(total=total, desc="Running jobs", unit="job") as pbar:
        for gpu in gpus:
            for _ in range(max(1, args.jobs_per_gpu)):
                t = threading.Thread(target=worker,
                                     args=(gpu, task_q, results, res_lock, pbar, args),
                                     daemon=True)
                t.start()
                threads.append(t)
        for t in threads:
            t.join()

    # Parse logs for success
    rows = []
    for bench, task_ids in benches_items:
        for task_id in task_ids:
            log_file = logs_dir / bench / f"task_{task_id}" / f"seed_{seed}.log"
            success = None
            if log_file.exists():
                text = log_file.read_text(errors="ignore")
                success = parse_success(text)
                if success is not None and success > 1.0:
                    success /= 100.0
            rows.append((bench, task_id, seed, success))

    # Write raw results CSV (+ returncode/timeout/gpu)
    outdir.mkdir(parents=True, exist_ok=True)
    rc_map = {(b, t): (rc, to, gpu) for (b, t, _, rc, to, gpu) in results}
    with (csv_path).open("w") as f:
        f.write("benchmark,task_id,seed,success,returncode,timed_out,gpu\n")
        for bench, task_id, seed, success in rows:
            rc, to, gpu = rc_map.get((bench, task_id), (None, False, None))
            f.write(f"{bench},{task_id},{seed},{'' if success is None else f'{success:.6f}'},"
                    f"{'' if rc is None else rc},{1 if to else 0},{'' if gpu is None else gpu}\n")

    # Per-suite means
    by_suite = {}
    for bench in BENCHMARKS:
        vals = [r[3] for r in rows if r[0] == bench and r[3] is not None]
        by_suite[bench] = (mean(vals) if vals else None)
    with (summary_path).open("w") as f:
        f.write("suite,mean_success\n")
        for bench in BENCHMARKS:
            ms = by_suite[bench]
            f.write(f"{bench},{'' if ms is None else f'{ms:.6f}'}\n")

    print(f"\nWrote raw results to: {csv_path}")
    print(f"Wrote per suite means to: {summary_path}")
    print("Tip: map suites → MolmoAct columns: Spatial=libero_spatial, Object=libero_object, Goal=libero_goal, Long=libero_long")

if __name__ == "__main__":
    main()