#!/usr/bin/env python3
# run_all_libero.py
import argparse, sys, subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

BENCHMARKS = {
    # "libero_object":  list(range(5)),
    # "libero_goal":    list(range(5)),
    "libero_spatial": list(range(10)),
    # "libero_10":      list(range(5)),
}

def run_one(task, args):
    bench, task_id, cmd, log_file = task
    log_file.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.now()
    start_str = start_dt.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[START {start_str}] Running: {' '.join(cmd)} -> {log_file}")

    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env.setdefault("MUJOCO_EGL_DEVICE_ID", str(args.device_id))
    env.setdefault("PYOPENGL_PLATFORM", "egl")

    with log_file.open("w") as f:
        f.write(f"[START {start_str}] CMD: {' '.join(cmd)}\n")
        f.write(f"[ENV] MUJOCO_GL={env['MUJOCO_GL']}\n")
        f.flush()
        try:
            rc = subprocess.run(
                cmd, stdout=f, stderr=subprocess.STDOUT, check=False,
                timeout=(args.timeout if args.timeout > 0 else None),
                env=env,
            ).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[TIMEOUT] Exceeded {args.timeout}s. Process was killed.\n")
            rc = -9
        finally:
            end_dt = datetime.now()
            end_str = end_dt.strftime("%Y-%m-%d %H:%M:%S")
            elapsed = (end_dt - start_dt).total_seconds()
            f.write(f"\n[END   {end_str}] Elapsed: {elapsed:.2f}s\n")

    return bench, task_id, rc, str(log_file), start_str, end_str, elapsed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="external_api_policy")
    p.add_argument("--algo", default="base")
    p.add_argument("--seed", default="0")
    p.add_argument("--load_task", type=int, default=0)
    p.add_argument("--device_id", type=int, default=0)
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--eval_script", default="libero/lifelong/evaluate.py")
    p.add_argument("--outdir", default="runs_libero_external_api")
    p.add_argument("--timeout", type=int, default=0, help="seconds; 0 = no timeout")
    p.add_argument("--max_tasks_parallel", "--jobs", dest="jobs", type=int, default=1)
    p.add_argument("--policy_config", type=str, default=None, help="Path to policy config JSON file")

    args = p.parse_args()

    seed = str(args.seed).strip()
    logs_dir = Path(args.outdir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build tasks
    tasks = []
    for bench, task_ids in BENCHMARKS.items():
        for task_id in task_ids:
            log_file = logs_dir / bench / f"task_{task_id}" / f"seed_{seed}.log"
            cmd = [
                args.python_bin, args.eval_script,
                "--benchmark", bench,
                "--task_id", str(task_id),
                "--algo", args.algo,
                "--policy", args.policy,
                "--seed", seed,
                "--load_task", str(args.load_task),
                "--device_id", str(args.device_id),
                "--envs", str(args.envs),
            ]
            if args.save_videos:
                cmd += ["--save-videos"]
            if args.policy_config:
                cmd += ["--policy_config", args.policy_config]

            tasks.append((bench, task_id, cmd, log_file))

    total = len(tasks)
    print(f"Submitting {total} jobs with up to {args.jobs} concurrent process(es).")
    print(f"Tail logs with: tail -f {logs_dir}/<suite>/task_<id>/seed_{seed}.log")

    # Run with capped concurrency; when one finishes, the executor schedules the next
    done_count = 0
    with ThreadPoolExecutor(max_workers=max(args.jobs, 1)) as ex:
        futures = [ex.submit(run_one, t, args) for t in tasks]
        for fut in as_completed(futures):
            bench, task_id, rc, logp, start_str, end_str, elapsed = fut.result()
            done_count += 1
            status = "OK" if rc == 0 else f"ERR rc={rc}"
            print(f"[{done_count}/{total}] {bench} task {task_id} -> {status} | {start_str} -> {end_str} ({elapsed:.2f}s) | log: {logp}")

    print("\nAll tasks finished. Logs:", logs_dir)

if __name__ == "__main__":
    main()