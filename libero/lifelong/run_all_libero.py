#!/usr/bin/env python3
# run_all_libero.py
import argparse
import sys
import subprocess
from pathlib import Path

# Edit suites/tasks here if needed
BENCHMARKS = {
    # "libero_object":  list(range(10)),
    # "libero_goal":    list(range(10)),
    "libero_spatial": list(range(10)),
    "libero_10":      list(range(10)),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="external_api_policy")
    p.add_argument("--algo", default="base")
    p.add_argument("--seed", default="100")
    p.add_argument("--load_task", type=int, default=0)
    p.add_argument("--device_id", type=int, default=0, help="index within CUDA_VISIBLE_DEVICES")
    p.add_argument("--envs", type=int, default=1, help="forwarded to evaluate.py")
    p.add_argument("--save-videos", action="store_true")
    p.add_argument("--python_bin", default=sys.executable)
    p.add_argument("--eval_script", default="libero/lifelong/evaluate.py")
    p.add_argument("--outdir", default="runs_libero_external_api")
    p.add_argument("--timeout", type=int, default=0, help="seconds; 0 = no timeout")
    args = p.parse_args()

    seed = str(args.seed).strip()
    outdir = Path(args.outdir)
    logs_dir = outdir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build task list (ordered)
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
            tasks.append((bench, task_id, cmd, log_file))

    print(f"Running {len(tasks)} job(s) sequentially.")
    print(f"Tail logs with: tail -f {logs_dir}/<suite>/task_<id>/seed_{seed}.log")

    # Run one-by-one
    for bench, task_id, cmd, log_file in tasks:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"[START] {bench} task {task_id}")
        print("Command:", " ".join(cmd))
        print(f"Log    : {log_file}")
        print("=" * 80)

        try:
            with log_file.open("w") as f:
                subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=(args.timeout if args.timeout > 0 else None),
                )
        except subprocess.TimeoutExpired:
            with log_file.open("a") as f:
                f.write(f"\n[TIMEOUT] Exceeded {args.timeout}s. Process was killed.\n")
            print(f"[TIMEOUT] {bench} task {task_id} exceeded {args.timeout}s")

        print(f"[DONE ] {bench} task {task_id}")

    print("\nAll tasks finished. Logs written under:", logs_dir)

if __name__ == "__main__":
    main()