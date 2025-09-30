import argparse
import sys
import os

# ---- minimal fix: use spawn + force headless EGL before any GL imports ----
import multiprocessing as _mp
try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    # already set by another module / previous run
    pass

# ---------------------------------------------------------------------------


# TODO: find a better way for this?
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import hydra
import json
import numpy as np
import pprint
import time
import torch
import wandb
import yaml
from easydict import EasyDict
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModel, pipeline, AutoTokenizer, logging
from pathlib import Path

from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.algos import *
from libero.lifelong.datasets import get_dataset, SequenceVLDataset, GroupedTaskDataset
from libero.lifelong.metric import (
    evaluate_loss,
    evaluate_success,
    raw_obs_to_tensor_obs,
    raw_obs_to_tensor_lerobot_obs,
)
from libero.lifelong.utils import (
    control_seed,
    safe_device,
    torch_load_model,
    NpEncoder,
    compute_flops,
)

from libero.lifelong.main import get_task_embs

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils

import time
from attrdict import AttrDict

# Add this small helper near the top (after imports)
def _dict_to_obj(d):
    if isinstance(d, dict):
        return AttrDict(**{k: _dict_to_obj(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_obj(x) for x in d]
    else:
        return d

benchmark_map = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

algo_map = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

policy_map = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
    "external_api_policy": "ExternalAPIPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    # for which task suite
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task_id", type=int, required=True)
    # method detail
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy", "external_api_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)
    parser.add_argument("--device_id", type=int)
    parser.add_argument("--save-videos", action="store_true")
    # parser.add_argument('--save_dir',  type=str, required=True)
    args = parser.parse_args()
    args.device_id = "cuda:" + str(args.device_id)
    args.save_dir = f"{args.experiment_dir}_saved"

    if args.algo == "multitask":
        assert args.ep in list(
            range(0, 50, 5)
        ), "[error] ep should be in [0, 5, ..., 50]"
    else:
        assert args.load_task in list(
            range(10)
        ), "[error] load_task should be in [0, ..., 9]"
    return args


def main():
    # REMOVE LATER
    # ============
    expert_actions = torch.load("episode0_actions.pt")
    # ============

    args = parse_args()
    # e.g., experiments/LIBERO_SPATIAL/Multitask/BCRNNPolicy_seed100/

    experiment_dir = os.path.join(
        args.experiment_dir,
        f"{benchmark_map[args.benchmark]}/"
        + f"{algo_map[args.algo]}/"
        + f"{policy_map[args.policy]}_seed{args.seed}",
    )

    # find the checkpoint
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run_")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    
    if experiment_id == 0:
        print(f"[error] cannot find the checkpoint under {experiment_dir}")
        sys.exit(0)

    run_folder = os.path.join(experiment_dir, f"run_{experiment_id:03d}")
    try:
        if args.algo == "multitask":
            model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
        else:
            model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")
            sd, cfg, previous_mask = torch_load_model(
                model_path, map_location=args.device_id
            )
    except:
        if args.policy == "external_api_policy":
            cfg_json = os.path.join(run_folder, "config.json")
            with open(cfg_json, "r") as f:
                cfg = _dict_to_obj(json.load(f))
            
            sd = {}
            previous_mask = {}
            print(f"[info] no checkpoint found, proceeding with ExternalAPIPolicy using config.json from {cfg_json}")
        else:
            print(f"[error] cannot find the checkpoint at {str(model_path)}")
            sys.exit(0)
        
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")

    cfg.device = args.device_id
    algo = safe_device(eval(algo_map[args.algo])(10, cfg), cfg.device)
    algo.policy.previous_mask = previous_mask

    if cfg.lifelong.algo == "PackNet":
        algo.eval()
        for module_idx, module in enumerate(algo.policy.modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                mask = algo.previous_masks[module_idx].to(cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(args.task_id + 1)] = 0.0
                # we never train norm layers
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    algo.policy.load_state_dict(sd)

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    # get the benchmark the task belongs to
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    descriptions = [benchmark.get_task(i).language for i in range(10)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)

    ### ======================= start evaluation ============================

    # 1. evaluate dataset loss
    try:
        dataset, shape_meta = get_dataset(
            dataset_path=os.path.join(
                cfg.folder, benchmark.get_task_demonstration(args.task_id)
            ),
            obs_modality=cfg.data.obs.modality,
            initialize_obs_utils=True,
            seq_len=cfg.data.seq_len,
        )
        dataset = GroupedTaskDataset(
            [dataset], task_embs[args.task_id : args.task_id + 1]
        )
    except:
        print(
            f"[error] failed to load task {args.task_id} name {benchmark.get_task_names()[args.task_id]}"
        )
        sys.exit(0)

    algo.eval()

    test_loss = 0.0

    # 2. evaluate success rate
    if args.algo == "multitask":
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_ep{args.ep}_on{args.task_id}.stats",
        )
    else:
        save_folder = os.path.join(
            args.save_dir,
            f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}.stats",
        )

    video_folder = os.path.join(
        args.save_dir,
        f"{args.benchmark}_{args.algo}_{args.policy}_{args.seed}_load{args.load_task}_on{args.task_id}_videos",
    )

    print("Language Instruction: ", task.language)

    with Timer() as t, VideoWriter(video_folder, args.save_videos) as video_writer:
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
            "camera_depths": True,
            # "controller": "JOINT_VELOCITY" if cfg.policy == "external_api_policy" else "OSC_POSE",
        }

        # >>> added: pre-compute camera intrinsics using a temporary single env
        temp_env = OffScreenRenderEnv(**env_args)
        cam_intrinsics = {}
        # Some setups pass a single int instead of list; normalize to list
        if isinstance(temp_env.camera_names, str):
            camera_names_iter = [temp_env.camera_names]
        else:
            camera_names_iter = list(temp_env.camera_names)

        # Likewise normalize widths / heights to lists
        def to_list(x, n):
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x for _ in range(n)]

        widths = to_list(temp_env.camera_widths, len(camera_names_iter))
        heights = to_list(temp_env.camera_heights, len(camera_names_iter))

        for cam in camera_names_iter:
            cam_id = temp_env.sim.model.camera_name2id(cam)
            fovy_deg = float(temp_env.sim.model.cam_fovy[cam_id])  # vertical field of view in degrees
            idx = camera_names_iter.index(cam)
            width = int(widths[idx])
            height = int(heights[idx])
            fovy_rad = np.deg2rad(fovy_deg)
            # Using vertical fov: fy = (H/2) / tan(fovy/2); fx scaled by aspect ratio
            fy = 0.5 * height / np.tan(fovy_rad / 2.0)
            fx = fy * (width / height)
            cam_intrinsics[cam] = {
                "camera_name": cam,
                "image_width": width,
                "image_height": height,
                "fovy_deg": fovy_deg,
                "focal_length_px": {"fx": float(fx), "fy": float(fy)},
                "intrinsic_matrix_K": [
                    [float(fx), 0.0, width / 2.0],
                    [0.0, float(fy), height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        temp_env.close()
        # >>> end added block

        # env_num = 20
        env_num = 1

        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
        )
        env.reset()
        env.seed(cfg.seed)
        algo.reset()

        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        indices = np.arange(env_num) % init_states.shape[0]
        init_states_ = init_states[indices]

        dones = [False] * env_num
        steps = 0
        obs = env.set_init_state(init_states_)
        task_emb = benchmark.get_task_emb(args.task_id)
        task_lang = task.language

        # >>> added: attach intrinsics to initial obs
        for k in range(env_num):
            # Store a shared reference (avoid deep copy); change key name if you prefer
            obs[k]["camera_intrinsics"] = cam_intrinsics
        # >>> end added

        num_success = 0
        for _ in range(5):  # simulate the physics without any actions
            ac_dim = cfg.shape_meta.ac_dim
            prev_obs, _, _, _ = env.step(np.zeros((env_num, ac_dim)))

        with torch.no_grad():
            while steps < cfg.eval.max_steps:
                steps += 1

                # if args.policy == "external_api_policy":
                #     data = raw_obs_to_tensor_lerobot_obs(obs, prev_obs, task_lang)
                # else:
                #     data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                
                # actions = algo.policy.get_action(data)
                
                if steps == len(expert_actions):
                    break
                
                actions = expert_actions[steps][[0]].repeat(env_num, 1).cpu().numpy()

                prev_obs = obs
                obs, reward, done, info = env.step(actions)

                # >>> added: attach intrinsics to initial obs
                for k in range(env_num):
                    # Store a shared reference (avoid deep copy); change key name if you prefer
                    obs[k]["camera_intrinsics"] = cam_intrinsics
                # >>> end added

                video_writer.append_vector_obs(
                    obs, dones, camera_name="agentview_image"
                )

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]
                if all(dones):
                    break

            for k in range(env_num):
                num_success += int(dones[k])

        success_rate = num_success / env_num
        env.close()

        eval_stats = {
            "loss": test_loss,
            "success_rate": success_rate,
        }

        os.system(f"mkdir -p {args.save_dir}")
        torch.save(eval_stats, save_folder)
    print(
        f"[info] finish for ckpt at {run_folder} in {t.get_elapsed_time()} sec for rollouts"
    )
    print(f"Results are saved at {save_folder}")
    print("Loss: ", test_loss, "Success: ", success_rate)


if __name__ == "__main__":
    main()
                                                               