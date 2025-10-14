import copy
import gc
import numpy as np
import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
import time
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.lifelong.utils import *

# import math

# def _quat_wxyz_to_R(q):
#     """Quaternion (w,x,y,z) -> 3x3 rotation matrix."""
#     w, x, y, z = [float(v) for v in q]
#     # normalize
#     n = (w*w + x*x + y*y + z*z) ** 0.5 + 1e-12
#     w, x, y, z = w/n, x/n, y/n, z/n
#     return torch.tensor([
#         [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
#         [  2*(x*y + z*w),   1 - 2*(x*x + z*z),   2*(y*z - x*w)],
#         [  2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x + y*y)]
#     ], dtype=torch.float64)

# def _R_to_euler_zyx(R):
#     """Rotation matrix -> (yaw, pitch, roll) with ZYX convention, radians."""
#     r00,r01,r02 = R[0]; r10,r11,r12 = R[1]; r20,r21,r22 = R[2]
#     yaw   = math.atan2(r10, r00)
#     pitch = math.asin(float(-r20))
#     roll  = math.atan2(r21, r22)
#     return torch.tensor([yaw, pitch, roll], dtype=torch.float32)

# def _wrap_pi(a):
#     return ((a + math.pi) % (2*math.pi)) - math.pi

# def _eef8_from(o, tool_frame="identity"):
#     """
#     Build [x,y,z,yaw,pitch,roll,gripL,gripR] from one obs dict.
#     Expects:
#       - 'robot0_eef_pos'        (3,)
#       - 'robot0_eef_quat'       (4,) in wxyz
#       - 'robot0_gripper_qpos'   (>=2,)
#     """
#     pos  = torch.as_tensor(o["robot0_eef_pos"], dtype=torch.float32)
#     quat = torch.as_tensor(o["robot0_eef_quat"], dtype=torch.float64)  # wxyz
#     R = _quat_wxyz_to_R(quat)

#     # Optional fixed tool-frame adjustment (common 180° flips)
#     if tool_frame == "flip_z":
#         R_off = torch.tensor([[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]], dtype=torch.float64)
#         R = R_off @ R
#     elif tool_frame == "flip_x":
#         R_off = torch.tensor([[1.,0.,0.],[0.,-1.,0.],[0.,0.,-1.]], dtype=torch.float64)
#         R = R_off @ R
#     elif tool_frame != "identity":
#         raise ValueError("tool_frame must be 'identity', 'flip_z', or 'flip_x'")

#     ypr = _R_to_euler_zyx(R)
#     ypr = torch.tensor([_wrap_pi(v.item()) for v in ypr], dtype=torch.float32)

#     grip = torch.as_tensor(o["robot0_gripper_qpos"], dtype=torch.float32)[:2]
#     return torch.cat([pos, ypr, grip], dim=0)  # [8]

import robosuite.utils.transform_utils as T  # robosuite's canonical conversions

def obs_batch_to_libero_robot_state(obs):
    """
    Convert a batch of robosuite-based LIBERO observations into LIBERO dataset 'state' format:
        [ x, y, z, rx, ry, rz, g1, g2 ]
    where (rx, ry, rz) is axis–angle (rotation vector) in radians.

    Args:
        obs_batch: sequence (list/tuple/np.ndarray) of observation dicts, each containing:
            - 'robot0_eef_pos'   : (3,) float  (meters, world/base frame)
            - 'robot0_eef_quat'  : (4,) float  (quaternion, **xyzw** in robosuite obs)
            - 'robot0_gripper_qpos': (2,) float (left/right finger joint positions)
    Returns:
        torch.Tensor with shape (8)
    """


    # 1) Position (3,)
    pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)

    # 2) Orientation: quaternion (robosuite uses xyzw in observations)
    quat_xyzw = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
    # Convert to axis–angle (rotation vector, 3,)
    rotvec = T.quat2axisangle(quat_xyzw).astype(np.float64)

    # 3) Gripper qpos (2,)
    grip = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64).reshape(2)

    out = np.concatenate([pos, rotvec, grip], axis=0).astype(np.float32)

    return torch.from_numpy(out)

def _rgb_to_chw01(img_np):
    t = torch.from_numpy(img_np).permute(2,0,1).contiguous()
    if t.dtype != torch.float32:
        t = t.float()
    if t.max() > 1.0:
        t = t / 255.0
    return t  # [C,H,W], float32 in [0,1]

def _depth_to_1hw(depth_np):
    t = torch.from_numpy(depth_np)
    if t.ndim == 2:               # HxW
        t = t.unsqueeze(0)
    elif t.ndim == 3 and t.shape[-1] == 1:  # HxWx1
        t = t.permute(2,0,1)
    elif t.ndim == 3 and t.shape[0] == 1:   # 1xHxW
        pass
    else:
        raise ValueError(f"Unexpected depth shape: {tuple(t.shape)}")
    return t.float()  # [1,H,W]

def raw_obs_to_tensor_lerobot_obs(obs, prev_obs, task_lang):
    """
    Multi-env converter to match HF/LeRobot LIBERO:
      - observation.state: [env, 2, 8] = [t, t_prev] of [x,y,z,yaw,pitch,roll,gripL,gripR]
      - observation.images.wrist1: [env, C, H, W] from robot0_eye_in_hand_image (RGB in [0,1])
      - observation.images.static1: [env, C, H, W] from agentview_image (RGB in [0,1])
      - observation.depths.static1: [env, 1, H, W] from agentview_depth
      - observation.intrinsics.static1: [env, 3, 3] from camera_intrinsics['agentview']['intrinsic_matrix_K']
      - observation.task_instr: [env] list[str]
    """
    
    env_num = len(obs)
    assert len(prev_obs) == env_num, "prev_obs must have the same length as obs"
    
    states      = [torch.stack([obs_batch_to_libero_robot_state(obs[k]),
                                obs_batch_to_libero_robot_state(prev_obs[k])], dim=0) for k in range(env_num)]
    wrist_imgs  = [_rgb_to_chw01(obs[k]["robot0_eye_in_hand_image"][::-1, ::-1].copy()) for k in range(env_num)]
    static_imgs = [_rgb_to_chw01(obs[k]["agentview_image"][::-1, ::-1].copy()) for k in range(env_num)]
    depths      = [_depth_to_1hw(obs[k]["agentview_depth"])         for k in range(env_num)]
    Ks          = [torch.as_tensor(obs[k]["camera_intrinsics"]["agentview"]["intrinsic_matrix_K"],
                                   dtype=torch.float32)
                   for k in range(env_num)]
    
    data = {
        "observation.state":             torch.stack(states, dim=0),       # [env, 2, 8]
        "observation.images.wrist1":     torch.stack(wrist_imgs, dim=0),   # [env, C, H, W]
        "observation.images.static1":    torch.stack(static_imgs, dim=0),  # [env, C, H, W]
        # "observation.depths.static1":    torch.stack(depths, dim=0),       # [env, 1, H, W]
        # "observation.intrinsics.static1":torch.stack(Ks, dim=0),           # [env, 3, 3]
        "observation.task_instr":        [task_lang] * env_num,
        "dataset_info": {
            "action_type": "eef",
            "robot_embodiment": "single_arm",
            "robot_type": "franka",
            "stereo_replace_depth": False,
            "handheld": False,
            "no_state": False,
            "obs_dof": 8,
            "action_dof": 7,
        },
        "inference_config": {
            "n_actions": 4,
            "n_inference_steps": 10,
        },
    }
    return data

    # New per-env list-of-dicts implementation
    env_num = len(obs)
    assert len(prev_obs) == env_num, "prev_obs must have the same length as obs"

    items = []
    for k in range(env_num):
        state_cur  = obs_batch_to_libero_robot_state(obs[k])
        state_prev = obs_batch_to_libero_robot_state(prev_obs[k])
        state = torch.stack([state_cur, state_prev], dim=0)  # [2, 8]

        wrist_img  = _rgb_to_chw01(obs[k]["robot0_eye_in_hand_image"][::-1, ::-1].copy())
        static_img = _rgb_to_chw01(obs[k]["agentview_image"][::-1, ::-1].copy())

        # Optional extras (kept similar to old version but not included by default)
        depth = _depth_to_1hw(obs[k]["agentview_depth"])
        K = torch.as_tensor(
            obs[k]["camera_intrinsics"]["agentview"]["intrinsic_matrix_K"],
            dtype=torch.float32,
        )

        item = {
            "observation.state":          state,       # [2, 8]
            "observation.images.wrist1":  wrist_img,   # [C, H, W]
            "observation.images.static1": static_img,  # [C, H, W]
            # "observation.depths.static1":    depth,   # [1, H, W]
            # "observation.intrinsics.static1": K,      # [3, 3]
            "observation.task_instr":     task_lang,
            "dataset_info": {
                "action_type": "eef",
                "robot_embodiment": "single_arm",
                "robot_type": "franka",
                "stereo_replace_depth": False,
                "handheld": False,
                "no_state": False,
                "obs_dof": 8,
                "action_dof": 7,
            },
            "inference_config": {
                "n_actions": 4,
                "n_inference_steps": 10,
            },
        }
        items.append(item)

    return items


def raw_obs_to_tensor_obs(obs, task_emb, cfg):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
        "obs": {},
        "task_emb": task_emb.repeat(env_num, 1),
    }

    all_obs_keys = []
    for modality_name, modality_list in cfg.data.obs.modality.items():
        for obs_name in modality_list:
            data["obs"][obs_name] = []
        all_obs_keys += modality_list

    for k in range(env_num):
        for obs_name in all_obs_keys:
            data["obs"][obs_name].append(
                ObsUtils.process_obs(
                    torch.from_numpy(obs[k][cfg.data.obs_key_mapping[obs_name]]),
                    obs_key=obs_name,
                ).float()
            )

    for key in data["obs"]:
        data["obs"][key] = torch.stack(data["obs"][key])

    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    return data


def evaluate_one_task_success(
    cfg, algo, task, task_emb, task_id, sim_states=None, task_str=""
):
    """
    Evaluate a single task's success rate
    sim_states: if not None, will keep track of all simulated states during
                evaluation, mainly for visualization and debugging purpose
    task_str:   the key to access sim_states dictionary
    """
    with Timer() as t:
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id)

        algo.eval()
        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # initiate evaluation envs
        env_args = {
            "bddl_file_name": os.path.join(
                cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": cfg.data.img_h,
            "camera_widths": cfg.data.img_w,
        }

        env_num = min(cfg.eval.num_procs, cfg.eval.n_eval) if cfg.eval.use_mp else 1
        eval_loop_num = (cfg.eval.n_eval + env_num - 1) // env_num

        # Try to handle the frame buffer issue
        env_creation = False

        count = 0
        while not env_creation and count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                else:
                    env = SubprocVectorEnv(
                        [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
                    )
                env_creation = True
            except:
                time.sleep(5)
                count += 1
        if count >= 5:
            raise Exception("Failed to create environment")

        ### Evaluation loop
        # get fixed init states to control the experiment randomness
        init_states_path = os.path.join(
            cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)
        num_success = 0
        for i in range(eval_loop_num):
            env.reset()
            indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
            init_states_ = init_states[indices]

            dones = [False] * env_num
            steps = 0
            algo.reset()
            obs = env.set_init_state(init_states_)

            # dummy actions [env_num, 7] all zeros for initial physics simulation
            dummy = np.zeros((env_num, 7))
            for _ in range(5):
                obs, _, _, _ = env.step(dummy)

            if task_str != "":
                sim_state = env.get_sim_state()
                for k in range(env_num):
                    if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                        sim_states[i * env_num + k].append(sim_state[k])

            while steps < cfg.eval.max_steps:
                steps += 1

                data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
                actions = algo.policy.get_action(data)

                obs, reward, done, info = env.step(actions)

                # record the sim states for replay purpose
                if task_str != "":
                    sim_state = env.get_sim_state()
                    for k in range(env_num):
                        if i * env_num + k < cfg.eval.n_eval and sim_states is not None:
                            sim_states[i * env_num + k].append(sim_state[k])

                # check whether succeed
                for k in range(env_num):
                    dones[k] = dones[k] or done[k]

                if all(dones):
                    break

            # a new form of success record
            for k in range(env_num):
                if i * env_num + k < cfg.eval.n_eval:
                    num_success += int(dones[k])

        success_rate = num_success / cfg.eval.n_eval
        env.close()
        gc.collect()
    print(f"[info] evaluate task {task_id} takes {t.get_elapsed_time():.1f} seconds")
    return success_rate


def evaluate_success(cfg, algo, benchmark, task_ids, result_summary=None):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        task_str = f"k{task_ids[-1]}_p{i}"
        curr_summary = result_summary[task_str] if result_summary is not None else None
        success_rate = evaluate_one_task_success(
            cfg, algo, task_i, task_emb, i, sim_states=curr_summary, task_str=task_str
        )
        successes.append(success_rate)
    return np.array(successes)


def evaluate_multitask_training_success(cfg, algo, benchmark, task_ids):
    """
    Evaluate the success rate for all task in task_ids.
    """
    algo.eval()
    successes = []
    for i in task_ids:
        task_i = benchmark.get_task(i)
        task_emb = benchmark.get_task_emb(i)
        success_rate = evaluate_one_task_success(cfg, algo, task_i, task_emb, i)
        successes.append(success_rate)
    return np.array(successes)


@torch.no_grad()
def evaluate_loss(cfg, algo, benchmark, datasets):
    """
    Evaluate the loss on all datasets.
    """
    algo.eval()
    losses = []
    for i, dataset in enumerate(datasets):
        if cfg.lifelong.algo == "PackNet":  # need preprocess weights for PackNet
            algo = algo.get_eval_algo(task_id=i)

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.eval.batch_size,
            num_workers=cfg.eval.num_workers,
            shuffle=False,
        )
        test_loss = 0
        for data in dataloader:
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=cfg.device)
            )
            loss = algo.policy.compute_loss(data)
            test_loss += loss.item()
        test_loss /= len(dataloader)
        losses.append(test_loss)
    return np.array(losses)
