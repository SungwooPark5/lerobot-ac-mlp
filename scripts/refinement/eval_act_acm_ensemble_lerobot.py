#!/usr/bin/env python

import argparse
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lerobot.envs.factory import make_env, make_env_config, make_env_pre_post_processors
from lerobot.envs.utils import (
    add_envs_task,
    close_envs,
    preprocess_observation,
)
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_IMAGES


class ACTACMFixedEnsemble:
    def __init__(
        self,
        act_path,
        acm_path,
        alpha,
        n_action_steps,
        dataset_repo_id,
        device,
    ):
        self.act_path = Path(act_path)
        self.acm_path = Path(acm_path)
        self.alpha = float(alpha)
        self.n_action_steps = int(n_action_steps)
        self.device = device

        print("[LOAD] ACT:", self.act_path)
        self.act = ACTPolicy.from_pretrained(self.act_path).to(device).eval()

        print("[LOAD] ACM:", self.acm_path)
        self.acm = ACMPolicy.from_pretrained(self.acm_path).to(device).eval()

        dataset = LeRobotDataset(dataset_repo_id)

        self.act_pre, _ = make_pre_post_processors(
            self.act.config,
            pretrained_path=self.act_path,
            dataset_stats=dataset.meta.stats,
        )
        self.acm_pre, self.acm_post = make_pre_post_processors(
            self.acm.config,
            pretrained_path=self.acm_path,
            dataset_stats=dataset.meta.stats,
        )

        self.queue = deque([], maxlen=self.n_action_steps)

    def reset(self):
        self.queue.clear()
        self.act.reset()
        self.acm.reset()

    @torch.no_grad()
    def predict_chunk(self, observation):
        """
        observation은 이미 LeRobot eval pipeline의
        preprocess_observation -> add_envs_task -> env_preprocessor
        까지 지난 dict임.
        """

        act_batch = self.act_pre(deepcopy(observation))
        acm_batch = self.acm_pre(deepcopy(observation))

        if self.act.config.image_features:
            act_batch = dict(act_batch)
            act_batch[OBS_IMAGES] = [act_batch[key] for key in self.act.config.image_features]

        if self.acm.config.image_features:
            acm_batch = dict(acm_batch)
            acm_batch[OBS_IMAGES] = [acm_batch[key] for key in self.acm.config.image_features]

        act_chunk = self.act.model(act_batch)[0]
        acm_chunk = self.acm.model(acm_batch)[0]

        mix_chunk = self.alpha * act_chunk + (1.0 - self.alpha) * acm_chunk
        return mix_chunk

    @torch.no_grad()
    def select_action(self, observation):
        """
        return: normalized policy action tensor, shape (B, action_dim)
        이후 main에서 acm_post + env_postprocessor를 적용함.
        """
        if len(self.queue) == 0:
            chunk = self.predict_chunk(observation)[:, : self.n_action_steps]
            self.queue.extend(chunk.transpose(0, 1))

        return self.queue.popleft()


def compute_action_smoothness(actions):
    """
    actions: np.ndarray, shape (T, action_dim)
    실제 env.step에 들어간 action sequence 기준 smoothness.
    """
    actions = np.asarray(actions, dtype=np.float32)

    if actions.shape[0] < 4:
        return {
            "exec_diff_l2": 0.0,
            "exec_diff_abs": 0.0,
            "exec_accel_l2": 0.0,
            "exec_accel_abs": 0.0,
            "exec_jerk_l2": 0.0,
            "exec_jerk_abs": 0.0,
            "exec_sign_flip": 0.0,
        }

    diff = actions[1:] - actions[:-1]
    accel = actions[2:] - 2.0 * actions[1:-1] + actions[:-2]
    jerk = actions[3:] - 3.0 * actions[2:-1] + 3.0 * actions[1:-2] - actions[:-3]

    d0 = diff[:-1]
    d1 = diff[1:]
    valid = (np.abs(d0) > 1e-6) & (np.abs(d1) > 1e-6)
    flips = (d0 * d1 < 0) & valid
    sign_flip = flips.sum() / max(valid.sum(), 1)

    return {
        "exec_diff_l2": float(np.linalg.norm(diff, axis=-1).mean()),
        "exec_diff_abs": float(np.abs(diff).mean()),
        "exec_accel_l2": float(np.linalg.norm(accel, axis=-1).mean()),
        "exec_accel_abs": float(np.abs(accel).mean()),
        "exec_jerk_l2": float(np.linalg.norm(jerk, axis=-1).mean()),
        "exec_jerk_abs": float(np.abs(jerk).mean()),
        "exec_sign_flip": float(sign_flip),
    }



def extract_success(info):
    if info is None:
        return False

    if isinstance(info, dict):
        if "is_success" in info:
            v = info["is_success"]
            if isinstance(v, np.ndarray):
                return bool(v.any())
            if isinstance(v, (list, tuple)):
                return any(bool(x) for x in v)
            return bool(v)

        if "final_info" in info:
            final_info = info["final_info"]

            if isinstance(final_info, dict):
                if "is_success" in final_info:
                    v = final_info["is_success"]
                    if isinstance(v, np.ndarray):
                        return bool(v.any())
                    if isinstance(v, (list, tuple)):
                        return any(bool(x) for x in v)
                    return bool(v)

                # Gymnasium vector env: final_info may be dict of arrays
                vals = []
                for v in final_info.values():
                    if isinstance(v, dict) and "is_success" in v:
                        vals.append(bool(v["is_success"]))
                if vals:
                    return any(vals)

            if isinstance(final_info, (list, tuple)):
                return any(extract_success(x) for x in final_info if x is not None)

    return False


def unwrap_make_env_output(envs):
    """
    Some LeRobot versions return nested dicts from make_env, e.g.
      {"aloha": {0: env}}
    instead of the env object directly.
    Unwrap until we get an object with reset/step.
    """
    depth = 0
    while isinstance(envs, dict):
        print(f"[DEBUG] make_env returned dict keys: {envs.keys()}")
        if len(envs) == 0:
            raise RuntimeError("make_env returned an empty dict")
        envs = next(iter(envs.values()))
        depth += 1
        if depth > 5:
            raise RuntimeError("Too many nested dict levels from make_env")
    return envs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--act_path", type=str, required=True)
    parser.add_argument("--acm_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--dataset_repo_id", type=str, default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--env_type", type=str, default="aloha")
    parser.add_argument("--task", type=str, default="AlohaTransferCube-v0")
    parser.add_argument("--n_action_steps", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("[ACT + ACM Fixed Weighted Ensemble Eval - LeRobot env]")
    print(f"ACT path        = {args.act_path}")
    print(f"ACM path        = {args.acm_path}")
    print(f"alpha           = {args.alpha}  # ACT weight")
    print(f"ACM weight      = {1.0 - args.alpha}")
    print(f"env_type        = {args.env_type}")
    print(f"task            = {args.task}")
    print(f"n_action_steps  = {args.n_action_steps}")
    print(f"eval_episodes   = {args.eval_episodes}")
    print(f"eval_batch_size = {args.eval_batch_size}")
    print(f"device          = {device}")
    print("=" * 100)

    policy = ACTACMFixedEnsemble(
        act_path=args.act_path,
        acm_path=args.acm_path,
        alpha=args.alpha,
        n_action_steps=args.n_action_steps,
        dataset_repo_id=args.dataset_repo_id,
        device=device,
    )

    env_cfg = make_env_config(args.env_type, task=args.task)
    envs = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)
    envs = unwrap_make_env_output(envs)

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy.acm.config,
    )

    all_success = []
    all_smoothness = []

    n_batches = int(np.ceil(args.eval_episodes / args.eval_batch_size))

    try:
        for batch_i in tqdm(range(n_batches), desc="Eval batches"):
            current_batch = min(
                args.eval_batch_size,
                args.eval_episodes - batch_i * args.eval_batch_size,
            )

            seeds = [args.seed + batch_i * args.eval_batch_size + i for i in range(args.eval_batch_size)]

            policy.reset()
            observation, info = envs.reset(seed=seeds)

            done = np.array([False] * args.eval_batch_size)
            success = np.array([False] * args.eval_batch_size)
            executed_actions = [[] for _ in range(args.eval_batch_size)]

            max_steps = envs.call("_max_episode_steps")[0]

            for step in range(max_steps):
                observation_proc = preprocess_observation(observation)
                observation_proc = add_envs_task(envs, observation_proc)
                observation_proc = env_preprocessor(observation_proc)

                with torch.inference_mode():
                    action = policy.select_action(observation_proc)

                # policy action unnormalize
                action = policy.acm_post(action)

                # env action postprocess
                action_transition = {"action": action}
                action_transition = env_postprocessor(action_transition)
                action = action_transition["action"]

                action_numpy = action.detach().cpu().numpy()

                # Save the exact action sent to env.step for rollout jitter analysis.
                for env_i in range(args.eval_batch_size):
                    if not done[env_i]:
                        executed_actions[env_i].append(action_numpy[env_i].copy())

                observation, reward, terminated, truncated, info = envs.step(action_numpy)

                if "final_info" in info:
                    final_info = info["final_info"]

                    if isinstance(final_info, dict):
                        if "is_success" in final_info:
                            success |= np.array(final_info["is_success"], dtype=bool)
                    elif isinstance(final_info, (list, tuple)):
                        for i, fi in enumerate(final_info):
                            if isinstance(fi, dict) and fi.get("is_success", False):
                                success[i] = True

                # fallback
                if extract_success(info):
                    success[:] = True

                done |= np.array(terminated, dtype=bool) | np.array(truncated, dtype=bool)

                if np.all(done):
                    break

            valid_success = success[:current_batch]
            all_success.extend(valid_success.tolist())

            batch_smooth = []
            for env_i in range(current_batch):
                if len(executed_actions[env_i]) > 0:
                    sm = compute_action_smoothness(np.stack(executed_actions[env_i], axis=0))
                    all_smoothness.append(sm)
                    batch_smooth.append(sm)

            if batch_smooth:
                batch_jerk = np.mean([m["exec_jerk_l2"] for m in batch_smooth])
                batch_accel = np.mean([m["exec_accel_l2"] for m in batch_smooth])
                batch_flip = np.mean([m["exec_sign_flip"] for m in batch_smooth])
            else:
                batch_jerk = batch_accel = batch_flip = float("nan")

            print(
                f"[Batch {batch_i+1}/{n_batches}] "
                f"success={valid_success.sum()}/{len(valid_success)} "
                f"running={np.mean(all_success):.4f} "
                f"exec_accel={batch_accel:.6f} "
                f"exec_jerk={batch_jerk:.6f} "
                f"exec_flip={batch_flip:.4f}"
            )

    finally:
        try:
            close_envs(envs)
        except NotImplementedError:
            envs.close()

    all_success = all_success[: args.eval_episodes]
    success_rate = float(np.mean(all_success))

    print("=" * 100)
    print("[RESULT]")
    print(f"alpha        = {args.alpha}")
    print(f"success_rate = {success_rate:.4f}")
    print(f"successes    = {sum(all_success)}/{len(all_success)}")

    if all_smoothness:
        keys = [
            "exec_diff_l2",
            "exec_diff_abs",
            "exec_accel_l2",
            "exec_accel_abs",
            "exec_jerk_l2",
            "exec_jerk_abs",
            "exec_sign_flip",
        ]
        print("")
        print("[EXECUTED ACTION SMOOTHNESS]")
        for k in keys:
            v = float(np.mean([m[k] for m in all_smoothness]))
            print(f"{k:20s} = {v:.8f}")

    print("=" * 100)


if __name__ == "__main__":
    main()