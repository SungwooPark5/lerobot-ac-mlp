import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import json
import shutil
import tempfile
import argparse
from pathlib import Path

import torch
import numpy as np

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task


BASELINE_DIR = Path(
    "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/"
    "acm_transfer_cube_test_100k/checkpoints/060000/pretrained_model"
)


def make_temp_policy_dir(src_dir: Path, coeff: float, n_action_steps: int):
    tmp_root = Path(tempfile.mkdtemp(prefix="tmp_acm60k_lerobot_te_"))
    dst_dir = tmp_root / "pretrained_model"
    shutil.copytree(src_dir, dst_dir)

    cfg_path = dst_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())

    cfg["temporal_ensemble_coeff"] = coeff
    if "n_action_steps" in cfg:
        cfg["n_action_steps"] = n_action_steps

    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[TEMP POLICY DIR] {dst_dir}")
    print(f"[TEMP CONFIG] temporal_ensemble_coeff={cfg.get('temporal_ensemble_coeff')}")
    print(f"[TEMP CONFIG] n_action_steps={cfg.get('n_action_steps')}")
    return dst_dir


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coeff", type=float, default=-0.01)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=50)
    parser.add_argument("--seed_offset", type=int, default=0)
    parser.add_argument("--n_action_steps", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("LeRobot internal TE eval: ACM 60k")
    print(f"BASELINE_DIR     = {BASELINE_DIR}")
    print(f"coeff            = {args.coeff}")
    print(f"n_action_steps   = {args.n_action_steps}")
    print(f"eval_episodes    = {args.eval_episodes}")
    print(f"eval_batch_size  = {args.eval_batch_size}")
    print(f"seed_offset      = {args.seed_offset}")
    print(f"device           = {device}")
    print("=" * 100)

    policy_dir = make_temp_policy_dir(
        BASELINE_DIR,
        coeff=args.coeff,
        n_action_steps=args.n_action_steps,
    )

    print(f"Loading ACM policy from {policy_dir}")
    policy = ACMPolicy.from_pretrained(str(policy_dir)).to(device)
    policy.eval()
    config = policy.config

    if hasattr(policy, "reset"):
        policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(policy_dir),
        preprocessor_overrides={
            "device_processor": {"device": str(device)},
        },
    )

    from lerobot.envs.configs import AlohaEnv
    env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
    env_obj = make_env(env_cfg, n_envs=args.eval_batch_size, use_async_envs=False)

    while isinstance(env_obj, dict):
        print("make_env returned dict keys:", env_obj.keys())
        env_obj = list(env_obj.values())[0]

    env = env_obj
    print("final env type:", type(env))

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    seeds = list(range(args.seed_offset, args.seed_offset + args.eval_batch_size))
    print(f"seeds = {seeds[:5]} ... {seeds[-5:]}")

    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(args.eval_batch_size, dtype=bool)
    successes = np.zeros(args.eval_batch_size, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    step = 0
    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(obs_dict)
        obs = add_envs_task(env, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        action_np = action.detach().cpu().numpy()

        obs_dict, reward, terminated, truncated, info = env.step(action_np)
        done = done | terminated | truncated

        if "final_info" in info:
            final_info = info["final_info"]

            if isinstance(final_info, dict):
                if "is_success" in final_info:
                    successes |= np.array(final_info["is_success"], dtype=bool)
            else:
                for i, fi in enumerate(final_info):
                    if isinstance(fi, dict) and fi.get("is_success", False):
                        successes[i] = True

        step += 1
        if step % 50 == 0:
            print(f"  step {step}/{max_steps}, done {done.sum()}/{args.eval_batch_size}, success {successes.sum()}/{args.eval_batch_size}")

    pc_success = successes.mean()

    print("\nResults:")
    print(f"  coeff = {args.coeff}")
    print(f"  seed_offset = {args.seed_offset}")
    print(f"  pc_success = {pc_success:.3f} ({successes.sum()}/{args.eval_batch_size})")

    env.close()


if __name__ == "__main__":
    main()
