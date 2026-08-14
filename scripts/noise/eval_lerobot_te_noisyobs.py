import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import argparse
import torch
import numpy as np
from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task


BASELINE_DIR = Path(
    "/home1/eunji24/lerobot_project/outputs/train/"
    "act_network_baseline_100k/checkpoints/last/pretrained_model"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_path", type=str, default=str(BASELINE_DIR))
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=50)

    # LeRobot internal TE 설정
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=-0.01)
    parser.add_argument("--n_action_steps", type=int, default=1)

    # observation noise
    parser.add_argument("--state_noise_std", type=float, default=0.0)
    parser.add_argument("--state_noise_ratio", type=float, default=0.0)
    parser.add_argument("--image_noise_std", type=float, default=0.0)
    parser.add_argument("--image_noise_prob", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=0)

    return parser.parse_args()


def set_policy_te_config(policy, args):
    """
    LeRobot ACTPolicy 내부 temporal ensemble 설정을 강제로 맞춘다.
    버전에 따라 config field 이름이 다를 수 있어서 가능한 이름들을 같이 처리.
    """

    # 핵심: ACT internal TE는 n_action_steps=1일 때 써야 함
    if hasattr(policy.config, "n_action_steps"):
        policy.config.n_action_steps = args.n_action_steps

    if hasattr(policy, "config"):
        # 흔한 이름
        if hasattr(policy.config, "temporal_ensemble_coeff"):
            policy.config.temporal_ensemble_coeff = args.temporal_ensemble_coeff

        # 혹시 다른 이름으로 있을 경우 대비
        if hasattr(policy.config, "temporal_ensemble_coef"):
            policy.config.temporal_ensemble_coef = args.temporal_ensemble_coeff

    # config를 로드 후 바꾸면 temporal_ensembler 객체가 자동 생성되지 않음.
    # 따라서 여기서 직접 생성해야 LeRobot internal TE가 실제로 켜진다.
    if args.temporal_ensemble_coeff is not None:
        if hasattr(policy.config, "chunk_size"):
            chunk_size = policy.config.chunk_size
        elif hasattr(policy.config, "chunk_size"):
            chunk_size = policy.config.chunk_size
        else:
            chunk_size = 100

        policy.temporal_ensembler = ACTTemporalEnsembler(
            args.temporal_ensemble_coeff,
            chunk_size,
        )

    print("[LEROBOT TE CONFIG]")
    print("n_action_steps =", getattr(policy.config, "n_action_steps", "NA"))
    print("temporal_ensemble_coeff =", getattr(policy.config, "temporal_ensemble_coeff", "NA"))
    print("temporal_ensemble_coef =", getattr(policy.config, "temporal_ensemble_coef", "NA"))
    print("has temporal_ensembler =", hasattr(policy, "temporal_ensembler"))


def reset_policy_buffers(policy):
    """
    episode 시작 시 LeRobot policy 내부 action queue / temporal ensemble buffer 초기화.
    단, 이 ACTPolicy 버전은 temporal_ensembler가 없는 상태에서 policy.reset()을 부르면 터질 수 있음.
    그래서 temporal_ensembler가 있을 때만 reset 호출.
    """
    if hasattr(policy, "temporal_ensembler"):
        try:
            policy.reset()
        except Exception as e:
            print("[WARN] policy.reset() failed:", repr(e))

    if hasattr(policy, "reset_action_queue"):
        try:
            policy.reset_action_queue()
        except Exception as e:
            print("[WARN] reset_action_queue() failed:", repr(e))

    if hasattr(policy, "_action_queue"):
        try:
            policy._action_queue.clear()
        except Exception:
            pass

    if hasattr(policy, "_actions"):
        try:
            policy._actions.clear()
        except Exception:
            pass

    if hasattr(policy, "temporal_ensembler"):
        try:
            policy.temporal_ensembler.reset()
        except Exception as e:
            print("[WARN] temporal_ensembler.reset() failed:", repr(e))


def add_observation_noise(obs, args, step=None):
    # ========================================================
    # state noise
    # ========================================================
    state_key = "observation.state"
    if state_key in obs and torch.is_tensor(obs[state_key]):
        if args.state_noise_std > 0.0 or args.state_noise_ratio > 0.0:
            state = obs[state_key]
            noise = torch.zeros_like(state)

            if args.state_noise_std > 0.0:
                noise = noise + args.state_noise_std * torch.randn_like(state)

            # batch std 기반이라 batch가 동일 state면 거의 0일 수 있음
            if args.state_noise_ratio > 0.0:
                scale = state.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
                noise = noise + args.state_noise_ratio * scale * torch.randn_like(state)

            obs[state_key] = state + noise

            if step == 0:
                print("[STATE NOISE DEBUG]")
                print("state mean abs:", state.abs().mean().item())
                print("state std mean:", state.std(dim=0).mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print(
                    "noise/state abs ratio:",
                    (noise.abs().mean() / state.abs().mean().clamp_min(1e-6)).item(),
                )

    # ========================================================
    # image noise
    # ========================================================
    if args.image_noise_std > 0.0:
        for key in list(obs.keys()):
            if not key.startswith("observation.images"):
                continue
            if not torch.is_tensor(obs[key]):
                continue

            img = obs[key]

            if args.image_noise_prob < 1.0:
                B = img.shape[0]
                apply_mask = torch.rand(B, device=img.device) < args.image_noise_prob
                view_shape = [B] + [1] * (img.ndim - 1)
                apply_mask = apply_mask.view(*view_shape).float()
                noise = args.image_noise_std * torch.randn_like(img) * apply_mask
            else:
                noise = args.image_noise_std * torch.randn_like(img)

            noisy_img = img + noise

            # 0~1 이미지면 clamp, normalize image면 clamp 안 함
            img_min = img.min().item()
            img_max = img.max().item()
            if img_min >= -0.1 and img_max <= 1.1:
                noisy_img = noisy_img.clamp(0.0, 1.0)

            obs[key] = noisy_img

            if step == 0:
                denom = img.abs().mean().clamp_min(1e-6)
                print(f"[IMAGE NOISE DEBUG] {key}")
                print("image min/max/mean:", img.min().item(), img.max().item(), img.mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print("noise/image abs ratio:", (noise.abs().mean() / denom).item())

    return obs


def main():
    args = parse_args()

    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)

    print(f"Loading LeRobot ACT policy from {args.policy_path}")
    policy = ACTPolicy.from_pretrained(args.policy_path).to(DEVICE)
    policy.eval()
    set_policy_te_config(policy, args)

    config = policy.config

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=args.policy_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
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

    print(
        f"\nRunning LeRobot TE rollout: "
        f"{args.eval_episodes} episodes, batch={args.eval_batch_size}"
    )

    seeds = list(range(args.eval_episodes))[:args.eval_batch_size]
    obs_dict, _ = env.reset(seed=seeds)
    reset_policy_buffers(policy)

    done = np.zeros(args.eval_batch_size, dtype=bool)
    successes = np.zeros(args.eval_batch_size, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    step = 0
    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(obs_dict)
        obs = add_envs_task(env, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        # 핵심: LeRobot ACTPolicy.select_action 전에 observation noise
        obs = add_observation_noise(obs, args, step=step)

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        action_np = action.to("cpu").numpy()

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
            print(f"step {step}/{max_steps}, done {done.sum()}/{args.eval_batch_size}")

    pc_success = successes.mean()

    print("\nResults LeRobot internal TE noisy-observation eval")
    print(f"pc_success = {pc_success:.3f} ({successes.sum()}/{args.eval_batch_size})")
    print(f"temporal_ensemble_coeff = {args.temporal_ensemble_coeff}")
    print(f"n_action_steps = {args.n_action_steps}")
    print(f"state_noise_std = {args.state_noise_std}")
    print(f"state_noise_ratio = {args.state_noise_ratio}")
    print(f"image_noise_std = {args.image_noise_std}")
    print(f"image_noise_prob = {args.image_noise_prob}")

    print("\nReference:")
    print("  ACT baseline commit n=50:       62%")
    print("  LeRobot TE fresh-favored old:   54%")
    print("  wrapper pure TE clean old:      58%")

    env.close()


if __name__ == "__main__":
    main()
