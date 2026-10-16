"""
Render ACM + Temporal Ensemble video.
ACM checkpoint를 로드해서 TE coeff=-0.01 적용한 영상 저장용.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import numpy as np
import imageio.v2 as imageio

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.envs.configs import AlohaEnv


POLICY_DIR = Path(
    "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_transfer_cube_test_100k/checkpoints/060000/pretrained_model"
)

OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/videos/acm_te_-001")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPISODES = 1
BATCH_SIZE = 1
VIDEO_FPS = 30

TE_COEFF = -0.01


def clean_frame(frame):
    if frame is None:
        return None

    if isinstance(frame, list):
        frame = frame[0]

    if isinstance(frame, tuple):
        frame = frame[0]

    if isinstance(frame, dict):
        frame = next(iter(frame.values()))

    if isinstance(frame, np.ndarray) and frame.ndim == 4:
        frame = frame[0]

    if not isinstance(frame, np.ndarray):
        return None

    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    return frame


def main():
    print("=" * 80)
    print("Render ACM + TE video")
    print("=" * 80)

    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")

    print(f"Loading policy config from {POLICY_DIR}")

    cli_overrides = [
        "chunk_size=100",
        "n_action_steps=1",
        f"temporal_ensemble_coeff={TE_COEFF}",
    ]

    policy_cfg = PreTrainedConfig.from_pretrained(
        str(POLICY_DIR),
        cli_overrides=cli_overrides,
    )

    print("Policy cfg loaded:")
    print(f"  type={policy_cfg.type}")
    print(f"  chunk_size={policy_cfg.chunk_size}")
    print(f"  n_action_steps={policy_cfg.n_action_steps}")
    print(f"  temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff}")

    policy = make_policy(
        cfg=policy_cfg,
        pretrained_path=str(POLICY_DIR),
    ).to(DEVICE)

    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(POLICY_DIR),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
    env_obj = make_env(env_cfg, n_envs=BATCH_SIZE, use_async_envs=False)

    while isinstance(env_obj, dict):
        print("make_env returned dict keys:", env_obj.keys())
        env_obj = list(env_obj.values())[0]

    env = env_obj
    print("final env type:", type(env))

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=policy_cfg,
    )

    try:
        max_steps = env.call("_max_episode_steps")[0]
    except Exception:
        max_steps = 400

    print(f"max_steps={max_steps}")

    for ep in range(N_EPISODES):
        print("\n" + "=" * 60)
        print(f"Episode {ep}")
        print("=" * 60)

        obs_dict, _ = env.reset(seed=[ep])

        if hasattr(policy, "reset"):
            policy.reset()

        done = np.zeros(BATCH_SIZE, dtype=bool)
        successes = np.zeros(BATCH_SIZE, dtype=bool)
        frames = []

        try:
            frame = clean_frame(env.render())
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"[VIDEO WARNING] initial render failed: {e}")

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
            action_np = action.to("cpu").numpy()

            obs_dict, reward, terminated, truncated, info = env.step(action_np)

            try:
                frame = clean_frame(env.render())
                if frame is not None:
                    frames.append(frame)
            except Exception as e:
                if step == 0:
                    print(f"[VIDEO WARNING] render failed: {e}")

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

            if step % 25 == 0:
                print(
                    f"step {step}/{max_steps}, "
                    f"done={done.sum()}/{BATCH_SIZE}, "
                    f"success={successes[0]}"
                )

        success = bool(successes[0])

        print("\nFinished episode.")
        print(f"  steps={step}")
        print(f"  success={success}")
        print(f"  frames={len(frames)}")

        if len(frames) == 0:
            print("[ERROR] No frames captured. env.render() did not return RGB frames.")
        else:
            video_path = OUT_DIR / f"acm_te_-001_seed{ep}_success{int(success)}.mp4"
            imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
            print(f"[VIDEO SAVED] {video_path}")

    env.close()


if __name__ == "__main__":
    main()