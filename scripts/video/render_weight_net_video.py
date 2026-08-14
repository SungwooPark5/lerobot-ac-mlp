"""
Render video for Residual TE-prior learned ensemble.
기존 eval_with_weight_net.py 건드리지 않고, 영상 확인용으로만 사용.
"""

import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path
import imageio.v2 as imageio

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task


BASELINE_DIR = Path("/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model")
WEIGHT_NET_PATH = Path("/home1/eunji24/lerobot_project/outputs/weight_network_residual/weight_net_final.pt")

OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/videos/residual_lam0.1_coeff0.01")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 영상용은 무조건 1개만
N_EPISODES = 1
BATCH_SIZE = 1
VIDEO_FPS = 30


class ResidualWeightNetwork(nn.Module):
    """Stage 2 residual weight network와 동일한 구조."""

    def __init__(
        self,
        L,
        state_dim,
        action_dim,
        te_coeff=0.01,
        d_model=128,
        n_heads=4,
        n_layers=2,
    ):
        super().__init__()
        self.L = L

        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.queries = nn.Embedding(L, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, 1)

        # residual 학습 때와 동일: TE prior + delta
        self.register_buffer(
            "te_logits",
            -te_coeff * torch.arange(L, dtype=torch.float32),
        )

    def forward(self, state, contribs):
        B = state.shape[0]

        state_tok = self.state_proj(state).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)

        memory = torch.cat([state_tok, chunk_toks], dim=1)

        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)

        delta = self.head(x).squeeze(-1)
        final_logits = self.te_logits.unsqueeze(0) + delta

        return final_logits, delta


class ResidualEnsemblePolicy:
    def __init__(self, baseline_policy, weight_net, L, te_coeff):
        self.baseline = baseline_policy
        self.weight_net = weight_net
        self.L = L
        self.te_coeff = te_coeff
        self.config = baseline_policy.config
        self.past_chunks = deque(maxlen=L)

    def reset(self):
        self.past_chunks.clear()

    def eval(self):
        self.baseline.eval()
        self.weight_net.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch):
        chunk = self.baseline.predict_action_chunk(batch)
        self.past_chunks.append(chunk)

        n = len(self.past_chunks)

        contribs = torch.stack(
            [self.past_chunks[-(i + 1)][:, i] for i in range(n)],
            dim=1,
        )

        # warm-up: 아직 L개 안 쌓였을 때는 fixed TE
        if n < self.L:
            decay_w = torch.exp(
                -self.te_coeff * torch.arange(n, device=contribs.device)
            )
            decay_w = decay_w / decay_w.sum()
            action = (decay_w.view(1, n, 1) * contribs).sum(dim=1)

            debug = {
                "n": n,
                "weight_max": float(decay_w.max().item()),
                "argmax": int(decay_w.argmax().item()),
                "effective_k": float(1.0 / (decay_w.pow(2).sum().item() + 1e-8)),
            }
            return action, debug

        state = batch["observation.state"]
        final_logits, delta = self.weight_net(state, contribs)
        weights = F.softmax(final_logits, dim=1)

        action = (weights.unsqueeze(-1) * contribs).sum(dim=1)

        w0 = weights[0]
        debug = {
            "n": n,
            "weight_max": float(w0.max().item()),
            "argmax": int(w0.argmax().item()),
            "effective_k": float(1.0 / (w0.pow(2).sum().item() + 1e-8)),
        }

        return action, debug


def clean_frame(frame):
    """env.render() 결과를 imageio가 저장 가능한 RGB array로 정리."""
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

    # float image면 uint8로 변환
    if frame.dtype != np.uint8:
        if frame.max() <= 1.0:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)
        else:
            frame = frame.clip(0, 255).astype(np.uint8)

    return frame


def main():
    print("=" * 80)
    print("Render residual learned ensemble video")
    print("=" * 80)

    print(f"Loading baseline from {BASELINE_DIR}")
    baseline = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    print(f"Loading weight network from {WEIGHT_NET_PATH}")
    ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)

    L = int(ckpt["L"])
    te_coeff = float(ckpt["te_coeff"])

    weight_net = ResidualWeightNetwork(
        L=L,
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        te_coeff=te_coeff,
        d_model=int(ckpt["d_model"]),
        n_heads=int(ckpt["n_heads"]),
        n_layers=int(ckpt["n_layers"]),
    ).to(DEVICE)

    weight_net.load_state_dict(ckpt["model"])
    weight_net.eval()

    print(f"  L={L}")
    print(f"  te_coeff={te_coeff}")
    print(f"  lambda_delta_reg={ckpt.get('lambda_delta_reg', '?')}")

    policy = ResidualEnsemblePolicy(
        baseline_policy=baseline,
        weight_net=weight_net,
        L=L,
        te_coeff=te_coeff,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    from lerobot.envs.configs import AlohaEnv

    env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
    env_obj = make_env(env_cfg, n_envs=BATCH_SIZE, use_async_envs=False)

    while isinstance(env_obj, dict):
        print("make_env returned dict keys:", env_obj.keys())
        env_obj = list(env_obj.values())[0]

    env = env_obj
    print("final env type:", type(env))

    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    seeds = [0]
    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(BATCH_SIZE, dtype=bool)
    successes = np.zeros(BATCH_SIZE, dtype=bool)

    try:
        max_steps = env.call("_max_episode_steps")[0]
    except Exception:
        max_steps = 400

    print(f"max_steps={max_steps}")

    policy.reset()
    frames = []

    # reset 직후 첫 프레임도 저장 시도
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
            action, debug = policy.select_action(obs)

        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        action_np = action.to("cpu").numpy()

        obs_dict, reward, terminated, truncated, info = env.step(action_np)

        # step 이후 frame 저장
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
                f"wmax={debug['weight_max']:.3f}, "
                f"argmax={debug['argmax']}, "
                f"effK={debug['effective_k']:.2f}"
            )

    success = bool(successes[0])
    print(f"\nFinished episode.")
    print(f"  steps={step}")
    print(f"  success={success}")
    print(f"  frames={len(frames)}")

    if len(frames) == 0:
        print("[ERROR] No frames captured. env.render() did not return RGB frames.")
    else:
        video_path = OUT_DIR / f"residual_seed0_success{int(success)}.mp4"
        imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
        print(f"[VIDEO SAVED] {video_path}")

    env.close()


if __name__ == "__main__":
    main()