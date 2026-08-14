import os, sys
from pathlib import Path
from collections import deque

sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio.v2 as imageio

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.envs.configs import AlohaEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_FPS = 30
N_EPISODES = 10
BATCH_SIZE = 1

BASELINE_DIR = Path("/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model")

RUNS = [
    "weight_network_residual_lam001",
    "weight_network_residual_lam1",
]

class ResidualWeightNetwork(nn.Module):
    def __init__(self, L, state_dim, action_dim, te_coeff=0.01, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.L = L
        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.queries = nn.Embedding(L, d_model)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1)
        self.register_buffer("te_logits", -te_coeff * torch.arange(L, dtype=torch.float32))

    def forward(self, state, contribs):
        B = state.shape[0]
        state_tok = self.state_proj(state).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)
        memory = torch.cat([state_tok, chunk_toks], dim=1)
        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)
        delta = self.head(x).squeeze(-1)
        return self.te_logits.unsqueeze(0) + delta

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

        if n < self.L:
            decay_w = torch.exp(-self.te_coeff * torch.arange(n, device=contribs.device))
            decay_w = decay_w / decay_w.sum()
            return (decay_w.view(1, n, 1) * contribs).sum(dim=1)

        state = batch["observation.state"]
        logits = self.weight_net(state, contribs)
        weights = F.softmax(logits, dim=1)
        return (weights.unsqueeze(-1) * contribs).sum(dim=1)

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

def get_success(info, successes):
    if "final_info" not in info:
        return successes
    final_info = info["final_info"]
    if isinstance(final_info, dict):
        if "is_success" in final_info:
            successes |= np.array(final_info["is_success"], dtype=bool)
    else:
        for i, fi in enumerate(final_info):
            if isinstance(fi, dict) and fi.get("is_success", False):
                successes[i] = True
    return successes

def main():
    os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")

    print("=" * 80)
    print("Loading baseline")
    print("=" * 80)

    baseline = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
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
        policy_cfg=config,
    )

    try:
        max_steps = env.call("_max_episode_steps")[0]
    except Exception:
        max_steps = 400

    print(f"max_steps={max_steps}")

    for run_name in RUNS:
        print("\\n" + "=" * 80)
        print(f"RUN: {run_name}")
        print("=" * 80)

        ckpt_path = Path(f"/home1/eunji24/lerobot_project/outputs/{run_name}/weight_net_final.pt")
        print(f"Loading weight net: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        weight_net = ResidualWeightNetwork(
            L=int(ckpt["L"]),
            state_dim=int(ckpt["state_dim"]),
            action_dim=int(ckpt["action_dim"]),
            te_coeff=float(ckpt["te_coeff"]),
            d_model=int(ckpt["d_model"]),
            n_heads=int(ckpt["n_heads"]),
            n_layers=int(ckpt["n_layers"]),
        ).to(DEVICE)

        weight_net.load_state_dict(ckpt["model"])
        weight_net.eval()

        policy = ResidualEnsemblePolicy(
            baseline_policy=baseline,
            weight_net=weight_net,
            L=int(ckpt["L"]),
            te_coeff=float(ckpt["te_coeff"]),
        )

        out_dir = Path(f"/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/eval/{run_name}/videos/aloha_0")
        out_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0

        for ep in range(N_EPISODES):
            print(f"\\nEpisode {ep}/9")

            obs_dict, _ = env.reset(seed=[ep])
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
                successes = get_success(info, successes)

                step += 1

                if step % 50 == 0:
                    print(f"step {step}/{max_steps}, done={done.sum()}, success={successes[0]}")

            success = bool(successes[0])
            success_count += int(success)

            video_path = out_dir / f"eval_episode_{ep}.mp4"

            if len(frames) == 0:
                print("[ERROR] No frames captured.")
            else:
                imageio.mimsave(video_path, frames, fps=VIDEO_FPS)
                print(f"[VIDEO SAVED] {video_path}")

            print(f"episode={ep}, steps={step}, success={success}, frames={len(frames)}")

        print(f"\\n[{run_name}] success_rate={success_count}/{N_EPISODES} = {success_count / N_EPISODES:.2f}")

    env.close()

if __name__ == "__main__":
    main()
