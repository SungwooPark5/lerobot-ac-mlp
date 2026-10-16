"""Stage 3: residual weight network로 baseline ACT eval."""
import sys
import argparse
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from pathlib import Path

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from pathlib import Path
import imageio.v2 as imageio

BASELINE_DIR = Path("/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model")
WEIGHT_NET_PATH = Path("/home1/eunji24/lerobot_project/outputs/weight_network_residual_vis_lam5/weight_net_final.pt")

N_EPISODES = 50
BATCH_SIZE = 50
TE_COEFF = 0.01
DEVICE = torch.device("cuda")

"""
SAVE_VIDEO = False
VIDEO_DIR = Path("/home1/eunji24/lerobot_project/outputs/videos/weightnet_vis_lam5")
VIDEO_EPISODES = 5
VIDEO_FPS = 30
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
"""


class ResidualWeightNetwork(nn.Module):
    def __init__(self, L, state_dim, action_dim, vis_dim, te_coeff=0.01,
                 d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.L = L
        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, d_model),
            nn.Dropout(p=0.1),
        )
        self.queries = nn.Embedding(L, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.head = nn.Linear(d_model, 1)

        self.register_buffer(
            "te_logits",
            +te_coeff * torch.arange(L, dtype=torch.float32)
        )

    def forward(self, state, image_feat, contribs):
        B = state.shape[0]
        state_tok = self.state_proj(state).unsqueeze(1)          # [B, 1, D]
        image_tok = self.image_proj(image_feat).unsqueeze(1)     # [B, 1, D]
        chunk_toks = self.chunk_proj(contribs)                   # [B, L, D]
        memory = torch.cat([state_tok, image_tok, chunk_toks], dim=1)  # [B, L+2, D]

        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)

        delta = self.head(x).squeeze(-1)
        final_logits = self.te_logits.unsqueeze(0) + delta
        return final_logits, delta


class EnsemblePolicy:
    def __init__(self, baseline_policy, weight_net, L, te_coeff=0.01,
                 backbone=None, image_key="observation.images.top"):
        self.baseline = baseline_policy
        self.weight_net = weight_net
        self.L = L
        self.te_coeff = te_coeff
        self.config = baseline_policy.config
        self.past_chunks = deque(maxlen=L)

        self.backbone = backbone if backbone is not None else baseline_policy.model.backbone
        self.backbone.eval()
        self.image_key = image_key

        self.latest_feat_map = None
        self.debug_hook_compare_done = False

        def _save_backbone_output(module, inputs, output):
            self.latest_feat_map = output

        self.hook_handle = self.backbone.register_forward_hook(_save_backbone_output)

    def reset(self):
        self.past_chunks.clear()

    def eval(self):
        self.baseline.eval()
        self.weight_net.eval()
        self.backbone.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch):
        chunk = self.baseline.predict_action_chunk(batch)


        self.past_chunks.append(chunk)
        n = len(self.past_chunks)

        contribs = torch.stack([
            self.past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        
        if n < self.L:
            decay_w = torch.exp(
                +self.te_coeff * torch.arange(n, device=contribs.device)
            )
            decay_w = decay_w / decay_w.sum()
            action = (decay_w.view(1, n, 1) * contribs).sum(dim=1)
        else:
            state = batch["observation.state"]

            out = self.latest_feat_map
            if out is None:
                raise RuntimeError("Backbone hook did not capture any feature map.")

            if isinstance(out, torch.Tensor):
                feat_map = out
            elif isinstance(out, dict):
                feat_map = out["feature_map"]
            elif isinstance(out, (list, tuple)):
                feat_map = out[-1]
            else:
                raise TypeError(f"Unsupported backbone output type: {type(out)}")

            if feat_map.ndim == 4:
                vis_feat = feat_map.mean(dim=(-2, -1))  # [B, C]
            elif feat_map.ndim == 3:
                vis_feat = feat_map.mean(dim=1)         # [B, C]
            elif feat_map.ndim == 2:
                vis_feat = feat_map
            else:
                raise ValueError(f"Unexpected feat_map shape: {feat_map.shape}")

            final_logits, _ = self.weight_net(state, vis_feat, contribs)
            weights = F.softmax(final_logits, dim=1).unsqueeze(-1)
            action = (weights * contribs).sum(dim=1)
        
        return action

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_net_path", type=str, default=str(WEIGHT_NET_PATH))
    parser.add_argument("--eval_episodes", type=int, default=N_EPISODES)
    parser.add_argument("--eval_batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--state_noise_std", type=float, default=0.0)
    parser.add_argument("--state_noise_ratio", type=float, default=0.0)
    parser.add_argument("--image_noise_std", type=float, default=0.0)
    parser.add_argument("--image_noise_prob", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    return parser.parse_args()


def add_observation_noise(obs, args, step=None):
    # ===== state noise =====
    state_key = "observation.state"
    if state_key in obs and torch.is_tensor(obs[state_key]):
        if args.state_noise_std > 0.0 or args.state_noise_ratio > 0.0:
            state = obs[state_key]
            noise = torch.zeros_like(state)

            if args.state_noise_std > 0.0:
                noise = noise + args.state_noise_std * torch.randn_like(state)

            if args.state_noise_ratio > 0.0:
                scale = state.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
                noise = noise + args.state_noise_ratio * scale * torch.randn_like(state)

            obs[state_key] = state + noise

            if step == 0:
                print("[STATE NOISE DEBUG]")
                print("state mean abs:", state.abs().mean().item())
                print("state std mean:", state.std(dim=0).mean().item())
                print("noise mean abs:", noise.abs().mean().item())
                print("noise/state abs ratio:", (noise.abs().mean() / state.abs().mean().clamp_min(1e-6)).item())

    # ===== image noise =====
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

            img_min = img.min().item()
            img_max = img.max().item()

            # 0~1 image면 clamp, normalize image면 clamp 안 함
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
    print(f"Loading baseline from {BASELINE_DIR}")
    baseline = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    print(f"Loading weight network from {args.weight_net_path}")
    ckpt = torch.load(Path(args.weight_net_path), map_location=DEVICE)
    L = ckpt["L"]
    vis_dim = ckpt["vis_dim"]
    image_key = ckpt.get("image_key", "observation.images.top")

    weight_net = ResidualWeightNetwork(
        L=L,
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        vis_dim=vis_dim,
        te_coeff=ckpt["te_coeff"],
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
    ).to(DEVICE)

    weight_net.load_state_dict(ckpt["model"])
    weight_net.eval()
    print(f"  L={L}, trained {ckpt['step']} steps, te_coeff={ckpt['te_coeff']}, vis_dim={vis_dim}, image_key={image_key}")

    policy = EnsemblePolicy(
        baseline,
        weight_net,
        L=L,
        te_coeff=TE_COEFF,
        backbone=baseline.model.backbone,
        image_key=image_key,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
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

    print(f"\nRunning rollout: {args.eval_episodes} episodes, {args.eval_batch_size} parallel envs...")
    seeds = list(range(args.eval_episodes))[:args.eval_batch_size]
    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(args.eval_batch_size, dtype=bool)
    successes = np.zeros(args.eval_batch_size, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    policy.reset()

    step = 0
    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(obs_dict)
        obs = add_envs_task(env, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        # baseline ACT가 chunk를 예측하기 전 observation에 noise 추가
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
            print(f"  step {step}/{max_steps}, done {done.sum()}/{BATCH_SIZE}")

    pc_success = successes.mean()

    print("\nResults: image-feature residual POS direction eval")
    print(f"  pc_success = {pc_success:.3f} ({successes.sum()}/{args.eval_batch_size})")
    print(f"  state_noise_std = {args.state_noise_std}")
    print(f"  state_noise_ratio = {args.state_noise_ratio}")
    print(f"  image_noise_std = {args.image_noise_std}")
    print(f"  image_noise_prob = {args.image_noise_prob}")
    print("  Compare:")
    print("    baseline n=50:        62%")
    print("    baseline n=1 + TE:    38%")
    print("    Plan C (L=10):        0%")
    print("    decoder L=100:        24%")
    print(f"    residual L={L}:       {pc_success * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
