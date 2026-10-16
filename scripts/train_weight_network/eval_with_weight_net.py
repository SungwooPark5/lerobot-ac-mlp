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
parser = argparse.ArgumentParser()
parser.add_argument("--weight_net_path", type=str, required=True)
args = parser.parse_args()

WEIGHT_NET_PATH = Path(args.weight_net_path)

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
                 d_model=128, n_heads=4, n_enc_layers=2, n_dec_layers=4):
        super().__init__()
        self.L = L
        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, d_model),
        )
        self.queries = nn.Embedding(L, d_model)
        self.chunk_pos = nn.Embedding(L, d_model)
        self.tok_type = nn.Embedding(3, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=0.0,
            dim_feedforward=4 * d_model, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads, dropout=0.0,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_dec_layers)
        self.head = nn.Linear(d_model, 1)

        self.register_buffer(
            "te_logits",
            -te_coeff * torch.arange(L, dtype=torch.float32)
        )

    def forward(self, state, image_feat, contribs):
        B = state.shape[0]
        type_emb = self.tok_type.weight  # [3, D]

        state_tok = self.state_proj(state).unsqueeze(1) + type_emb[0].view(1, 1, -1)
        image_tok = self.image_proj(image_feat).unsqueeze(1) + type_emb[1].view(1, 1, -1)
        chunk_toks = (self.chunk_proj(contribs)
                        + self.chunk_pos.weight.unsqueeze(0)
                        + type_emb[2].view(1, 1, -1))
        
        memory = torch.cat([state_tok, image_tok, chunk_toks], dim=1)  # [B, L+2, D]
        memory = self.encoder(memory)

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
                -self.te_coeff * torch.arange(n, device=contribs.device)
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

            final_logits, delta = self.weight_net(state, vis_feat, contribs)
            w_flat = F.softmax(final_logits, dim=1)  # [B, L]

            if not hasattr(self, "debug_weight_printed"):
                self.debug_weight_printed = False

            if not self.debug_weight_printed:
                print("\n[WEIGHT DEBUG]")
                print("final_logits[0, :10] =", final_logits[0, :10].detach().cpu().numpy())
                print("final_logits[0, -10:] =", final_logits[0, -10:].detach().cpu().numpy())
                print("delta[0, :10] =", delta[0, :10].detach().cpu().numpy())
                print("delta[0, -10:] =", delta[0, -10:].detach().cpu().numpy())
                print("weights[0, :10] =", w_flat[0, :10].detach().cpu().numpy())
                print("weights[0, -10:] =", w_flat[0, -10:].detach().cpu().numpy())
                print("argmax =", int(w_flat[0].argmax().item()))
                print("max =", float(w_flat[0].max().item()))
                print("entropy_ratio =", float((-(w_flat[0] * (w_flat[0] + 1e-9).log()).sum() / np.log(self.L)).item()))
                self.debug_weight_printed = True

            weights = w_flat.unsqueeze(-1)
            action = (weights * contribs).sum(dim=1)
        
        return action


def main():
    print(f"Loading baseline from {BASELINE_DIR}")
    baseline = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    print(f"Loading weight network from {WEIGHT_NET_PATH}")
    ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)
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
        n_enc_layers=ckpt["n_enc_layers"],
        n_dec_layers=ckpt["n_dec_layers"],
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

    print(f"\nRunning rollout: {N_EPISODES} episodes, {BATCH_SIZE} parallel envs...")
    seeds = list(range(N_EPISODES))[:BATCH_SIZE]
    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(BATCH_SIZE, dtype=bool)
    successes = np.zeros(BATCH_SIZE, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    policy.reset()

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

    print("\nResults:")
    print(f"  pc_success = {pc_success:.3f} ({successes.sum()}/{BATCH_SIZE})")
    print("  Compare:")
    print("    baseline n=50:        62%")
    print("    baseline n=1 + TE:    38%")
    print("    Plan C (L=10):        0%")
    print("    decoder L=100:        24%")
    print(f"    residual L={L}:       {pc_success * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
