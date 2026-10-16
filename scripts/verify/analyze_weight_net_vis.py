import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
WEIGHT_NET_PATH = Path("/home1/eunji24/lerobot_project/outputs/weight_network_residual/weight_net_final.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_BATCHES = 200
BATCH_SIZE = 256


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
            te_coeff * torch.arange(L, dtype=torch.float32)
        )

    def forward(self, state, image_feat, contribs):
        B = state.shape[0]

        state_tok = self.state_proj(state).unsqueeze(1)
        image_tok = self.image_proj(image_feat).unsqueeze(1)
        chunk_toks = self.chunk_proj(contribs)

        memory = torch.cat([state_tok, image_tok, chunk_toks], dim=1)

        q = self.queries.weight.unsqueeze(0).expand(B, -1, -1)
        x = self.decoder(q, memory)

        delta = self.head(x).squeeze(-1)
        final_logits = self.te_logits.unsqueeze(0) + delta
        return final_logits, delta


print("Loading cache...")
cache = np.load(CACHE_PATH)

chunks_cpu = torch.from_numpy(cache["chunks"].copy()).float()
states_cpu = torch.from_numpy(cache["states"].copy()).float()
actions_cpu = torch.from_numpy(cache["actions"].copy()).float()
vis_feats_cpu = torch.from_numpy(cache["vis_feats"].copy()).float()
ep_starts = cache["ep_starts"].astype(np.int64)

N, chunk_size, action_dim = chunks_cpu.shape
state_dim = states_cpu.shape[1]
vis_dim = vis_feats_cpu.shape[1]

print(f"N={N}, chunk_size={chunk_size}, state_dim={state_dim}, action_dim={action_dim}, vis_dim={vis_dim}")

print("Loading weight net...")
ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)

L = ckpt["L"]
model = ResidualWeightNetwork(
    L=L,
    state_dim=ckpt["state_dim"],
    action_dim=ckpt["action_dim"],
    vis_dim=ckpt["vis_dim"],
    te_coeff=ckpt["te_coeff"],
    d_model=ckpt["d_model"],
    n_heads=ckpt["n_heads"],
    n_layers=ckpt["n_layers"],
).to(DEVICE)

model.load_state_dict(ckpt["model"])
model.eval()

print(f"L={L}, step={ckpt.get('step')}, te_coeff={ckpt['te_coeff']}, lambda={ckpt.get('lambda_delta_reg')}")

ep_lens = ep_starts[:, 1] - ep_starts[:, 0]
valid_eps = np.where(ep_lens >= L)[0]


def sample_batch(batch_size):
    eps = np.random.choice(valid_eps, size=batch_size, replace=True)

    win_starts = np.zeros(batch_size, dtype=np.int64)
    for i, ep in enumerate(eps):
        ep_from = ep_starts[ep, 0]
        ep_len = ep_lens[ep]
        offset = np.random.randint(0, ep_len - L + 1)
        win_starts[i] = ep_from + offset

    contribs_frame_idx = win_starts[:, None] + (L - 1) - np.arange(L)[None, :]
    pos_idx = np.broadcast_to(np.arange(L), (batch_size, L))

    contribs = chunks_cpu[contribs_frame_idx, pos_idx]
    target_idx = win_starts + L - 1

    states = states_cpu[target_idx]
    vis_feats = vis_feats_cpu[target_idx]
    targets = actions_cpu[target_idx]

    return (
        states.to(DEVICE),
        vis_feats.to(DEVICE),
        contribs.to(DEVICE),
        targets.to(DEVICE),
    )


all_weight_max = []
all_entropy_ratio = []
all_argmax = []
all_delta_abs_mean = []
all_delta_abs_max = []
all_l1 = []
all_te_l1 = []
all_weight_mean = []

with torch.no_grad():
    te_logits = ckpt["te_coeff"] * torch.arange(L, device=DEVICE, dtype=torch.float32)
    te_weights = F.softmax(te_logits, dim=0)

    for _ in range(N_BATCHES):
        s, v, c, t = sample_batch(BATCH_SIZE)

        final_logits, delta = model(s, v, c)
        weights = F.softmax(final_logits, dim=1)

        ensembled = (weights.unsqueeze(-1) * c).sum(dim=1)
        te_ensembled = (te_weights.view(1, L, 1) * c).sum(dim=1)

        l1 = F.l1_loss(ensembled, t, reduction="none").mean(dim=1)
        te_l1 = F.l1_loss(te_ensembled, t, reduction="none").mean(dim=1)

        entropy = -(weights * (weights + 1e-9).log()).sum(dim=1)
        entropy_ratio = entropy / np.log(L)

        all_weight_max.append(weights.max(dim=1).values.cpu())
        all_entropy_ratio.append(entropy_ratio.cpu())
        all_argmax.append(weights.argmax(dim=1).cpu())
        all_delta_abs_mean.append(delta.abs().mean(dim=1).cpu())
        all_delta_abs_max.append(delta.abs().max(dim=1).values.cpu())
        all_l1.append(l1.cpu())
        all_te_l1.append(te_l1.cpu())
        all_weight_mean.append(weights.cpu())

weight_max = torch.cat(all_weight_max).numpy()
entropy_ratio = torch.cat(all_entropy_ratio).numpy()
argmax = torch.cat(all_argmax).numpy()
delta_abs_mean = torch.cat(all_delta_abs_mean).numpy()
delta_abs_max = torch.cat(all_delta_abs_max).numpy()
l1 = torch.cat(all_l1).numpy()
te_l1 = torch.cat(all_te_l1).numpy()
weight_mean = torch.cat(all_weight_mean).mean(dim=0).numpy()

print("\n========== Weight statistics ==========")
print(f"weight_max mean:       {weight_max.mean():.4f}")
print(f"weight_max p50/p90:    {np.percentile(weight_max, 50):.4f} / {np.percentile(weight_max, 90):.4f}")
print(f"entropy_ratio mean:    {entropy_ratio.mean():.4f}")
print(f"entropy_ratio p50/p10: {np.percentile(entropy_ratio, 50):.4f} / {np.percentile(entropy_ratio, 10):.4f}")
print(f"delta_abs_mean:        {delta_abs_mean.mean():.6f}")
print(f"delta_abs_max mean:    {delta_abs_max.mean():.6f}")
print(f"L1 learned:            {l1.mean():.6f}")
print(f"L1 pure TE:            {te_l1.mean():.6f}")
print(f"L1 improvement:        {(te_l1.mean() - l1.mean()):.6f}")

print("\n========== Argmax position ==========")
print(f"argmax mean index:     {argmax.mean():.2f}")
print(f"argmax p10/p50/p90:    {np.percentile(argmax, 10):.0f} / {np.percentile(argmax, 50):.0f} / {np.percentile(argmax, 90):.0f}")

hist = np.bincount(argmax, minlength=L)
top = hist.argsort()[::-1][:10]

print("\nTop-10 argmax positions:")
for i in top:
    print(f"  idx={i:3d}, count={hist[i]:6d}, ratio={hist[i] / hist.sum():.4f}")

print("\n========== Mean learned weights ==========")
for i in list(range(10)) + list(range(L - 10, L)):
    print(f"idx={i:3d}, mean_w={weight_mean[i]:.6f}")

print("\nInterpretation hint:")
print("  - entropy_ratio ~1.0 and low weight_max => almost uniform / weak selection")
print("  - delta_abs_mean near 0 => almost pure TE")
print("  - argmax concentrated at one index => fixed decay-like behavior")
print("  - learned L1 close to TE L1 => network is not changing much")
