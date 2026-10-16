import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--lambda_delta_reg", type=float, default=1.0)
parser.add_argument("--num_batches", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
args = parser.parse_args()

CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/acm_baseline_chunks/chunks.npz")
WEIGHT_NET_PATH = Path(
    f"/home1/eunji24/lerobot_project/outputs/weight_network_acm_residual/lambda{args.lambda_delta_reg}/weight_net_final.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


print("Loading cache...")
cache = np.load(CACHE_PATH)
chunks_cpu = torch.from_numpy(cache["chunks"]).float()
states_cpu = torch.from_numpy(cache["states"]).float()
ep_starts = cache["ep_starts"].astype(np.int64)

N, chunk_size, action_dim = chunks_cpu.shape
state_dim = states_cpu.shape[1]

print("Loading weight net:", WEIGHT_NET_PATH)
ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)

L = ckpt["L"]
model = ResidualWeightNetwork(
    L=L,
    state_dim=ckpt["state_dim"],
    action_dim=ckpt["action_dim"],
    te_coeff=ckpt["te_coeff"],
    d_model=ckpt["d_model"],
    n_heads=ckpt["n_heads"],
    n_layers=ckpt["n_layers"],
).to(DEVICE)

model.load_state_dict(ckpt["model"])
model.eval()

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

    return states.to(DEVICE), contribs.to(DEVICE)


all_fixed_wmax = []
all_res_wmax = []
all_fixed_effk = []
all_res_effk = []
all_fixed_argmax = []
all_res_argmax = []
all_delta_abs = []
all_action_diff = []

with torch.no_grad():
    fixed_logits = model.te_logits.unsqueeze(0)
    fixed_w = F.softmax(fixed_logits, dim=1).squeeze(0)

    for _ in range(args.num_batches):
        states, contribs = sample_batch(args.batch_size)

        final_logits, delta = model(states, contribs)
        res_w = F.softmax(final_logits, dim=1)

        fixed_w_b = fixed_w.unsqueeze(0).expand_as(res_w)

        fixed_action = (fixed_w_b.unsqueeze(-1) * contribs).sum(dim=1)
        res_action = (res_w.unsqueeze(-1) * contribs).sum(dim=1)

        all_fixed_wmax.append(fixed_w_b.max(dim=1).values.cpu())
        all_res_wmax.append(res_w.max(dim=1).values.cpu())

        all_fixed_effk.append((1.0 / (fixed_w_b ** 2).sum(dim=1)).cpu())
        all_res_effk.append((1.0 / (res_w ** 2).sum(dim=1)).cpu())

        all_fixed_argmax.append(fixed_w_b.argmax(dim=1).cpu())
        all_res_argmax.append(res_w.argmax(dim=1).cpu())

        all_delta_abs.append(delta.abs().mean(dim=1).cpu())
        all_action_diff.append((res_action - fixed_action).abs().mean(dim=1).cpu())

fixed_wmax = torch.cat(all_fixed_wmax)
res_wmax = torch.cat(all_res_wmax)
fixed_effk = torch.cat(all_fixed_effk)
res_effk = torch.cat(all_res_effk)
fixed_argmax = torch.cat(all_fixed_argmax)
res_argmax = torch.cat(all_res_argmax)
delta_abs = torch.cat(all_delta_abs)
action_diff = torch.cat(all_action_diff)

print("\n=== ACM weight spike / mode collapse stats ===")
print(f"lambda = {args.lambda_delta_reg}")
print(f"L = {L}")
print()
print(f"Fixed weight_max mean  : {fixed_wmax.mean().item():.4f}")
print(f"Residual weight_max mean: {res_wmax.mean().item():.4f}")
print(f"Residual weight_max max : {res_wmax.max().item():.4f}")
print()
print(f"Fixed effective_k median  : {fixed_effk.median().item():.2f}")
print(f"Residual effective_k median: {res_effk.median().item():.2f}")
print()
print(f"Fixed wmax > 0.5  : {(fixed_wmax > 0.5).float().mean().item()*100:.2f}%")
print(f"Residual wmax > 0.5: {(res_wmax > 0.5).float().mean().item()*100:.2f}%")
print()
print(f"delta_abs mean: {delta_abs.mean().item():.6f}")
print(f"action_diff mean |residual - fixedTE|: {action_diff.mean().item():.6f}")
print()

# argmax concentration
for name, arg in [("Fixed", fixed_argmax), ("Residual", res_argmax)]:
    hist = torch.bincount(arg, minlength=L).float()
    hist = hist / hist.sum()
    topv, topi = torch.topk(hist, k=5)
    print(f"{name} argmax top5:")
    for i, v in zip(topi.tolist(), topv.tolist()):
        print(f"  index {i}: {v*100:.2f}%")