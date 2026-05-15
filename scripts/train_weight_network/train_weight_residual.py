"""Stage 2 v2: Residual weight network on TE prior."""
import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/weight_network_residual")
OUT_DIR.mkdir(parents=True, exist_ok=True)

L = 100
TE_COEFF = 0.01
LAMBDA_DELTA_REG = 0.1

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
LR = 3e-4
STEPS = 50000
BATCH_SIZE = 64
LOG_FREQ = 100
SAVE_FREQ = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualWeightNetwork(nn.Module):
    def __init__(self, L, state_dim, action_dim, te_coeff=0.01,
                 d_model=128, n_heads=4, n_layers=2):
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

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

        self.register_buffer(
            "te_logits",
            -te_coeff * torch.arange(L, dtype=torch.float32)
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

chunks_cpu = torch.from_numpy(cache["chunks"].copy()).float()
states_cpu = torch.from_numpy(cache["states"].copy()).float()
actions_cpu = torch.from_numpy(cache["actions"].copy()).float()
ep_starts = cache["ep_starts"].astype(np.int64)

N, chunk_size, action_dim = chunks_cpu.shape
state_dim = states_cpu.shape[1]
print(f"  N={N}, chunk_size={chunk_size}, state_dim={state_dim}, action_dim={action_dim}")
print(f"  cache size: chunks={chunks_cpu.element_size() * chunks_cpu.nelement() / 1e6:.1f}MB")

ep_lens = ep_starts[:, 1] - ep_starts[:, 0]
print(
    f"  episode length: min={ep_lens.min()}, "
    f"max={ep_lens.max()}, mean={ep_lens.mean():.0f}"
)
assert ep_lens.min() >= L, f"Some episodes too short for L={L}!"


def sample_batch(batch_size):
    valid_eps = np.where(ep_lens >= L)[0]
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
    targets = actions_cpu[target_idx]

    return states.to(DEVICE), contribs.to(DEVICE), targets.to(DEVICE)


model = ResidualWeightNetwork(
    L=L,
    state_dim=state_dim,
    action_dim=action_dim,
    te_coeff=TE_COEFF,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

wandb.init(
    project="lerobot",
    name=f"weight_residual_L{L}_lambda{LAMBDA_DELTA_REG}",
    config={
        "L": L,
        "lr": LR,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "te_coeff": TE_COEFF,
        "lambda_delta_reg": LAMBDA_DELTA_REG,
    },
)

with torch.no_grad():
    s, c, t = sample_batch(256)
    final_logits, delta = model(s, c)
    weights = F.softmax(final_logits, dim=1)

    init_l1 = F.l1_loss((weights.unsqueeze(-1) * c).sum(dim=1), t).item()
    init_delta_mse = (delta ** 2).mean().item()
    init_delta_abs = delta.abs().mean().item()
    init_weight_max = weights.max(dim=1).values.mean().item()
    init_entropy = -(weights * (weights + 1e-9).log()).sum(dim=1).mean().item()

    print("\nInitial state: TE prior + delta=0")
    print(f"  l1: {init_l1:.4f}")
    print(f"  delta_mse: {init_delta_mse:.8f}")
    print(f"  delta_abs_mean: {init_delta_abs:.8f}")
    print(f"  weight_max: {init_weight_max:.4f}")
    print(f"  entropy_ratio: {init_entropy / np.log(L):.4f}")

print(f"\nTraining {STEPS} steps with lambda_delta_reg={LAMBDA_DELTA_REG}...")
for step in tqdm(range(STEPS)):
    s, c, t = sample_batch(BATCH_SIZE)

    final_logits, delta = model(s, c)
    weights = F.softmax(final_logits, dim=1).unsqueeze(-1)
    ensembled = (weights * c).sum(dim=1)

    l1 = F.l1_loss(ensembled, t)
    delta_reg = (delta ** 2).mean()
    loss = l1 + LAMBDA_DELTA_REG * delta_reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % LOG_FREQ == 0:
        with torch.no_grad():
            w_flat = weights[:, :, 0]
            entropy = -(w_flat * (w_flat + 1e-9).log()).sum(dim=1).mean()
            wandb.log({
                "loss": loss.item(),
                "l1": l1.item(),
                "delta_reg": delta_reg.item(),
                "step": step,
                "weight_entropy": entropy.item(),
                "weight_entropy_ratio": entropy.item() / np.log(L),
                "weight_max": w_flat.max(dim=1).values.mean().item(),
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
                "logits_std_per_step": final_logits.std(dim=1).mean().item(),
            })

    if (step + 1) % SAVE_FREQ == 0:
        torch.save({
            "model": model.state_dict(),
            "step": step + 1,
            "L": L,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "te_coeff": TE_COEFF,
            "lambda_delta_reg": LAMBDA_DELTA_REG,
        }, OUT_DIR / f"weight_net_step{step + 1}.pt")

torch.save({
    "model": model.state_dict(),
    "step": STEPS,
    "L": L,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
    "te_coeff": TE_COEFF,
    "lambda_delta_reg": LAMBDA_DELTA_REG,
}, OUT_DIR / "weight_net_final.pt")

print(f"\nDone. Saved to {OUT_DIR / 'weight_net_final.pt'}")
