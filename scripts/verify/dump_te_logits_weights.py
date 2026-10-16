import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


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


def print_vector(name, x, max_items=15):
    x = np.asarray(x)
    print(f"\n[{name}] shape={x.shape}")
    print(" first:", np.array2string(x[:max_items], precision=6, suppress_small=False))
    print(" last :", np.array2string(x[-max_items:], precision=6, suppress_small=False))
    print(f" sum={x.sum():.6f}, max={x.max():.6f}, argmax={x.argmax()}, min={x.min():.6f}, argmin={x.argmin()}")


def save_csv(path, rows):
    import csv
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    keys = rows[0].keys()
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[SAVED] {path}")


def build_contribs_from_cache(chunks_cpu, ep_starts, ep_lens, L, batch_size, seed=0):
    rng = np.random.default_rng(seed)
    valid_eps = np.where(ep_lens >= L)[0]
    eps = rng.choice(valid_eps, size=batch_size, replace=True)

    win_starts = np.zeros(batch_size, dtype=np.int64)
    for i, ep in enumerate(eps):
        ep_from = ep_starts[ep, 0]
        ep_len = ep_lens[ep]
        offset = rng.integers(0, ep_len - L + 1)
        win_starts[i] = ep_from + offset

    # index 0 = 최신 chunk의 현재 action
    # index L-1 = 가장 오래된 chunk의 현재 action
    contribs_frame_idx = win_starts[:, None] + (L - 1) - np.arange(L)[None, :]
    pos_idx = np.broadcast_to(np.arange(L), (batch_size, L))

    contribs = chunks_cpu[contribs_frame_idx, pos_idx]

    target_idx = win_starts + L - 1
    return contribs, target_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--weight_net_path", type=str, default="")
    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--coeff", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="outputs/te_weight_dump.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L = args.L
    coeff = args.coeff

    print("=" * 80)
    print("TE / weight-network logits dump")
    print(f"cache_path      = {args.cache_path}")
    print(f"weight_net_path = {args.weight_net_path if args.weight_net_path else '(none)'}")
    print(f"L               = {L}")
    print(f"coeff           = {coeff}")
    print(f"device          = {device}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1) lerobot-style TE convention
    # ------------------------------------------------------------------
    # 기존 eval warmup 코드에서 쓰던 방식:
    # weight_i ∝ exp(-coeff * i)
    # i=0 최신, i=L-1 과거
    lerobot_logits = (-coeff * torch.arange(L, dtype=torch.float32)).numpy()
    lerobot_weights = torch.softmax(torch.from_numpy(lerobot_logits), dim=0).numpy()

    print_vector("LEROBOT_STYLE_TE logits = -coeff * arange(L)", lerobot_logits)
    print_vector("LEROBOT_STYLE_TE weights = softmax(logits)", lerobot_weights)

    # ------------------------------------------------------------------
    # 2) 우리 pure TE / weight-net prior convention
    # ------------------------------------------------------------------
    # 우리 residual weight net 코드:
    # te_logits = coeff * arange(L)
    # weight_i ∝ exp(coeff * i)
    # i=0 최신, i=L-1 과거
    our_te_logits = (coeff * torch.arange(L, dtype=torch.float32)).numpy()
    our_te_weights = torch.softmax(torch.from_numpy(our_te_logits), dim=0).numpy()

    print_vector("OUR_TE_PRIOR logits = coeff * arange(L)", our_te_logits)
    print_vector("OUR_TE_PRIOR weights = softmax(logits)", our_te_weights)

    # ------------------------------------------------------------------
    # 3) cache 로드
    # ------------------------------------------------------------------
    cache = np.load(args.cache_path)
    chunks_cpu = torch.from_numpy(cache["chunks"].copy()).float()
    states_cpu = torch.from_numpy(cache["states"].copy()).float()
    actions_cpu = torch.from_numpy(cache["actions"].copy()).float()
    vis_feats_cpu = torch.from_numpy(cache["vis_feats"].copy()).float()
    ep_starts = cache["ep_starts"].astype(np.int64)
    ep_lens = ep_starts[:, 1] - ep_starts[:, 0]

    N, chunk_size, action_dim = chunks_cpu.shape
    state_dim = states_cpu.shape[1]
    vis_dim = vis_feats_cpu.shape[1]

    print("\n[CACHE]")
    print(f" N={N}, chunk_size={chunk_size}, state_dim={state_dim}, action_dim={action_dim}, vis_dim={vis_dim}")

    contribs, target_idx = build_contribs_from_cache(
        chunks_cpu=chunks_cpu,
        ep_starts=ep_starts,
        ep_lens=ep_lens,
        L=L,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    states = states_cpu[target_idx].to(device)
    vis_feats = vis_feats_cpu[target_idx].to(device)
    targets = actions_cpu[target_idx].to(device)
    contribs = contribs.to(device)

    rows = []

    # pure TE action error도 같이 확인
    with torch.no_grad():
        lw = torch.from_numpy(lerobot_weights).float().to(device)
        ow = torch.from_numpy(our_te_weights).float().to(device)

        lerobot_action = (lw.view(1, L, 1) * contribs).sum(dim=1)
        our_te_action = (ow.view(1, L, 1) * contribs).sum(dim=1)

        lerobot_l1 = F.l1_loss(lerobot_action, targets).item()
        our_te_l1 = F.l1_loss(our_te_action, targets).item()

    print("\n[PURE TE L1 on sampled batch]")
    print(f" lerobot_style_TE_l1 = {lerobot_l1:.6f}")
    print(f" our_TE_prior_l1     = {our_te_l1:.6f}")

    for i in range(L):
        rows.append({
            "type": "lerobot_style_TE",
            "sample": -1,
            "idx": i,
            "age_meaning": "0_latest_to_Lminus1_oldest",
            "logit": float(lerobot_logits[i]),
            "delta": "",
            "final_logit": float(lerobot_logits[i]),
            "weight": float(lerobot_weights[i]),
        })
        rows.append({
            "type": "our_TE_prior",
            "sample": -1,
            "idx": i,
            "age_meaning": "0_latest_to_Lminus1_oldest",
            "logit": float(our_te_logits[i]),
            "delta": "",
            "final_logit": float(our_te_logits[i]),
            "weight": float(our_te_weights[i]),
        })

    # ------------------------------------------------------------------
    # 4) weight network logits / delta / weights
    # ------------------------------------------------------------------
    if args.weight_net_path:
        ckpt = torch.load(args.weight_net_path, map_location=device)

        model = ResidualWeightNetwork(
            L=ckpt["L"],
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            vis_dim=ckpt["vis_dim"],
            te_coeff=ckpt["te_coeff"],
            d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"],
        ).to(device)

        model.load_state_dict(ckpt["model"])
        model.eval()

        with torch.no_grad():
            final_logits, delta = model(states, vis_feats, contribs)
            weights = F.softmax(final_logits, dim=1)
            pred_action = (weights.unsqueeze(-1) * contribs).sum(dim=1)
            net_l1 = F.l1_loss(pred_action, targets).item()

        print("\n[WEIGHT NET]")
        print(f" ckpt step={ckpt.get('step')}")
        print(f" ckpt te_coeff={ckpt.get('te_coeff')}")
        print(f" ckpt lambda={ckpt.get('lambda_delta_reg')}")
        print(f" sampled_batch_l1={net_l1:.6f}")

        f0 = final_logits[0].detach().cpu().numpy()
        d0 = delta[0].detach().cpu().numpy()
        w0 = weights[0].detach().cpu().numpy()

        print_vector("WEIGHT_NET sample0 delta", d0)
        print_vector("WEIGHT_NET sample0 final_logits", f0)
        print_vector("WEIGHT_NET sample0 weights", w0)

        print("\n[WEIGHT NET stats over batch]")
        print(f" delta_abs_mean = {delta.abs().mean().item():.8f}")
        print(f" delta_abs_max  = {delta.abs().max().item():.8f}")
        print(f" weight_max_mean = {weights.max(dim=1).values.mean().item():.8f}")
        entropy = -(weights * (weights + 1e-9).log()).sum(dim=1).mean()
        print(f" entropy_ratio = {(entropy / np.log(L)).item():.8f}")

        for b in range(args.batch_size):
            fb = final_logits[b].detach().cpu().numpy()
            db = delta[b].detach().cpu().numpy()
            wb = weights[b].detach().cpu().numpy()

            for i in range(L):
                rows.append({
                    "type": "weight_net",
                    "sample": b,
                    "idx": i,
                    "age_meaning": "0_latest_to_Lminus1_oldest",
                    "logit": float(ckpt["te_coeff"] * i),
                    "delta": float(db[i]),
                    "final_logit": float(fb[i]),
                    "weight": float(wb[i]),
                })

    save_csv(args.out_csv, rows)


if __name__ == "__main__":
    main()
