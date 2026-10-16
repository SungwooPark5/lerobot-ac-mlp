"""Stage 2: TE-fixed visual action residual network.

final_action = fixed_TE_action + alpha_residual * delta_action

Inputs:
  state
  cached visual feature
  contribs from past ACT chunks

Output:
  delta_action [B, action_dim]
"""

import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None


DEFAULT_CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
DEFAULT_OUT_ROOT = Path("/home1/eunji24/lerobot_project/outputs/te_action_residual_act_vis")


class VisualActionResidualNetwork(nn.Module):
    def __init__(
        self,
        L,
        state_dim,
        action_dim,
        vis_dim,
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_delta=1.0,
        dropout=0.1,
    ):
        super().__init__()
        self.L = L
        self.max_delta = max_delta

        self.state_proj = nn.Linear(state_dim, d_model)
        self.chunk_proj = nn.Linear(action_dim, d_model)
        self.image_proj = nn.Sequential(
            nn.LayerNorm(vis_dim),
            nn.Linear(vis_dim, d_model),
            nn.Dropout(p=dropout),
        )

        self.query = nn.Embedding(1, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, action_dim),
        )

        # 중요: 시작점이 pure TE가 되도록 delta_action = 0에서 시작
        last = self.head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, state, image_feat, contribs):
        # state:      [B, state_dim]
        # image_feat: [B, vis_dim]
        # contribs:   [B, L, action_dim]
        B = state.shape[0]

        state_tok = self.state_proj(state).unsqueeze(1)       # [B, 1, D]
        image_tok = self.image_proj(image_feat).unsqueeze(1)  # [B, 1, D]
        chunk_toks = self.chunk_proj(contribs)                # [B, L, D]

        memory = torch.cat([state_tok, image_tok, chunk_toks], dim=1)  # [B, L+2, D]

        q = self.query.weight.unsqueeze(0).expand(B, -1, -1)  # [B, 1, D]
        x = self.decoder(q, memory).squeeze(1)                # [B, D]

        raw_delta = self.head(x)                              # [B, action_dim]
        delta_action = torch.tanh(raw_delta) * self.max_delta
        return delta_action


def make_te_weights(L, te_coeff, device):
    # runtime convention: contribs[:, 0] = freshest
    # positive te_coeff => fresh favored
    w = torch.exp(-te_coeff * torch.arange(L, device=device, dtype=torch.float32))
    w = w / w.sum()
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--out_root", type=str, default=str(DEFAULT_OUT_ROOT))

    parser.add_argument("--L", type=int, default=100)
    parser.add_argument("--te_coeff", type=float, default=0.01)
    parser.add_argument("--alpha_residual", type=float, default=0.1)
    parser.add_argument("--lambda_residual", type=float, default=0.1)
    parser.add_argument("--max_delta", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=10000)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="lerobot")
    parser.add_argument("--wandb_name", type=str, default=None)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_root) / (
        f"alpha{args.alpha_residual}_lambda{args.lambda_residual}"
        f"_vis_freshfirst"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cache...")
    cache = np.load(args.cache_path)

    required = ["chunks", "states", "actions", "vis_feats", "ep_starts"]
    for k in required:
        if k not in cache:
            raise KeyError(f"cache에 '{k}'가 없음. 현재 keys={list(cache.keys())}")

    chunks_cpu = torch.from_numpy(cache["chunks"].copy()).float()
    states_cpu = torch.from_numpy(cache["states"].copy()).float()
    actions_cpu = torch.from_numpy(cache["actions"].copy()).float()
    vis_feats_cpu = torch.from_numpy(cache["vis_feats"].copy()).float()

    ep_starts = cache["ep_starts"].astype(np.int64)

    N, chunk_size, action_dim = chunks_cpu.shape
    state_dim = states_cpu.shape[1]
    vis_dim = vis_feats_cpu.shape[-1]

    ep_lens = ep_starts[:, 1] - ep_starts[:, 0]

    print(f"N={N}, chunk_size={chunk_size}, state_dim={state_dim}, action_dim={action_dim}")
    print(f"vis_feats={tuple(vis_feats_cpu.shape)}, vis_dim={vis_dim}")
    print(f"episode length: min={ep_lens.min()}, max={ep_lens.max()}, mean={ep_lens.mean():.0f}")

    assert ep_lens.min() >= args.L, f"Some episodes too short for L={args.L}!"

    te_weights = make_te_weights(args.L, args.te_coeff, device)
    print("\n[TE DEBUG]")
    print("Runtime contrib convention: contribs[:, 0] = freshest")
    print("te_coeff:", args.te_coeff)
    print("te_weights first 5:", te_weights[:5].detach().cpu().numpy())
    print("te_weights last 5:", te_weights[-5:].detach().cpu().numpy())
    print("te_weights argmax:", te_weights.argmax().item())

    def sample_batch(batch_size):
        valid_eps = np.where(ep_lens >= args.L)[0]
        eps = np.random.choice(valid_eps, size=batch_size, replace=True)

        win_starts = np.zeros(batch_size, dtype=np.int64)
        for i, ep in enumerate(eps):
            ep_from = ep_starts[ep, 0]
            ep_len = ep_lens[ep]
            offset = np.random.randint(0, ep_len - args.L + 1)
            win_starts[i] = ep_from + offset

        # contribs[:, 0] = freshest
        # target_idx = current timestep
        contribs_frame_idx = win_starts[:, None] + (args.L - 1) - np.arange(args.L)[None, :]
        pos_idx = np.broadcast_to(np.arange(args.L), (batch_size, args.L))

        contribs = chunks_cpu[contribs_frame_idx, pos_idx]

        target_idx = win_starts + args.L - 1
        states = states_cpu[target_idx]
        vis_feats = vis_feats_cpu[target_idx]
        targets = actions_cpu[target_idx]

        return (
            states.to(device),
            vis_feats.to(device),
            contribs.to(device),
            targets.to(device),
        )

    model = VisualActionResidualNetwork(
        L=args.L,
        state_dim=state_dim,
        action_dim=action_dim,
        vis_dim=vis_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_delta=args.max_delta,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb import 실패")
        wandb_name = args.wandb_name or f"te_action_residual_act_vis_alpha{args.alpha_residual}_lambda{args.lambda_residual}"
        wandb.init(
            project=args.wandb_project,
            name=wandb_name,
            config=vars(args) | {
                "state_dim": state_dim,
                "action_dim": action_dim,
                "vis_dim": vis_dim,
            },
        )

    with torch.no_grad():
        s, v, c, t = sample_batch(256)
        a_te = (te_weights.view(1, args.L, 1) * c).sum(dim=1)
        delta_a = model(s, v, c)
        pred = a_te + args.alpha_residual * delta_a

        init_l1_te = F.l1_loss(a_te, t).item()
        init_l1_pred = F.l1_loss(pred, t).item()
        print("\nInitial state:")
        print(f"  pure TE l1: {init_l1_te:.6f}")
        print(f"  pred l1:    {init_l1_pred:.6f}")
        print(f"  delta abs mean: {delta_a.abs().mean().item():.8f}")
        print(f"  alpha*delta abs mean: {(args.alpha_residual * delta_a).abs().mean().item():.8f}")

    print(f"\nTraining {args.steps} steps...")
    for step in tqdm(range(args.steps)):
        s, v, c, t = sample_batch(args.batch_size)

        a_te = (te_weights.view(1, args.L, 1) * c).sum(dim=1)
        delta_a = model(s, v, c)
        pred = a_te + args.alpha_residual * delta_a

        l1 = F.l1_loss(pred, t)
        te_l1 = F.l1_loss(a_te, t)
        delta_reg = (delta_a ** 2).mean()
        smooth_abs = (pred - a_te).abs().mean()

        loss = l1 + args.lambda_residual * delta_reg

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_freq == 0:
            log = {
                "loss": loss.item(),
                "l1": l1.item(),
                "te_l1": te_l1.item(),
                "delta_reg": delta_reg.item(),
                "delta_abs_mean": delta_a.abs().mean().item(),
                "delta_abs_max": delta_a.abs().max().item(),
                "alpha_delta_abs_mean": (args.alpha_residual * delta_a).abs().mean().item(),
                "pred_minus_te_abs_mean": smooth_abs.item(),
                "step": step,
            }
            if args.use_wandb:
                wandb.log(log)
            if step % (args.log_freq * 10) == 0:
                print(log)

        if (step + 1) % args.save_freq == 0:
            save_path = out_dir / f"weight_net_step{step + 1}.pt"
            torch.save({
                "model": model.state_dict(),
                "step": step + 1,
                "L": args.L,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "vis_dim": vis_dim,
                "image_key": "observation.images.top",
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "te_coeff": args.te_coeff,
                "alpha_residual": args.alpha_residual,
                "lambda_residual": args.lambda_residual,
                "max_delta": args.max_delta,
                "base": "act",
                "fresh_index": "first",
                "type": "te_fixed_visual_action_residual",
            }, save_path)

    final_path = out_dir / "weight_net_final.pt"
    torch.save({
        "model": model.state_dict(),
        "step": args.steps,
        "L": args.L,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "vis_dim": vis_dim,
        "image_key": "observation.images.top",
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "te_coeff": args.te_coeff,
        "alpha_residual": args.alpha_residual,
        "lambda_residual": args.lambda_residual,
        "max_delta": args.max_delta,
        "base": "act",
        "fresh_index": "first",
        "type": "te_fixed_visual_action_residual",
    }, final_path)

    print(f"\nDone. Saved to {final_path}")


if __name__ == "__main__":
    main()
