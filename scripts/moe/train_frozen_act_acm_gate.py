import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm

ACT_CACHE = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")
ACM_CACHE = Path("/home1/eunji24/lerobot_project/cache/acm_60k_chunks_vis/chunks.npz")
OUT_DIR = Path("/home1/eunji24/lerobot_project/outputs/frozen_act_acm_gate")
OUT_DIR.mkdir(parents=True, exist_ok=True)

L = 100
D_MODEL = 128
BATCH_SIZE = 128
STEPS = 50000
LR = 3e-4

LAMBDA_PRIOR = 1.0
LAMBDA_SMOOTH = 0.1
PRIOR_W = 0.7

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrozenACTACMGate(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=128):
        super().__init__()

        in_dim = state_dim + action_dim * 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # sigmoid(logit) ~= 0.7로 시작
        prior_logit = torch.logit(torch.tensor(PRIOR_W))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, prior_logit.item())

    def forward(self, state, act_chunk, acm_chunk):
        """
        state:     [B, state_dim]
        act_chunk: [B, L, action_dim]
        acm_chunk: [B, L, action_dim]
        """

        B, L, A = act_chunk.shape

        state_rep = state[:, None, :].expand(B, L, -1)
        x = torch.cat([state_rep, act_chunk, acm_chunk], dim=-1)

        logit = self.net(x).squeeze(-1)   # [B, L]
        w = torch.sigmoid(logit)          # [B, L]

        final_chunk = w[..., None] * act_chunk + (1 - w[..., None]) * acm_chunk

        return final_chunk, w


def load_cache():
    act = np.load(ACT_CACHE)
    acm = np.load(ACM_CACHE)

    act_chunks = act["chunks"]
    acm_chunks = acm["chunks"]

    states = act["states"]
    actions = act["actions"]

    assert act_chunks.shape == acm_chunks.shape
    assert states.shape[0] == act_chunks.shape[0]
    assert actions.shape[0] == act_chunks.shape[0]

    return act_chunks, acm_chunks, states, actions


def sample_batch(act_chunks, acm_chunks, states, actions):
    N = act_chunks.shape[0]
    idx = np.random.randint(0, N, size=BATCH_SIZE)

    act_b = torch.tensor(act_chunks[idx], dtype=torch.float32, device=DEVICE)
    acm_b = torch.tensor(acm_chunks[idx], dtype=torch.float32, device=DEVICE)
    s_b = torch.tensor(states[idx], dtype=torch.float32, device=DEVICE)
    y_b = torch.tensor(actions[idx], dtype=torch.float32, device=DEVICE)

    return s_b, act_b, acm_b, y_b


def main():
    print("Loading cache...")
    act_chunks, acm_chunks, states, actions = load_cache()

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]

    model = FrozenACTACMGate(state_dim, action_dim, D_MODEL).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    for step in tqdm(range(1, STEPS + 1)):
        s, act_c, acm_c, y = sample_batch(act_chunks, acm_chunks, states, actions)

        final_chunk, w = model(s, act_c, acm_c)

        # 현재 frame에 해당하는 action은 chunk index 0 기준
        pred = final_chunk[:, 0, :]

        l1 = F.l1_loss(pred, y)

        prior_loss = ((w.mean(dim=1) - PRIOR_W) ** 2).mean()

        smooth_loss = (w[:, 1:] - w[:, :-1]).abs().mean()

        loss = l1 + LAMBDA_PRIOR * prior_loss + LAMBDA_SMOOTH * smooth_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            print(
                f"step={step} "
                f"loss={loss.item():.6f} "
                f"l1={l1.item():.6f} "
                f"prior={prior_loss.item():.6f} "
                f"smooth={smooth_loss.item():.6f} "
                f"w_mean={w.mean().item():.4f} "
                f"w_std={w.std().item():.4f}"
            )

        if step % 10000 == 0:
            ckpt = {
                "model": model.state_dict(),
                "state_dim": state_dim,
                "action_dim": action_dim,
                "L": L,
                "d_model": D_MODEL,
                "prior_w": PRIOR_W,
                "lambda_prior": LAMBDA_PRIOR,
                "lambda_smooth": LAMBDA_SMOOTH,
            }
            torch.save(ckpt, OUT_DIR / f"gate_step_{step}.pt")

    torch.save(
        {
            "model": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "L": L,
            "d_model": D_MODEL,
            "prior_w": PRIOR_W,
            "lambda_prior": LAMBDA_PRIOR,
            "lambda_smooth": LAMBDA_SMOOTH,
        },
        OUT_DIR / "gate_final.pt",
    )


if __name__ == "__main__":
    main()