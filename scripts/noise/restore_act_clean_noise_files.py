from pathlib import Path

train_code = r'''import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb


parser = argparse.ArgumentParser()
parser.add_argument("--lambda_delta_reg", type=float, default=0.1)
parser.add_argument("--steps", type=int, default=50000)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--seed", type=int, default=0)

# candidate corruption 실험용. 기본값 0이면 clean residual과 동일.
parser.add_argument("--noise_std", type=float, default=0.0)
parser.add_argument("--corrupt_prob", type=float, default=0.0)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

CACHE_PATH = Path("/home1/eunji24/lerobot_project/cache/baseline_chunks/chunks.npz")

L = 100
TE_COEFF = 0.01
LAMBDA_DELTA_REG = args.lambda_delta_reg

D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
LR = args.lr
STEPS = args.steps
BATCH_SIZE = 64
LOG_FREQ = 100
SAVE_FREQ = 10000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = Path(
    f"/home1/eunji24/lerobot_project/outputs/"
    f"weight_network_act_residual_clean_noise/"
    f"lambda{LAMBDA_DELTA_REG}_noise{args.noise_std}_p{args.corrupt_prob}"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


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

        # contribs[0] = freshest, contribs[L-1] = oldest
        # fresh-favored TE prior
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


print("Loading ACT cache...")
cache = np.load(CACHE_PATH)

chunks_cpu = torch.from_numpy(cache["chunks"].copy()).float()
states_cpu = torch.from_numpy(cache["states"].copy()).float()
actions_cpu = torch.from_numpy(cache["actions"].copy()).float()
ep_starts = cache["ep_starts"].astype(np.int64)

N, chunk_size, action_dim = chunks_cpu.shape
state_dim = states_cpu.shape[1]

print(f"N={N}, chunk_size={chunk_size}, state_dim={state_dim}, action_dim={action_dim}")
print(f"lambda={LAMBDA_DELTA_REG}, steps={STEPS}, lr={LR}")
print(f"noise_std={args.noise_std}, corrupt_prob={args.corrupt_prob}")
print(f"OUT_DIR={OUT_DIR}")

ep_lens = ep_starts[:, 1] - ep_starts[:, 0]
assert ep_lens.min() >= L, f"Some episodes too short for L={L}!"
valid_eps = np.where(ep_lens >= L)[0]


def add_candidate_corruption(contribs):
    if args.noise_std <= 0.0 or args.corrupt_prob <= 0.0:
        mask = torch.zeros((contribs.shape[0], contribs.shape[1], 1), dtype=torch.bool)
        return contribs, mask

    mask = torch.rand((contribs.shape[0], contribs.shape[1], 1)) < args.corrupt_prob
    noise = args.noise_std * torch.randn_like(contribs)
    return contribs + mask.float() * noise, mask


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
    contribs, corrupt_mask = add_candidate_corruption(contribs)

    target_idx = win_starts + L - 1
    states = states_cpu[target_idx]
    targets = actions_cpu[target_idx]

    return states.to(DEVICE), contribs.to(DEVICE), targets.to(DEVICE), corrupt_mask.to(DEVICE)


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
    name=(
        f"act_residual_clean_noise_L{L}"
        f"_lambda{LAMBDA_DELTA_REG}"
        f"_noise{args.noise_std}_p{args.corrupt_prob}"
    ),
    config={
        "model": "act",
        "L": L,
        "lr": LR,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "te_coeff": TE_COEFF,
        "te_direction": "fresh_favored",
        "lambda_delta_reg": LAMBDA_DELTA_REG,
        "noise_std": args.noise_std,
        "corrupt_prob": args.corrupt_prob,
        "out_dir": str(OUT_DIR),
    },
)

with torch.no_grad():
    s, c, t, m = sample_batch(256)
    final_logits, delta = model(s, c)
    weights = F.softmax(final_logits, dim=1)

    init_l1 = F.l1_loss((weights.unsqueeze(-1) * c).sum(dim=1), t).item()
    init_delta_abs = delta.abs().mean().item()
    init_weight_max = weights.max(dim=1).values.mean().item()
    init_entropy = -(weights * (weights + 1e-9).log()).sum(dim=1).mean().item()

    print("\nInitial state: TE prior + delta=0")
    print(f"  l1: {init_l1:.4f}")
    print(f"  delta_abs_mean: {init_delta_abs:.8f}")
    print(f"  weight_max: {init_weight_max:.4f}")
    print(f"  entropy_ratio: {init_entropy / np.log(L):.4f}")

print(f"\nTraining ACT residual clean/noise: {STEPS} steps, lambda={LAMBDA_DELTA_REG}")

for step in tqdm(range(STEPS)):
    s, c, t, corrupt_mask = sample_batch(BATCH_SIZE)

    final_logits, delta = model(s, c)
    weights_flat = F.softmax(final_logits, dim=1)
    weights = weights_flat.unsqueeze(-1)
    ensembled = (weights * c).sum(dim=1)

    l1 = F.l1_loss(ensembled, t)
    delta_reg = (delta ** 2).mean()
    loss = l1 + LAMBDA_DELTA_REG * delta_reg

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % LOG_FREQ == 0:
        with torch.no_grad():
            entropy = -(weights_flat * (weights_flat + 1e-9).log()).sum(dim=1).mean()

            errors = (c - t.unsqueeze(1)).abs().mean(dim=-1)
            pred_best_idx = weights_flat.argmax(dim=1)
            oracle_best_idx = errors.argmin(dim=1)
            best_match = (pred_best_idx == oracle_best_idx).float().mean()

            log_dict = {
                "loss": loss.item(),
                "l1": l1.item(),
                "delta_reg": delta_reg.item(),
                "step": step,
                "weight_entropy": entropy.item(),
                "weight_entropy_ratio": entropy.item() / np.log(L),
                "weight_max": weights_flat.max(dim=1).values.mean().item(),
                "delta_abs_mean": delta.abs().mean().item(),
                "delta_abs_max": delta.abs().max().item(),
                "logits_std_per_step": final_logits.std(dim=1).mean().item(),
                "oracle/best_match": best_match.item(),
                "oracle/best_error": errors.min(dim=1).values.mean().item(),
                "oracle/weighted_error": (weights_flat * errors).sum(dim=1).mean().item(),
                "oracle/uniform_error": errors.mean(dim=1).mean().item(),
            }

            mask_flat = corrupt_mask.squeeze(-1)
            if mask_flat.any() and (~mask_flat).any():
                corrupt_w = weights_flat[mask_flat].mean()
                clean_w = weights_flat[~mask_flat].mean()
                log_dict.update({
                    "noise/weight_clean_candidates": clean_w.item(),
                    "noise/weight_corrupt_candidates": corrupt_w.item(),
                    "noise/weight_clean_minus_corrupt": (clean_w - corrupt_w).item(),
                    "noise/corrupt_ratio": mask_flat.float().mean().item(),
                })

            wandb.log(log_dict)

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
            "te_direction": "fresh_favored",
            "lambda_delta_reg": LAMBDA_DELTA_REG,
            "noise_std": args.noise_std,
            "corrupt_prob": args.corrupt_prob,
            "base_policy": "act",
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
    "te_direction": "fresh_favored",
    "lambda_delta_reg": LAMBDA_DELTA_REG,
    "noise_std": args.noise_std,
    "corrupt_prob": args.corrupt_prob,
    "base_policy": "act",
}, OUT_DIR / "weight_net_final.pt")

print(f"\nDone. Saved to {OUT_DIR / 'weight_net_final.pt'}")
'''


eval_code = r'''import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
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
from lerobot.envs.configs import AlohaEnv


BASELINE_DIR = Path("/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="residual", choices=["pure_te", "residual"])
parser.add_argument("--weight_net_path", type=str, default=None)
parser.add_argument("--eval_episodes", type=int, default=50)
parser.add_argument("--eval_batch_size", type=int, default=50)
parser.add_argument("--state_noise_std", type=float, default=0.0)
parser.add_argument("--state_noise_ratio", type=float, default=0.0)
parser.add_argument("--noise_seed", type=int, default=0)
parser.add_argument("--L", type=int, default=100)
parser.add_argument("--te_coeff", type=float, default=0.01)
args = parser.parse_args()


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


class EnsemblePolicy:
    def __init__(self, baseline_policy, weight_net, L, mode="residual", te_coeff=0.01):
        self.baseline = baseline_policy
        self.weight_net = weight_net
        self.L = L
        self.mode = mode
        self.te_coeff = te_coeff
        self.config = baseline_policy.config
        self.past_chunks = deque(maxlen=L)

    def reset(self):
        self.past_chunks.clear()

    def eval(self):
        self.baseline.eval()
        if self.weight_net is not None:
            self.weight_net.eval()
        return self

    @torch.no_grad()
    def select_action(self, batch):
        chunk = self.baseline.predict_action_chunk(batch)
        self.past_chunks.append(chunk)
        n = len(self.past_chunks)

        contribs = torch.stack([
            self.past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        if self.mode == "pure_te" or n < self.L:
            decay_w = torch.exp(
                -self.te_coeff * torch.arange(n, device=contribs.device)
            )
            decay_w = decay_w / decay_w.sum()
            action = (decay_w.view(1, n, 1) * contribs).sum(dim=1)
            return action

        state = batch["observation.state"]
        final_logits, _ = self.weight_net(state, contribs)
        weights = F.softmax(final_logits, dim=1).unsqueeze(-1)
        action = (weights * contribs).sum(dim=1)
        return action


def add_state_noise(obs):
    if args.state_noise_std <= 0 and args.state_noise_ratio <= 0:
        return obs

    key = "observation.state"
    state = obs[key]

    noise = torch.zeros_like(state)

    if args.state_noise_std > 0:
        noise = noise + args.state_noise_std * torch.randn_like(state)

    if args.state_noise_ratio > 0:
        scale = state.detach().std(dim=0, keepdim=True).clamp_min(1e-6)
        noise = noise + args.state_noise_ratio * scale * torch.randn_like(state)

    obs[key] = state + noise
    return obs


def run_one_batch(env, policy, preprocessor, postprocessor,
                  env_preprocessor, env_postprocessor, seeds):
    batch_size = len(seeds)
    obs_dict, _ = env.reset(seed=seeds)

    done = np.zeros(batch_size, dtype=bool)
    successes = np.zeros(batch_size, dtype=bool)
    max_steps = env.call("_max_episode_steps")[0]

    policy.reset()

    step = 0
    while not np.all(done) and step < max_steps:
        obs = preprocess_observation(obs_dict)
        obs = add_envs_task(env, obs)
        obs = env_preprocessor(obs)
        obs = preprocessor(obs)

        # baseline chunk prediction과 residual state input 둘 다 noisy state를 봄
        obs = add_state_noise(obs)

        with torch.inference_mode():
            action = policy.select_action(obs)

        action = postprocessor(action)
        action = env_postprocessor({"action": action})["action"]
        action_np = action.detach().cpu().numpy()

        obs_dict, reward, terminated, truncated, info = env.step(action_np)
        done = done | terminated | truncated

        if "final_info" in info:
            final_infos = info.get("final_info", [None] * batch_size)

            if isinstance(final_infos, dict):
                if "is_success" in final_infos:
                    successes |= np.array(final_infos["is_success"], dtype=bool)
            else:
                for i, fi in enumerate(final_infos):
                    if isinstance(fi, dict) and fi.get("is_success", False):
                        successes[i] = True

        step += 1
        if step % 50 == 0:
            print(f"step {step}/{max_steps}, done {done.sum()}/{batch_size}")

    return successes


def main():
    torch.manual_seed(args.noise_seed)
    np.random.seed(args.noise_seed)

    print(f"Loading ACT baseline from {BASELINE_DIR}")
    baseline = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    weight_net = None
    L = args.L

    if args.mode == "residual":
        if args.weight_net_path is None:
            raise ValueError("--weight_net_path is required when --mode residual")

        weight_net_path = Path(args.weight_net_path)
        print(f"Loading weight network from {weight_net_path}")
        ckpt = torch.load(weight_net_path, map_location=DEVICE)

        L = ckpt["L"]

        weight_net = ResidualWeightNetwork(
            L=L,
            state_dim=ckpt["state_dim"],
            action_dim=ckpt["action_dim"],
            te_coeff=ckpt["te_coeff"],
            d_model=ckpt["d_model"],
            n_heads=ckpt["n_heads"],
            n_layers=ckpt["n_layers"],
        ).to(DEVICE)

        weight_net.load_state_dict(ckpt["model"])
        weight_net.eval()

        print(
            f"Loaded residual net: L={L}, "
            f"lambda={ckpt.get('lambda_delta_reg')}, "
            f"base={ckpt.get('base_policy')}, "
            f"te_direction={ckpt.get('te_direction')}, "
            f"trained_noise_std={ckpt.get('noise_std')}, "
            f"trained_corrupt_prob={ckpt.get('corrupt_prob')}"
        )
    else:
        print(f"Running pure TE wrapper with L={L}")

    policy = EnsemblePolicy(
        baseline_policy=baseline,
        weight_net=weight_net,
        L=L,
        mode=args.mode,
        te_coeff=args.te_coeff,
    ).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    env_cfg = AlohaEnv(task="AlohaTransferCube-v0")
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg=env_cfg,
        policy_cfg=config,
    )

    print("\nEval config:")
    print(f"  mode={args.mode}")
    print(f"  eval_episodes={args.eval_episodes}")
    print(f"  eval_batch_size={args.eval_batch_size}")
    print(f"  L={L}")
    print(f"  te_coeff={args.te_coeff}")
    print(f"  state_noise_std={args.state_noise_std}")
    print(f"  state_noise_ratio={args.state_noise_ratio}")
    print(f"  noise_seed={args.noise_seed}")

    all_successes = []
    start = 0

    while start < args.eval_episodes:
        cur_batch = min(args.eval_batch_size, args.eval_episodes - start)
        seeds = list(range(start, start + cur_batch))

        print(f"\nRunning ACT rollout batch: seeds {seeds[0]}..{seeds[-1]}, batch={cur_batch}")

        env = make_env(env_cfg, n_envs=cur_batch, use_async_envs=False)
        while isinstance(env, dict):
            print("make_env returned dict keys:", env.keys())
            env = list(env.values())[0]

        successes = run_one_batch(
            env=env,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            seeds=seeds,
        )

        all_successes.extend(successes.tolist())
        env.close()
        start += cur_batch

    all_successes = np.array(all_successes, dtype=bool)
    pc_success = all_successes.mean()

    print("\nResults ACT noisy-state eval")
    print(f"mode = {args.mode}")
    print(f"state_noise_std = {args.state_noise_std}")
    print(f"state_noise_ratio = {args.state_noise_ratio}")
    print(f"pc_success = {pc_success:.3f} ({all_successes.sum()}/{len(all_successes)})")

    print("\nReference:")
    print("  ACT baseline n=50 commit:        62%")
    print("  LeRobot TE fresh-favored:        54%")
    print("  our wrapper pure TE no network:  58%")
    print("  residual clean old result:       check your ckpt")


if __name__ == "__main__":
    main()
'''

Path("train_weight_residual_act_clean_noise.py").write_text(train_code)
Path("eval_with_weight_net_act_clean_noisyobs.py").write_text(eval_code)

print("Restored:")
print("  train_weight_residual_act_clean_noise.py")
print("  eval_with_weight_net_act_clean_noisyobs.py")
