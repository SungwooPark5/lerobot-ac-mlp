from pathlib import Path

# ============================================================
# make_acm_vis_files.py
# ACM image-feature residual weight network files 생성
# ============================================================

ROOT = Path(".").resolve()

ACM_BASELINE_DIR = "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/acm_transfer_cube_test_100k/checkpoints/last/pretrained_model"
CACHE_DIR = "/home1/eunji24/lerobot_project/cache/acm_baseline_chunks_vis"
CACHE_PATH = "/home1/eunji24/lerobot_project/cache/acm_baseline_chunks_vis/chunks.npz"
OUT_DIR = "/home1/eunji24/lerobot_project/outputs/weight_network_acm_residual_vis_lam5"


# ============================================================
# 1) Stage 1: ACM precompute chunks + visual features
# ============================================================

precompute_code = f'''"""Stage 1: baseline ACM으로 dataset 모든 frame chunk + image feature 미리 계산."""
import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

BASELINE_DIR = Path("{ACM_BASELINE_DIR}")
DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
CACHE_DIR = Path("{CACHE_DIR}")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_visual_feature(policy, batch):
    """
    ACM/ACT backbone output 형식이 다를 수 있어서 안전하게 처리.
    최종적으로 [B, C] image feature 반환.
    """
    image = batch["observation.images.top"]
    out = policy.model.backbone(image)

    if isinstance(out, torch.Tensor):
        feat_map = out
    elif isinstance(out, dict):
        if "feature_map" in out:
            feat_map = out["feature_map"]
        else:
            # dict인데 feature_map key가 없으면 마지막 value 사용
            feat_map = list(out.values())[-1]
    elif isinstance(out, (list, tuple)):
        feat_map = out[-1]
    else:
        raise TypeError(f"Unsupported backbone output type: {{type(out)}}")

    if feat_map.ndim == 4:
        vis_feat = feat_map.mean(dim=(-2, -1))  # [B, C, H, W] -> [B, C]
    elif feat_map.ndim == 3:
        vis_feat = feat_map.mean(dim=1)         # [B, T, C] -> [B, C]
    elif feat_map.ndim == 2:
        vis_feat = feat_map                     # [B, C]
    else:
        raise ValueError(f"Unexpected feat_map shape: {{feat_map.shape}}")

    return vis_feat


print(f"Loading ACM baseline from {{BASELINE_DIR}}")
config = PreTrainedConfig.from_pretrained(str(BASELINE_DIR))
print(f"  use_vae: {{config.use_vae}}, chunk_size: {{config.chunk_size}}")

policy = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
policy.eval()

preprocessor, _ = make_pre_post_processors(
    policy_cfg=config,
    pretrained_path=str(BASELINE_DIR),
    preprocessor_overrides={{
        "device_processor": {{"device": str(DEVICE)}},
    }},
)

print(f"Loading dataset {{DATASET_REPO_ID}}")
dataset = LeRobotDataset(DATASET_REPO_ID)
N = len(dataset)
print(f"  Total frames: {{N}}, Episodes: {{dataset.num_episodes}}")

sample0 = dataset[0]
state_dim = sample0["observation.state"].shape[0]
action_dim = sample0["action"].shape[0]
print(f"  state_dim={{state_dim}}, action_dim={{action_dim}}")

chunks_arr = np.zeros((N, config.chunk_size, action_dim), dtype=np.float32)
states_arr = np.zeros((N, state_dim), dtype=np.float32)
actions_arr = np.zeros((N, action_dim), dtype=np.float32)
ep_idx_arr = np.zeros(N, dtype=np.int64)

# vis_dim은 ACM backbone에 따라 다를 수 있어서 list로 받은 뒤 stack
vis_feats_list = []

ep_starts = []
frames_per_ep = N // dataset.num_episodes

for ep in range(dataset.num_episodes):
    f = ep * frames_per_ep
    t = (ep + 1) * frames_per_ep
    ep_starts.append((f, t))

print(f"  frames_per_episode={{frames_per_ep}}")

for ep, (f, t) in enumerate(ep_starts):
    ep_idx_arr[f:t] = ep

print("Computing ACM chunks + visual features...")
with torch.no_grad():
    for idx in tqdm(range(N)):
        sample = dataset[idx]

        batch = {{}}
        for k, v in sample.items():
            if torch.is_tensor(v):
                batch[k] = v.unsqueeze(0)
            else:
                batch[k] = v

        batch = preprocessor(batch)

        vis_feat = extract_visual_feature(policy, batch)
        chunk = policy.predict_action_chunk(batch)

        chunks_arr[idx] = chunk[0].detach().cpu().numpy()
        states_arr[idx] = sample["observation.state"].numpy()
        actions_arr[idx] = sample["action"].numpy()
        vis_feats_list.append(vis_feat[0].detach().cpu().numpy())

vis_feats_arr = np.stack(vis_feats_list).astype(np.float32)

print("Saving ACM cache...")
np.savez_compressed(
    CACHE_DIR / "chunks.npz",
    chunks=chunks_arr,
    states=states_arr,
    actions=actions_arr,
    vis_feats=vis_feats_arr,
    image_key=np.array(["observation.images.top"]),
    episode_idx=ep_idx_arr,
    ep_starts=np.array(ep_starts),
)

print(f"Saved to {{CACHE_DIR / 'chunks.npz'}}")
print(f"  chunks: {{chunks_arr.shape}}")
print(f"  states: {{states_arr.shape}}")
print(f"  actions: {{actions_arr.shape}}")
print(f"  vis_feats: {{vis_feats_arr.shape}}")
print(f"  size: ~{{chunks_arr.nbytes / 1e6:.1f}} MB chunks + {{states_arr.nbytes / 1e6:.1f}} MB states")
'''


# ============================================================
# 2) Stage 2: ACM residual weight network with image feature
# ============================================================

train_code = f'''"""Stage 2: ACM Residual weight network on TE prior + image feature."""
import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb

CACHE_PATH = Path("{CACHE_PATH}")
OUT_DIR = Path("{OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

L = 100
TE_COEFF = 0.01
LAMBDA_DELTA_REG = 5.0

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

        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

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


print("Loading ACM image cache...")
cache = np.load(CACHE_PATH)

chunks_cpu = torch.from_numpy(cache["chunks"].copy()).float()
states_cpu = torch.from_numpy(cache["states"].copy()).float()
actions_cpu = torch.from_numpy(cache["actions"].copy()).float()
vis_feats_cpu = torch.from_numpy(cache["vis_feats"].copy()).float()
vis_dim = vis_feats_cpu.shape[-1]
ep_starts = cache["ep_starts"].astype(np.int64)

N, chunk_size, action_dim = chunks_cpu.shape
state_dim = states_cpu.shape[1]

print(f"  N={{N}}, chunk_size={{chunk_size}}, state_dim={{state_dim}}, action_dim={{action_dim}}")
print(f"  vis_feats={{vis_feats_cpu.shape}}, vis_dim={{vis_dim}}")
print(f"  cache size: chunks={{chunks_cpu.element_size() * chunks_cpu.nelement() / 1e6:.1f}}MB")

ep_lens = ep_starts[:, 1] - ep_starts[:, 0]
print(
    f"  episode length: min={{ep_lens.min()}}, "
    f"max={{ep_lens.max()}}, mean={{ep_lens.mean():.0f}}"
)
assert ep_lens.min() >= L, f"Some episodes too short for L={{L}}!"


def sample_batch(batch_size):
    valid_eps = np.where(ep_lens >= L)[0]
    eps = np.random.choice(valid_eps, size=batch_size, replace=True)

    win_starts = np.zeros(batch_size, dtype=np.int64)
    for i, ep in enumerate(eps):
        ep_from = ep_starts[ep, 0]
        ep_len = ep_lens[ep]
        offset = np.random.randint(0, ep_len - L + 1)
        win_starts[i] = ep_from + offset

    # 최신 chunk가 contribs[:, 0], 그 다음이 contribs[:, 1] ...
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


model = ResidualWeightNetwork(
    L=L,
    state_dim=state_dim,
    action_dim=action_dim,
    vis_dim=vis_dim,
    te_coeff=TE_COEFF,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
print(f"Model params: {{sum(p.numel() for p in model.parameters()):,}}")

wandb.init(
    project="lerobot",
    name=f"acm_weight_residual_vis_lam5_L{{L}}_lambda{{LAMBDA_DELTA_REG}}",
    config={{
        "base": "acm",
        "L": L,
        "lr": LR,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "vis_dim": vis_dim,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "te_coeff": TE_COEFF,
        "lambda_delta_reg": LAMBDA_DELTA_REG,
    }},
)

with torch.no_grad():
    s, v, c, t = sample_batch(256)
    final_logits, delta = model(s, v, c)
    weights = F.softmax(final_logits, dim=1)

    init_l1 = F.l1_loss((weights.unsqueeze(-1) * c).sum(dim=1), t).item()
    init_delta_mse = (delta ** 2).mean().item()
    init_delta_abs = delta.abs().mean().item()
    init_weight_max = weights.max(dim=1).values.mean().item()
    init_entropy = -(weights * (weights + 1e-9).log()).sum(dim=1).mean().item()

    print("\\nInitial state: TE prior + delta=0")
    print(f"  l1: {{init_l1:.4f}}")
    print(f"  delta_mse: {{init_delta_mse:.8f}}")
    print(f"  delta_abs_mean: {{init_delta_abs:.8f}}")
    print(f"  weight_max: {{init_weight_max:.4f}}")
    print(f"  entropy_ratio: {{init_entropy / np.log(L):.4f}}")

print(f"\\nTraining {{STEPS}} steps with lambda_delta_reg={{LAMBDA_DELTA_REG}}...")
for step in tqdm(range(STEPS)):
    s, v, c, t = sample_batch(BATCH_SIZE)

    final_logits, delta = model(s, v, c)
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

            wandb.log({{
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
            }})

    if (step + 1) % SAVE_FREQ == 0:
        torch.save({{
            "model": model.state_dict(),
            "step": step + 1,
            "L": L,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "vis_dim": vis_dim,
            "image_key": "observation.images.top",
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "n_layers": N_LAYERS,
            "te_coeff": TE_COEFF,
            "lambda_delta_reg": LAMBDA_DELTA_REG,
            "base": "acm",
        }}, OUT_DIR / f"weight_net_step{{step + 1}}.pt")

torch.save({{
    "model": model.state_dict(),
    "step": STEPS,
    "L": L,
    "state_dim": state_dim,
    "action_dim": action_dim,
    "vis_dim": vis_dim,
    "image_key": "observation.images.top",
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
    "te_coeff": TE_COEFF,
    "lambda_delta_reg": LAMBDA_DELTA_REG,
    "base": "acm",
}}, OUT_DIR / "weight_net_final.pt")

print(f"\\nDone. Saved to {{OUT_DIR / 'weight_net_final.pt'}}")
'''


# ============================================================
# 3) Stage 3: ACM eval with image feature weight net
# ============================================================

eval_code = f'''"""Stage 3: ACM residual weight network + image feature eval."""
import sys
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

BASELINE_DIR = Path("{ACM_BASELINE_DIR}")
WEIGHT_NET_PATH = Path("{OUT_DIR}/weight_net_final.pt")

N_EPISODES = 50
BATCH_SIZE = 50
TE_COEFF = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.latest_backbone_output = None

        def _save_backbone_output(module, inputs, output):
            self.latest_backbone_output = output

        self.hook_handle = self.backbone.register_forward_hook(_save_backbone_output)

    def reset(self):
        self.past_chunks.clear()

    def eval(self):
        self.baseline.eval()
        self.weight_net.eval()
        self.backbone.eval()
        return self

    def _extract_hook_visual_feature(self):
        out = self.latest_backbone_output
        if out is None:
            raise RuntimeError("Backbone hook did not capture any feature map.")

        if isinstance(out, torch.Tensor):
            feat_map = out
        elif isinstance(out, dict):
            if "feature_map" in out:
                feat_map = out["feature_map"]
            else:
                feat_map = list(out.values())[-1]
        elif isinstance(out, (list, tuple)):
            feat_map = out[-1]
        else:
            raise TypeError(f"Unsupported backbone output type: {{type(out)}}")

        if feat_map.ndim == 4:
            vis_feat = feat_map.mean(dim=(-2, -1))
        elif feat_map.ndim == 3:
            vis_feat = feat_map.mean(dim=1)
        elif feat_map.ndim == 2:
            vis_feat = feat_map
        else:
            raise ValueError(f"Unexpected feat_map shape: {{feat_map.shape}}")

        return vis_feat

    @torch.no_grad()
    def select_action(self, batch):
        # predict_action_chunk 내부에서 backbone forward가 돌고,
        # forward hook이 latest_backbone_output을 저장함.
        chunk = self.baseline.predict_action_chunk(batch)

        self.past_chunks.append(chunk)
        n = len(self.past_chunks)

        # 최신 chunk의 현재 action: past_chunks[-1][:, 0]
        # 한 step 전 chunk의 현재 action: past_chunks[-2][:, 1]
        # ...
        contribs = torch.stack([
            self.past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        if n < self.L:
            # 학습 전 warmup 구간은 기존 TE prior만 사용
            decay_w = torch.exp(
                -self.te_coeff * torch.arange(n, device=contribs.device)
            )
            decay_w = decay_w / decay_w.sum()
            action = (decay_w.view(1, n, 1) * contribs).sum(dim=1)
        else:
            state = batch["observation.state"]
            vis_feat = self._extract_hook_visual_feature()

            final_logits, _ = self.weight_net(state, vis_feat, contribs)
            weights = F.softmax(final_logits, dim=1).unsqueeze(-1)
            action = (weights * contribs).sum(dim=1)

        return action


def main():
    print(f"Loading ACM baseline from {{BASELINE_DIR}}")
    baseline = ACTPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE)
    baseline.eval()
    config = baseline.config

    print(f"Loading ACM image weight network from {{WEIGHT_NET_PATH}}")
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
        n_layers=ckpt["n_layers"],
    ).to(DEVICE)

    weight_net.load_state_dict(ckpt["model"])
    weight_net.eval()

    print(
        f"  L={{L}}, trained {{ckpt['step']}} steps, "
        f"te_coeff={{ckpt['te_coeff']}}, vis_dim={{vis_dim}}, image_key={{image_key}}"
    )

    policy = EnsemblePolicy(
        baseline,
        weight_net,
        L=L,
        te_coeff=TE_COEFF,
        backbone=baseline.model.backbone,
        image_key=image_key,
    ).eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(BASELINE_DIR),
        preprocessor_overrides={{
            "device_processor": {{"device": str(DEVICE)}},
        }},
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

    print(f"\\nRunning ACM image rollout: {{N_EPISODES}} episodes, batch={{BATCH_SIZE}}")
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
        action = env_postprocessor({{"action": action}})["action"]
        action_np = action.detach().to("cpu").numpy()

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
            print(f"  step {{step}}/{{max_steps}}, done {{done.sum()}}/{{BATCH_SIZE}}")

    pc_success = successes.mean()

    print("\\nResults:")
    print(f"  pc_success = {{pc_success:.3f}} ({{successes.sum()}}/{{BATCH_SIZE}})")
    print("  Compare:")
    print("    ACM baseline n=50:      56%")
    print("    ACM pure TE:            52%")
    print("    ACM lerobot TE -0.01:   26%")
    print(f"    ACM residual vis L={{L}}: {{pc_success * 100:.1f}}%")

    env.close()


if __name__ == "__main__":
    main()
'''


files = {
    "precompute_chunks_acm_vis.py": precompute_code,
    "train_weight_residual_acm_vis.py": train_code,
    "eval_with_weight_net_acm_vis.py": eval_code,
}

for name, code in files.items():
    path = ROOT / name
    path.write_text(code, encoding="utf-8")
    print(f"[OK] wrote {path}")

print()
print("생성 완료.")
print("실행 순서:")
print("  1) python precompute_chunks_acm_vis.py")
print("  2) python train_weight_residual_acm_vis.py")
print("  3) python eval_with_weight_net_acm_vis.py")
