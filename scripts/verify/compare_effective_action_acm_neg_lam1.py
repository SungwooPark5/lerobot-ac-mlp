import sys
sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

import json
import shutil
import tempfile
from pathlib import Path
from collections import deque
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lerobot.policies.acm.modeling_acm import ACMPolicy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors


DATASET_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"

BASELINE_DIR = Path(
    "/home1/eunji24/lerobot_project/lerobot-ac-mlp/outputs/train/"
    "acm_transfer_cube_test_100k/checkpoints/060000/pretrained_model"
)

WEIGHT_NET_PATH = Path(
    "/home1/eunji24/lerobot_project/outputs/"
    "weight_network_acm_60k_residual_vis_neg_coeff_lam1p0/weight_net_final.pt"
)

OUT_CSV = Path(
    "/home1/eunji24/lerobot_project/outputs/"
    "effective_action_compare_acm60k_neg_lam1.csv"
)

L = 100
TE_COEFF = -0.01
START_IDX = 0
NUM_FRAMES = 160
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


def make_temp_policy_dir_with_coeff(src_dir, coeff, n_action_steps=1):
    src_dir = Path(src_dir)
    tmp_root = Path(tempfile.mkdtemp(prefix="tmp_lerobot_te_effective_"))
    dst_dir = tmp_root / "pretrained_model"
    shutil.copytree(src_dir, dst_dir)

    cfg_path = dst_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())

    cfg["temporal_ensemble_coeff"] = coeff
    if "n_action_steps" in cfg:
        cfg["n_action_steps"] = n_action_steps

    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[TEMP POLICY DIR] {dst_dir}")
    print(f"[TEMP CONFIG] temporal_ensemble_coeff={coeff}, n_action_steps={cfg.get('n_action_steps')}")
    return dst_dir


def reset_policy(policy):
    if hasattr(policy, "reset"):
        try:
            policy.reset()
            return
        except Exception as e:
            print(f"[RESET WARN] {e}")


def make_batch(sample, preprocessor):
    batch = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0)
        else:
            batch[k] = v
    return preprocessor(batch)


def extract_visual_feature_from_hook(out):
    if out is None:
        raise RuntimeError("Backbone hook did not capture output.")

    if isinstance(out, torch.Tensor):
        feat_map = out
    elif isinstance(out, dict):
        feat_map = out["feature_map"] if "feature_map" in out else list(out.values())[-1]
    elif isinstance(out, (list, tuple)):
        feat_map = out[-1]
    else:
        raise TypeError(f"Unsupported backbone output type: {type(out)}")

    if feat_map.ndim == 4:
        return feat_map.mean(dim=(-2, -1))
    if feat_map.ndim == 3:
        return feat_map.mean(dim=1)
    if feat_map.ndim == 2:
        return feat_map

    raise ValueError(f"Unexpected feat_map shape: {feat_map.shape}")


@torch.no_grad()
def main():
    print("=" * 100)
    print("Effective action comparison: LeRobot internal TE vs weight net")
    print(f"BASELINE_DIR    = {BASELINE_DIR}")
    print(f"WEIGHT_NET_PATH = {WEIGHT_NET_PATH}")
    print(f"TE_COEFF        = {TE_COEFF}")
    print(f"L               = {L}")
    print(f"frames          = {START_IDX} ~ {START_IDX + NUM_FRAMES - 1}")
    print(f"DEVICE          = {DEVICE}")
    print("=" * 100)

    # 1. LeRobot internal TE policy
    lr_dir = make_temp_policy_dir_with_coeff(
        BASELINE_DIR,
        coeff=TE_COEFF,
        n_action_steps=1,
    )

    policy_lr = ACMPolicy.from_pretrained(str(lr_dir)).to(DEVICE).eval()
    reset_policy(policy_lr)

    pre_lr, post_lr = make_pre_post_processors(
        policy_cfg=policy_lr.config,
        pretrained_path=str(lr_dir),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    # 2. Our chunk policy
    policy_ours = ACMPolicy.from_pretrained(str(BASELINE_DIR)).to(DEVICE).eval()
    reset_policy(policy_ours)

    pre_ours, post_ours = make_pre_post_processors(
        policy_cfg=policy_ours.config,
        pretrained_path=str(BASELINE_DIR),
        preprocessor_overrides={"device_processor": {"device": str(DEVICE)}},
    )

    latest_backbone_output = {"value": None}

    def hook_fn(module, inputs, output):
        latest_backbone_output["value"] = output

    hook = policy_ours.model.backbone.register_forward_hook(hook_fn)

    # 3. Weight net
    ckpt = torch.load(WEIGHT_NET_PATH, map_location=DEVICE)

    weight_net = ResidualWeightNetwork(
        L=ckpt["L"],
        state_dim=ckpt["state_dim"],
        action_dim=ckpt["action_dim"],
        vis_dim=ckpt["vis_dim"],
        te_coeff=ckpt["te_coeff"],
        d_model=ckpt["d_model"],
        n_heads=ckpt["n_heads"],
        n_layers=ckpt["n_layers"],
    ).to(DEVICE)

    weight_net.load_state_dict(ckpt["model"])
    weight_net.eval()

    print(f"[CKPT] step={ckpt.get('step')}, te_coeff={ckpt.get('te_coeff')}, lambda={ckpt.get('lambda_delta_reg')}")

    dataset = LeRobotDataset(DATASET_REPO_ID)
    past_chunks = deque(maxlen=L)

    rows = []

    diffs_lr_vs_pure_equiv = []
    diffs_lr_vs_wn_raw = []
    diffs_lr_vs_wn_flip = []
    diffs_pure_equiv_vs_wn_raw = []
    diffs_pure_equiv_vs_wn_flip = []

    print()
    print("t,n,diff_lr_pure_equiv,diff_lr_wn_raw,diff_lr_wn_flip,wn_argmax,wn_flip_argmax,wn_max")
    print("-" * 100)

    for local_t in range(NUM_FRAMES):
        idx = START_IDX + local_t
        sample = dataset[idx]

        # LeRobot internal TE action
        batch_lr = make_batch(sample, pre_lr)
        action_lr = policy_lr.select_action(batch_lr)
        action_lr = post_lr(action_lr)

        # Our chunk candidates
        batch_ours = make_batch(sample, pre_ours)

        latest_backbone_output["value"] = None
        chunk = policy_ours.predict_action_chunk(batch_ours)
        vis_feat = extract_visual_feature_from_hook(latest_backbone_output["value"])

        past_chunks.append(chunk)
        n = len(past_chunks)

        contribs = torch.stack([
            past_chunks[-(i + 1)][:, i] for i in range(n)
        ], dim=1)

        # Pure TE, LeRobot-equivalent on our contribs
        logits_pure = TE_COEFF * torch.arange(
            n, device=contribs.device, dtype=contribs.dtype
        )
        w_pure = torch.softmax(logits_pure, dim=0)
        w_pure_equiv = torch.flip(w_pure, dims=[0])

        action_pure_equiv = (
            w_pure_equiv.view(1, n, 1) * contribs
        ).sum(dim=1)
        action_pure_equiv = post_ours(action_pure_equiv)

        diff_lr_pure_equiv = (action_lr - action_pure_equiv).abs().max().item()
        diffs_lr_vs_pure_equiv.append(diff_lr_pure_equiv)

        # Weight net only available when n == L
        if n == L:
            state = batch_ours["observation.state"]

            final_logits, delta = weight_net(state, vis_feat, contribs)
            weights_raw = F.softmax(final_logits, dim=1)
            weights_flip = torch.flip(weights_raw, dims=[1])

            action_wn_raw = (weights_raw.unsqueeze(-1) * contribs).sum(dim=1)
            action_wn_flip = (weights_flip.unsqueeze(-1) * contribs).sum(dim=1)

            action_wn_raw = post_ours(action_wn_raw)
            action_wn_flip = post_ours(action_wn_flip)

            diff_lr_wn_raw = (action_lr - action_wn_raw).abs().max().item()
            diff_lr_wn_flip = (action_lr - action_wn_flip).abs().max().item()
            diff_pure_wn_raw = (action_pure_equiv - action_wn_raw).abs().max().item()
            diff_pure_wn_flip = (action_pure_equiv - action_wn_flip).abs().max().item()

            diffs_lr_vs_wn_raw.append(diff_lr_wn_raw)
            diffs_lr_vs_wn_flip.append(diff_lr_wn_flip)
            diffs_pure_equiv_vs_wn_raw.append(diff_pure_wn_raw)
            diffs_pure_equiv_vs_wn_flip.append(diff_pure_wn_flip)

            wn_argmax = int(weights_raw[0].argmax().item())
            wn_flip_argmax = int(weights_flip[0].argmax().item())
            wn_max = float(weights_raw[0].max().item())

            final_logit_max = float(final_logits[0].max().item())
            final_logit_min = float(final_logits[0].min().item())
            delta_abs_mean = float(delta.abs().mean().item())

        else:
            diff_lr_wn_raw = np.nan
            diff_lr_wn_flip = np.nan
            diff_pure_wn_raw = np.nan
            diff_pure_wn_flip = np.nan
            wn_argmax = -1
            wn_flip_argmax = -1
            wn_max = np.nan
            final_logit_max = np.nan
            final_logit_min = np.nan
            delta_abs_mean = np.nan

        if local_t < 10 or local_t in [98, 99, 100, 101, NUM_FRAMES - 1]:
            print(
                f"{local_t},{n},"
                f"{diff_lr_pure_equiv:.10f},"
                f"{diff_lr_wn_raw:.10f},"
                f"{diff_lr_wn_flip:.10f},"
                f"{wn_argmax},{wn_flip_argmax},{wn_max}"
            )

        row = {
            "local_t": local_t,
            "dataset_idx": idx,
            "n": n,
            "diff_lr_vs_pure_equiv": diff_lr_pure_equiv,
            "diff_lr_vs_weightnet_raw": diff_lr_wn_raw,
            "diff_lr_vs_weightnet_flip": diff_lr_wn_flip,
            "diff_pure_equiv_vs_weightnet_raw": diff_pure_wn_raw,
            "diff_pure_equiv_vs_weightnet_flip": diff_pure_wn_flip,
            "weightnet_argmax_raw": wn_argmax,
            "weightnet_argmax_flip": wn_flip_argmax,
            "weightnet_max_weight": wn_max,
            "final_logit_max": final_logit_max,
            "final_logit_min": final_logit_min,
            "delta_abs_mean": delta_abs_mean,
        }

        # action vector 저장
        for j in range(action_lr.shape[-1]):
            row[f"action_lr_{j}"] = float(action_lr[0, j].item())
            row[f"action_pure_equiv_{j}"] = float(action_pure_equiv[0, j].item())

            if n == L:
                row[f"action_wn_raw_{j}"] = float(action_wn_raw[0, j].item())
                row[f"action_wn_flip_{j}"] = float(action_wn_flip[0, j].item())
            else:
                row[f"action_wn_raw_{j}"] = np.nan
                row[f"action_wn_flip_{j}"] = np.nan

        rows.append(row)

    hook.remove()

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 100)
    print("[SUMMARY]")
    print(f"mean diff LeRobot vs pure_equiv      = {np.mean(diffs_lr_vs_pure_equiv):.10f}")
    print(f"max  diff LeRobot vs pure_equiv      = {np.max(diffs_lr_vs_pure_equiv):.10f}")

    if len(diffs_lr_vs_wn_raw) > 0:
        print(f"mean diff LeRobot vs weightnet_raw   = {np.mean(diffs_lr_vs_wn_raw):.10f}")
        print(f"max  diff LeRobot vs weightnet_raw   = {np.max(diffs_lr_vs_wn_raw):.10f}")
        print(f"mean diff LeRobot vs weightnet_flip  = {np.mean(diffs_lr_vs_wn_flip):.10f}")
        print(f"max  diff LeRobot vs weightnet_flip  = {np.max(diffs_lr_vs_wn_flip):.10f}")

        print(f"mean diff pure_equiv vs weightnet_raw  = {np.mean(diffs_pure_equiv_vs_wn_raw):.10f}")
        print(f"mean diff pure_equiv vs weightnet_flip = {np.mean(diffs_pure_equiv_vs_wn_flip):.10f}")

    print(f"[SAVED] {OUT_CSV}")


if __name__ == "__main__":
    main()
