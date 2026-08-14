"""
TE Mode-Collapse Analysis
========================
질문: temporal ensemble의 가중평균이 각 후보 중 GT에 가장 가까웠던 것보다
더 나쁜 예측을 내놓는가?

모델: act_network_baseline_100k (chunk_size=100, TE 미사용으로 훈련됨)
TE를 사후 적용하면서 후보 전부 보존 → GT와 비교

핵심 메트릭:
  te_l1   = L1(TE_output, GT)
  best_l1 = min_i L1(candidate_i, GT)
  delta   = te_l1 - best_l1  > 0 이면 TE가 오히려 손해
"""

import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, "/home1/eunji24/lerobot_project/lerobot-ac-mlp/src")

# groot_n1.py가 Python 3.12 @dataclass 버그로 임포트 실패함 → mock으로 우회
# factory.py에서 isinstance(cfg, GrootConfig) 체크가 있으므로
# GrootConfig는 실제 클래스로 심어줘야 함
class _FakeGrootConfig:
    pass

_groot_cfg_mock = MagicMock()
_groot_cfg_mock.GrootConfig = _FakeGrootConfig

sys.modules["lerobot.policies.groot.groot_n1"] = MagicMock()
sys.modules["lerobot.policies.groot.modeling_groot"] = MagicMock()
sys.modules["lerobot.policies.groot.configuration_groot"] = _groot_cfg_mock
sys.modules["lerobot.policies.groot"] = MagicMock()

from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ── 설정 ─────────────────────────────────────────────────────────────────────
CKPT = "/home1/eunji24/lerobot_project/outputs/train/act_network_baseline_100k/checkpoints/last/pretrained_model"
DATASET_REPO = "lerobot/aloha_sim_transfer_cube_human"
DATASET_ROOT = Path("/home1/eunji24/.cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human")
N_EPISODES   = 20           # 분석할 에피소드 수
TE_COEFFS    = [0.0, 0.01, 0.1, 1.0]  # 여러 coeff 비교
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────────────────────

def load_policy_and_preprocessor():
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(CKPT)
    policy.eval().to(DEVICE)

    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_path=CKPT,
    )
    return policy, preprocessor


def load_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # delta_indices 없이 로드 → 단순히 프레임별 obs+action 접근
    ds = LeRobotDataset(
        repo_id=DATASET_REPO,
        root=DATASET_ROOT,
    )
    return ds


def get_episode_indices(ds, ep_idx):
    """해당 episode의 dataset 인덱스 목록 반환"""
    ep_col = ds.hf_dataset["episode_index"]
    indices = [i for i, e in enumerate(ep_col) if e == ep_idx]
    return indices


def frame_to_obs(ds, frame_idx):
    """
    단일 프레임에서 obs dict를 반환 (정규화 전, CPU tensor).
    LeRobotDataset.__getitem__을 delta_indices 없이 사용.
    """
    item = ds.hf_dataset[frame_idx]

    obs = {}
    # state
    obs["observation.state"] = torch.tensor(
        item["observation.state"], dtype=torch.float32
    )
    # image (video key → 이미 tensor로 decode됨)
    # hf_dataset에는 PIL or np.array 형태일 수 있어 직접 접근
    obs["observation.images.top"] = item["observation.images.top"]
    if not isinstance(obs["observation.images.top"], torch.Tensor):
        obs["observation.images.top"] = torch.tensor(
            np.array(obs["observation.images.top"]), dtype=torch.float32
        )
    # (H,W,C) → (C,H,W)
    if obs["observation.images.top"].ndim == 3 and obs["observation.images.top"].shape[-1] in (1, 3, 4):
        obs["observation.images.top"] = obs["observation.images.top"].permute(2, 0, 1)

    # GT action (raw, 나중에 normalized space에서 비교)
    gt_action = torch.tensor(item["action"], dtype=torch.float32)
    return obs, gt_action


def te_weights(n: int, coeff: float) -> torch.Tensor:
    """
    TE weight: w[i] for candidate i, where i=0=newest, i=n-1=oldest.
    ACT TE는 oldest에 가장 높은 weight (index 0).
    => w[i] = exp(-coeff * (n-1-i))
    """
    ages = torch.arange(n - 1, -1, -1, dtype=torch.float32)  # [n-1, n-2, ..., 0]
    # oldest candidate (i=n-1) has age=0 → highest weight
    # actually: age of candidate i = n-1-i from the "first added" perspective
    # TE weight index = n-1-i (oldest=0, newest=n-1)
    w = torch.exp(-coeff * (n - 1 - torch.arange(n, dtype=torch.float32)))
    return w / w.sum()


def analyze_episode(policy, preprocessor, ds, ep_idx, te_coeff):
    frame_indices = get_episode_indices(ds, ep_idx)
    if not frame_indices:
        return []

    chunk_size = policy.config.chunk_size
    chunk_buffer = deque(maxlen=chunk_size)  # stores (chunk_size, action_dim) tensors

    results = []

    for tau, fidx in enumerate(frame_indices):
        obs, gt_action_raw = frame_to_obs(ds, fidx)

        # 전처리 → normalized batch on device
        batch = preprocessor(obs)
        # preprocessor output: dict with (1, ...) shaped tensors

        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
        chunk_buffer.append(chunk[0].cpu())  # (chunk_size, action_dim)

        n = len(chunk_buffer)

        # 현재 τ에 대한 후보들: chunk_buffer[-(i+1)][i] for i=0..n-1
        # i=0: newest chunk의 [0]번째 액션 = 현재 타임스텝에 대한 최신 예측
        candidates = torch.stack(
            [chunk_buffer[-(i + 1)][i] for i in range(n)], dim=0
        )  # (n, action_dim)

        # GT action을 normalized space로 변환
        # preprocessor의 normalizer stats를 사용해 직접 normalize
        gt_norm = normalize_action(gt_action_raw, preprocessor)

        # TE 가중합
        w = te_weights(n, te_coeff).unsqueeze(-1)  # (n, 1)
        te_action = (w * candidates).sum(0)  # (action_dim,)

        te_l1   = F.l1_loss(te_action, gt_norm).item()
        best_l1 = min(F.l1_loss(candidates[i], gt_norm).item() for i in range(n))
        cand_std = candidates.std(0).mean().item()

        results.append({
            "tau": tau,
            "n_candidates": n,
            "te_l1": te_l1,
            "best_l1": best_l1,
            "delta": te_l1 - best_l1,
            "cand_std": cand_std,
            "te_worse": te_l1 > best_l1,
        })

    return results


def normalize_action(action_raw: torch.Tensor, preprocessor) -> torch.Tensor:
    """
    action_raw (action_dim,)을 normalizer stats로 정규화.
    preprocessor pipeline에서 normalizer step을 찾아 action stats 추출.
    """
    for step in preprocessor.steps:
        if hasattr(step, "stats") and "action" in getattr(step, "stats", {}):
            stats = step.stats["action"]
            mean = stats["mean"].cpu()
            std  = stats["std"].cpu()
            return (action_raw - mean) / (std + 1e-8)
    # fallback: normalize not found, return raw
    return action_raw


def print_summary(all_results, te_coeff):
    if not all_results:
        print("No results.")
        return

    deltas    = np.array([r["delta"] for r in all_results])
    te_worse  = np.array([r["te_worse"] for r in all_results])
    cand_std  = np.array([r["cand_std"] for r in all_results])
    n_cands   = np.array([r["n_candidates"] for r in all_results])
    te_l1s    = np.array([r["te_l1"] for r in all_results])
    best_l1s  = np.array([r["best_l1"] for r in all_results])

    print(f"\n{'='*60}")
    print(f"TE coeff = {te_coeff:.3f}  |  총 스텝 수: {len(all_results)}")
    print(f"{'='*60}")
    print(f"  mean(te_l1):               {te_l1s.mean():.4f}")
    print(f"  mean(best_l1):             {best_l1s.mean():.4f}")
    print(f"  mean(delta=te-best):       {deltas.mean():.4f}  {'← TE가 평균적으로 손해' if deltas.mean()>0 else '← TE가 평균적으로 이득'}")
    print(f"  % TE worse than best:      {te_worse.mean()*100:.1f}%")
    print(f"  mean candidate diversity:  {cand_std.mean():.4f}")
    print()
    # n_candidates별 breakdown
    print("  n_candidates | % TE worse | mean(delta) | mean(cand_std)")
    for n in sorted(set(n_cands)):
        mask = n_cands == n
        if mask.sum() == 0:
            continue
        print(f"    {n:3d}        |  {te_worse[mask].mean()*100:5.1f}%    |  {deltas[mask].mean():+.4f}     | {cand_std[mask].mean():.4f}")
    print()
    # diversity별 분석: high vs low diversity
    med_std = np.median(cand_std)
    hi_mask = cand_std > med_std
    lo_mask = ~hi_mask
    print(f"  High diversity (std>{med_std:.3f}): delta={deltas[hi_mask].mean():+.4f}, TE worse={te_worse[hi_mask].mean()*100:.1f}%")
    print(f"  Low  diversity (std<{med_std:.3f}): delta={deltas[lo_mask].mean():+.4f}, TE worse={te_worse[lo_mask].mean()*100:.1f}%")


def main():
    print("Loading policy and preprocessor...")
    policy, preprocessor = load_policy_and_preprocessor()
    print(f"  chunk_size={policy.config.chunk_size}, action_dim=14")

    print("Loading dataset...")
    ds = load_dataset()
    n_total_ep = ds.num_episodes
    n_ep = min(N_EPISODES, n_total_ep)
    print(f"  {n_total_ep} episodes available → analyzing first {n_ep}")

    for te_coeff in TE_COEFFS:
        all_results = []
        for ep_idx in range(n_ep):
            ep_results = analyze_episode(policy, preprocessor, ds, ep_idx, te_coeff)
            all_results.extend(ep_results)
            if (ep_idx + 1) % 5 == 0:
                print(f"  coeff={te_coeff}: {ep_idx+1}/{n_ep} episodes done...", flush=True)
        print_summary(all_results, te_coeff)

    print("\nDone.")


if __name__ == "__main__":
    main()
