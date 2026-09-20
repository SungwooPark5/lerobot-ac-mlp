"""diag0 — BiMamba 역방향 브랜치가 관측 정보를 나르는지 확인한다.

주장(코드 구조상):
    decoder_in = zeros,  decoder_pos_embed = nn.Embedding(K, D)  → 쿼리 Q 는 배치 무관 상수
    combined   = [C ; Q]                                          → C 만 관측 의존
    backward   = flip(scan(flip(combined)))                       → flip 후 Q 가 시퀀스 맨 앞
    scan 은 causal → 쿼리 위치 출력은 Q 만의 함수 = 상수

이 스크립트는 그 상수성을 실측한다.

    BWD @ query  : 관측이 달라도 동일해야 함     (max|Δ| == 0)
    FWD @ query  : 관측이 다르면 달라야 함       (max|Δ| >> 0, 민감도 대조군)

FWD 까지 0 이 나오면 두 배치가 사실 같다는 뜻이므로 테스트가 깨진 것이다.

서버에서:
    cd ~/lerobot_project/lerobot-ac-mlp
    python notebooks/libero/diag0_backward_const.py
    python notebooks/libero/diag0_backward_const.py --tag bimamba_pure_k150   # 다른 K
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

REPO_ID = "HuggingFaceVLA/libero"


def resolve_ckpt(tag: str, seed: int, step: int | None, task: str) -> Path:
    """exp5/common_final 규약대로 pretrained_model 디렉토리를 찾는다."""
    base = Path(os.environ.get("LEROBOT_OUTPUT", Path.home() / "lerobot_project" / "outputs"))
    run = base / "final" / "train" / task / tag / f"seed{seed}" / "checkpoints"
    if not run.is_dir():
        raise FileNotFoundError(f"checkpoints 디렉토리가 없다: {run}")

    if step is not None:
        cand = [run / str(step)]
    else:
        cand = []
        if (run / "last").exists():
            cand.append(run / "last")
        numeric = sorted((d for d in run.iterdir() if d.name.isdigit()), key=lambda d: int(d.name))
        cand.extend(reversed(numeric))

    for c in cand:
        pm = c / "pretrained_model"
        if pm.is_dir():
            return pm
    raise FileNotFoundError(f"pretrained_model 을 찾지 못했다. 뒤져본 곳: {[str(c) for c in cand]}")


def make_batch(ds, indices, preprocessor):
    loader = DataLoader(Subset(ds, indices), batch_size=len(indices), shuffle=False, num_workers=0)
    return preprocessor(next(iter(loader)))


def main() -> int:
    ap = argparse.ArgumentParser()
    # 주의: 논문의 BiMamba 는 bimamba_pure 다. bimamba 태그는 carry 붙은 bimos 다.
    ap.add_argument("--tag", default="bimamba_pure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=150_000, help="-1 이면 최신 체크포인트")
    ap.add_argument("--task", default="libero_10")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    step = None if args.step < 0 else args.step
    ckpt = resolve_ckpt(args.tag, args.seed, step, args.task)
    print(f"[ckpt] {ckpt}")

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    print(f"[cfg ] type={cfg.type}  chunk_size={cfg.chunk_size}  "
          f"n_action_steps={cfg.n_action_steps}  bimamba={getattr(cfg, 'use_bimamba_decoder', None)}")

    if not getattr(cfg, "use_bimamba_decoder", False):
        print("\n!! 이 체크포인트는 BiMamba 가 아니다. --tag 를 확인할 것.")
        return 2

    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(args.device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    meta = LeRobotDatasetMetadata(REPO_ID)
    ds = LeRobotDataset(REPO_ID, delta_timestamps=resolve_delta_timestamps(cfg, meta))
    print(f"[data] {REPO_ID}  frames={len(ds)}")

    # 겹치지 않고 최대한 멀리 떨어진 두 인덱스 묶음 → 관측이 확실히 다르게.
    B = args.batch
    stride = max(1, len(ds) // (2 * B + 2))
    idx_a = [stride * (i + 1) for i in range(B)]
    idx_b = [stride * (B + i + 2) for i in range(B)]
    print(f"[idx ] A={idx_a}\n       B={idx_b}")

    batch_a = make_batch(ds, idx_a, preprocessor)
    batch_b = make_batch(ds, idx_b, preprocessor)

    # ── hook ────────────────────────────────────────────────────────────────────
    # mamba2_stateful_forward 는 layer(...) 를 호출하지 않고 layer.in_proj / layer.out_proj 를
    # 직접 부른다. 따라서 Mamba2 모듈이 아니라 out_proj(nn.Linear)에 걸어야 한다.
    dec = policy.model.decoder
    if getattr(dec, "backward_layers", None) is None:
        print("\n!! decoder.backward_layers 가 없다. 예상과 다른 디코더다.")
        return 2

    cap: dict[str, list[torch.Tensor]] = {"fwd": [], "bwd": []}
    handles = [
        dec.forward_layers[-1].out_proj.register_forward_hook(
            lambda m, i, o, k="fwd": cap[k].append(o.detach().float().cpu())),
        dec.backward_layers[-1].out_proj.register_forward_hook(
            lambda m, i, o, k="bwd": cap[k].append(o.detach().float().cpu())),
    ]

    with torch.no_grad():
        policy.reset()
        act_a = policy.predict_action_chunk(batch_a).detach().float().cpu()
        policy.reset()
        act_b = policy.predict_action_chunk(batch_b).detach().float().cpu()

    for h in handles:
        h.remove()

    if len(cap["fwd"]) != 2 or len(cap["bwd"]) != 2:
        print(f"\n!! hook 이 예상과 다르게 걸렸다. fwd={len(cap['fwd'])} bwd={len(cap['bwd'])} "
              f"(각각 2 여야 한다). n_decoder_layers 나 호출 경로를 확인할 것.")
        return 2

    K = cfg.chunk_size
    # forward 스택 입력은 [C ; Q] → 쿼리는 뒤쪽 K개
    fwd_a, fwd_b = cap["fwd"][0][:, -K:, :], cap["fwd"][1][:, -K:, :]
    # backward 스택 입력은 flip([C ; Q]) = [q_K..q_1, c_M..c_1] → 쿼리는 앞쪽 K개
    bwd_a, bwd_b = cap["bwd"][0][:, :K, :], cap["bwd"][1][:, :K, :]

    def dmax(x, y):
        return (x - y).abs().max().item()

    def spread(x):
        """배치 안에서 샘플들끼리 얼마나 다른가 (관측이 다 다르므로 0 이면 상수)."""
        return (x - x[:1]).abs().max().item()

    print("\n" + "=" * 68)
    print(f"{'':28s} {'배치 A vs B':>18s} {'배치 내 샘플 간':>18s}")
    print("-" * 68)
    print(f"{'BWD @ query  (상수여야 함)':28s} {dmax(bwd_a, bwd_b):18.3e} {spread(bwd_a):18.3e}")
    print(f"{'FWD @ query  (달라야 함)':28s} {dmax(fwd_a, fwd_b):18.3e} {spread(fwd_a):18.3e}")
    print(f"{'출력 action  (달라야 함)':28s} {dmax(act_a, act_b):18.3e} {spread(act_a):18.3e}")
    print("=" * 68)

    # 역방향 기여가 실제로 의미 있는 크기인지 — 위치 정체성 가설의 사전 점검.
    fn = fwd_a.norm(dim=-1)                      # (B, K)
    bn = bwd_a.norm(dim=-1)
    ratio = (bn / fn.clamp_min(1e-9)).mean(0)    # (K,)
    picks = [0, K // 4, K // 2, 3 * K // 4, K - 1]
    print("\n||bwd|| / ||fwd||  (위치별, 1 에 가까울수록 역방향 기여가 큼)")
    print("  " + "  ".join(f"k={k+1:<4d}{ratio[k].item():6.3f}" for k in picks))
    print(f"  전체 평균 {ratio.mean().item():.3f}   최소 {ratio.min().item():.3f}   "
          f"최대 {ratio.max().item():.3f}")

    # ── 판정 ────────────────────────────────────────────────────────────────────
    bwd_const = dmax(bwd_a, bwd_b) == 0.0 and spread(bwd_a) == 0.0
    fwd_varies = dmax(fwd_a, fwd_b) > 0.0 and spread(fwd_a) > 0.0

    print()
    if not fwd_varies:
        print("판정 불가 — FWD 도 상수로 나왔다. 두 배치가 같은 관측일 가능성이 크다.")
        print("            --batch 를 키우거나 idx_a / idx_b 를 바꿔서 다시 볼 것.")
        return 2
    if bwd_const:
        print("확정 — 역방향 브랜치의 쿼리 위치 출력은 관측과 무관한 상수다.")
        print("       BiMamba 의 역방향 스택은 정보를 나르지 않고, head 직전에")
        print("       위치별 상수벡터 b_k 를 주입하는 역할만 한다.")
        print("       → '앞쪽 쿼리가 뒤쪽을 본다' 는 설명은 이 구현에서 성립하지 않는다.")
    else:
        print("반증 — 역방향 출력이 관측에 따라 변한다. 구조 분석이 틀렸다.")
        print("       스캔이 causal 이 아니거나 쿼리가 관측 의존일 수 있다. 경로를 다시 볼 것.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
