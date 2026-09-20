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

주의: 노트북 커널 python 이 아니라 mamba_ssm 이 깔린 venv 로 돌려야 한다
(common_v23.py:43-44 의 PYTHON, 기본값 ~/lerobot_project/lerobot_env/bin/python).
diag0_backward_const.ipynb 이 알아서 그 venv 로 subprocess 호출한다.

셸에서:
    PY=~/lerobot_project/lerobot_env/bin/python
    PYTHONPATH=src $PY notebooks/libero/diag0_backward_const.py
    PYTHONPATH=src $PY notebooks/libero/diag0_backward_const.py --tags all --json /tmp/diag0.json
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

# 논문의 BiMamba 는 bimamba_pure 다. bimamba 태그는 carry 붙은 bimos 이므로 쓰지 않는다.
DEFAULT_TAG = "bimamba_pure"
K_TAGS = {50: "bimamba_pure_k50", 100: "bimamba_pure", 150: "bimamba_pure_k150"}

_DS_CACHE: dict = {}


def resolve_ckpt(tag: str, seed: int = 0, step: int | None = 150_000,
                 task: str = "libero_10") -> Path:
    """exp5/common_final 규약대로 pretrained_model 디렉토리를 찾는다."""
    base = Path(os.environ.get("LEROBOT_OUTPUT", Path.home() / "lerobot_project" / "outputs"))
    run_dir = base / "final" / "train" / task / tag / f"seed{seed}" / "checkpoints"
    if not run_dir.is_dir():
        raise FileNotFoundError(f"checkpoints 디렉토리가 없다: {run_dir}")

    if step is not None:
        cand = [run_dir / str(step)]
    else:
        cand = []
        if (run_dir / "last").exists():
            cand.append(run_dir / "last")
        numeric = sorted((d for d in run_dir.iterdir() if d.name.isdigit()),
                         key=lambda d: int(d.name))
        cand.extend(reversed(numeric))

    for c in cand:
        pm = c / "pretrained_model"
        if pm.is_dir():
            return pm
    raise FileNotFoundError(f"pretrained_model 을 찾지 못했다. 뒤져본 곳: {[str(c) for c in cand]}")


def _make_batch(ds, indices, preprocessor):
    loader = DataLoader(Subset(ds, indices), batch_size=len(indices), shuffle=False, num_workers=0)
    return preprocessor(next(iter(loader)))


def run(tag: str = DEFAULT_TAG, seed: int = 0, step: int | None = 150_000,
        task: str = "libero_10", batch: int = 4, device: str | None = None,
        verbose: bool = True) -> dict:
    """상수성 검사를 한 번 돌리고 결과 dict 를 돌려준다.

    반환 키: ok, verdict, tag, K, ckpt, bwd_ab, bwd_spread, fwd_ab, fwd_spread,
             act_ab, act_spread, ratio (K 길이 텐서, ||bwd||/||fwd||)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {"tag": tag, "ok": False, "verdict": "unknown"}

    ckpt = resolve_ckpt(tag, seed, step, task)
    out["ckpt"] = str(ckpt)
    if verbose:
        print(f"[ckpt] {ckpt}")

    cfg = PreTrainedConfig.from_pretrained(ckpt)
    K = cfg.chunk_size
    out["K"] = K
    if verbose:
        print(f"[cfg ] type={cfg.type}  chunk_size={K}  n_action_steps={cfg.n_action_steps}  "
              f"bimamba={getattr(cfg, 'use_bimamba_decoder', None)}")

    if not getattr(cfg, "use_bimamba_decoder", False):
        out["verdict"] = "not_bimamba"
        if verbose:
            print("\n!! 이 체크포인트는 BiMamba 가 아니다. tag 를 확인할 것.")
        return out

    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )

    # 데이터셋은 K(=delta_timestamps)마다 다르므로 K 로 캐시한다. 재로딩이 느려서.
    if K not in _DS_CACHE:
        meta = LeRobotDatasetMetadata(REPO_ID)
        _DS_CACHE[K] = LeRobotDataset(REPO_ID, delta_timestamps=resolve_delta_timestamps(cfg, meta))
    ds = _DS_CACHE[K]
    if verbose:
        print(f"[data] {REPO_ID}  frames={len(ds)}")

    # 겹치지 않고 최대한 멀리 떨어진 두 인덱스 묶음 → 관측이 확실히 다르게.
    stride = max(1, len(ds) // (2 * batch + 2))
    idx_a = [stride * (i + 1) for i in range(batch)]
    idx_b = [stride * (batch + i + 2) for i in range(batch)]
    out["idx_a"], out["idx_b"] = idx_a, idx_b
    if verbose:
        print(f"[idx ] A={idx_a}\n       B={idx_b}")

    batch_a = _make_batch(ds, idx_a, preprocessor)
    batch_b = _make_batch(ds, idx_b, preprocessor)

    # ── hook ────────────────────────────────────────────────────────────────────
    # mamba2_stateful_forward 는 layer(...) 를 호출하지 않고 layer.in_proj / layer.out_proj 를
    # 직접 부른다. 따라서 Mamba2 모듈이 아니라 out_proj(nn.Linear)에 걸어야 한다.
    dec = policy.model.decoder
    if getattr(dec, "backward_layers", None) is None:
        out["verdict"] = "no_backward_stack"
        if verbose:
            print("\n!! decoder.backward_layers 가 없다. 예상과 다른 디코더다.")
        return out

    cap: dict[str, list[torch.Tensor]] = {"fwd": [], "bwd": []}
    handles = [
        dec.forward_layers[-1].out_proj.register_forward_hook(
            lambda m, i, o, k="fwd": cap[k].append(o.detach().float().cpu())),
        dec.backward_layers[-1].out_proj.register_forward_hook(
            lambda m, i, o, k="bwd": cap[k].append(o.detach().float().cpu())),
    ]
    try:
        with torch.no_grad():
            policy.reset()
            act_a = policy.predict_action_chunk(batch_a).detach().float().cpu()
            policy.reset()
            act_b = policy.predict_action_chunk(batch_b).detach().float().cpu()
    finally:
        for h in handles:
            h.remove()

    if len(cap["fwd"]) != 2 or len(cap["bwd"]) != 2:
        out["verdict"] = "hook_mismatch"
        if verbose:
            print(f"\n!! hook 이 예상과 다르게 걸렸다. fwd={len(cap['fwd'])} bwd={len(cap['bwd'])} "
                  f"(각각 2 여야 한다). n_decoder_layers 나 호출 경로를 확인할 것.")
        return out

    # forward 스택 입력은 [C ; Q] → 쿼리는 뒤쪽 K개
    fwd_a, fwd_b = cap["fwd"][0][:, -K:, :], cap["fwd"][1][:, -K:, :]
    # backward 스택 입력은 flip([C ; Q]) = [q_K..q_1, c_M..c_1] → 쿼리는 앞쪽 K개
    bwd_a, bwd_b = cap["bwd"][0][:, :K, :], cap["bwd"][1][:, :K, :]

    def dmax(x, y):
        return (x - y).abs().max().item()

    def spread(x):
        """배치 안에서 샘플들끼리 얼마나 다른가 (관측이 다 다르므로 0 이면 상수)."""
        return (x - x[:1]).abs().max().item()

    out.update(
        bwd_ab=dmax(bwd_a, bwd_b), bwd_spread=spread(bwd_a),
        fwd_ab=dmax(fwd_a, fwd_b), fwd_spread=spread(fwd_a),
        act_ab=dmax(act_a, act_b), act_spread=spread(act_a),
    )

    # 역방향 기여가 실제로 의미 있는 크기인지 — 위치 정체성 가설의 사전 점검.
    fn = fwd_a.norm(dim=-1)                    # (B, K)
    bn = bwd_a.norm(dim=-1)
    out["ratio"] = (bn / fn.clamp_min(1e-9)).mean(0)   # (K,)

    # ── 판정 ────────────────────────────────────────────────────────────────────
    bwd_const = out["bwd_ab"] == 0.0 and out["bwd_spread"] == 0.0
    fwd_varies = out["fwd_ab"] > 0.0 and out["fwd_spread"] > 0.0
    out["verdict"] = "inconclusive" if not fwd_varies else ("confirmed" if bwd_const else "refuted")
    out["ok"] = out["verdict"] == "confirmed"

    if verbose:
        report(out)
    return out


def report(res: dict) -> None:
    """run() 결과를 표로 찍는다."""
    K = res["K"]
    print("\n" + "=" * 68)
    print(f"{res['tag']}  (K={K})")
    print(f"{'':28s} {'배치 A vs B':>18s} {'배치 내 샘플 간':>18s}")
    print("-" * 68)
    print(f"{'BWD @ query  (상수여야 함)':28s} {res['bwd_ab']:18.3e} {res['bwd_spread']:18.3e}")
    print(f"{'FWD @ query  (달라야 함)':28s} {res['fwd_ab']:18.3e} {res['fwd_spread']:18.3e}")
    print(f"{'출력 action  (달라야 함)':28s} {res['act_ab']:18.3e} {res['act_spread']:18.3e}")
    print("=" * 68)

    ratio = res["ratio"]
    picks = [0, K // 4, K // 2, 3 * K // 4, K - 1]
    print("\n||bwd|| / ||fwd||  (위치별, 1 에 가까울수록 역방향 기여가 큼)")
    print("  " + "  ".join(f"k={k+1:<4d}{ratio[k].item():6.3f}" for k in picks))
    print(f"  전체 평균 {ratio.mean().item():.3f}   최소 {ratio.min().item():.3f}   "
          f"최대 {ratio.max().item():.3f}")

    print()
    v = res["verdict"]
    if v == "inconclusive":
        print("판정 불가 — FWD 도 상수로 나왔다. 두 배치가 같은 관측일 가능성이 크다.")
        print("            batch 를 키우거나 idx_a / idx_b 를 바꿔서 다시 볼 것.")
    elif v == "confirmed":
        print("확정 — 역방향 브랜치의 쿼리 위치 출력은 관측과 무관한 상수다.")
        print("       BiMamba 의 역방향 스택은 정보를 나르지 않고, head 직전에")
        print("       위치별 상수벡터 b_k 를 주입하는 역할만 한다.")
        print("       → '앞쪽 쿼리가 뒤쪽을 본다' 는 설명은 이 구현에서 성립하지 않는다.")
    elif v == "refuted":
        print("반증 — 역방향 출력이 관측에 따라 변한다. 구조 분석이 틀렸다.")
        print("       스캔이 causal 이 아니거나 쿼리가 관측 의존일 수 있다. 경로를 다시 볼 것.")
    else:
        print(f"중단 — {v}")


def to_jsonable(res: dict) -> dict:
    """run() 결과를 json 으로 덤프 가능하게 바꾼다 (ratio 텐서 → list)."""
    d = dict(res)
    if "ratio" in d:
        d["ratio"] = [float(x) for x in d["ratio"]]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=DEFAULT_TAG,
                    help="쉼표로 여러 개, 또는 'all' (=K_TAGS 전체)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=150_000, help="-1 이면 최신 체크포인트")
    ap.add_argument("--task", default="libero_10")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None, help="결과를 이 경로에 json 으로 덤프")
    a = ap.parse_args()

    tags = list(K_TAGS.values()) if a.tags == "all" else [t.strip() for t in a.tags.split(",")]
    step = None if a.step < 0 else a.step

    out: dict = {}
    for tag in tags:
        if len(tags) > 1:
            print("\n" + "#" * 68 + f"\n# {tag}\n" + "#" * 68)
        try:
            out[tag] = run(tag=tag, seed=a.seed, step=step, task=a.task,
                           batch=a.batch, device=a.device, verbose=True)
        except FileNotFoundError as e:
            print(f"건너뜀 — {e}")
            out[tag] = {"tag": tag, "ok": False, "verdict": "no_checkpoint", "error": str(e)}

    if a.json:
        import json as _json
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps({k: to_jsonable(v) for k, v in out.items()},
                                 ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[json] {p}")

    return 0 if all(r["ok"] for r in out.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
