"""diag3 — b_k 의 무엇이 일을 하는가. 학습 없이 표만 바꿔 가며 잰다.

diag0/diag1/diag2 가 확정한 것:
    역방향 브랜치 = 관측 무관 상수표 b_k (항등식)
    순방향은 위치가 뭉개져 있고(활용률 0.27), b_k 가 갈라 준다(0.59)
    b_k 를 끄면 위치별 오차가 1.4~2.3 배

남은 질문: **b_k 의 어떤 성질이 일을 하는가?**
    크기인가, 위치마다 다르다는 것인가, 아니면 학습된 특정 대응인가.

diag1 의 compile_decoder 가 임의의 표를 받으므로 학습 없이 잴 수 있다.
한 번 컴파일한 뒤 dec._bk 만 갈아 끼우면 되므로 변형 하나당 forward 한 번이다.

변형:
    real          b 그대로                              (기준)
    zero          0                                     (하한)
    mean          모든 위치에 b 의 평균벡터 하나        <- 핵심 대조
    shuffle       같은 벡터들, 위치만 섞음
    random_ortho  랜덤 직교 행, b 의 위치별 norm 으로 맞춤
    alpha=X       X * b                                 (크기 스윕)

**mean 이 제일 중요하다.** 크기와 공통 성분은 그대로 두고 *위치별 변화만* 없앤다.
    mean ~ zero  -> 이득은 전부 "위치마다 다르다" 에서 온다
    mean ~ real  -> 그냥 큰 상수 하나 더하는 것과 같다 (위치와 무관)

⚠️ 해석 주의. 모든 변형은 학습 때 없던 표를 추론에 꽂는 것이라 그 자체로 OOD 다.
   따라서 **변형 하나의 절대 성능은 의미가 약하고, 변형끼리의 비교만 읽어야 한다.**
   shuffle 과 random_ortho 는 둘 다 "모델이 본 적 없는 표" 라는 점에서 동등하게
   OOD 이므로, 그 둘의 차이는 OOD 효과가 상쇄된 뒤 남는 것이다.

주의: 노트북 커널이 아니라 mamba_ssm 이 깔린 venv 로 돌려야 한다
(common_v23.py:43-44). diag3_bk_variants.ipynb 이 알아서 그 venv 로 부른다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from diag0_backward_const import DEFAULT_TAG, K_TAGS, REPO_ID, _make_batch, resolve_ckpt
from diag1_bk_identity import compile_decoder, extract_bk
from diag2_position_collapse import separation

_DS_CACHE: dict = {}

DEFAULT_ALPHAS = (0.25, 0.5, 0.75, 1.5, 2.0, 4.0)


# ── 표 변형 ────────────────────────────────────────────────────────────────────

def make_variants(b: torch.Tensor, alphas=DEFAULT_ALPHAS, seed: int = 0) -> dict:
    """b (K, D) 에서 비교할 표들을 만든다. 순서가 리포트 순서다."""
    K, D = b.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    norms = b.norm(dim=-1, keepdim=True)                       # (K, 1)

    # 랜덤 직교 행을 b 의 위치별 norm 으로 맞춘다 — 분리도는 최대, 내용은 무의미.
    q = torch.linalg.qr(torch.randn(D, K, generator=g))[0].T   # (K, D), 직교 단위행
    rand_ortho = (q.to(b.device).to(b.dtype)) * norms

    perm = torch.randperm(K, generator=g).to(b.device)

    out = {
        "real": b,
        "zero": torch.zeros_like(b),
        # 크기·공통성분 유지, 위치별 변화만 제거. 이 노트북의 핵심 대조.
        "mean": b.mean(0, keepdim=True).expand_as(b).contiguous(),
        "shuffle": b[perm].contiguous(),
        "random_ortho": rand_ortho,
    }
    for a in alphas:
        out[f"alpha={a:g}"] = b * a
    return out


# ── 측정 ───────────────────────────────────────────────────────────────────────

def measure(policy, batch, gt, K: int) -> dict:
    """현재 dec._bk 상태로 예측하고 활용률·오차를 잰다."""
    dec = policy.model.decoder
    with torch.no_grad():
        policy.reset()
        pred = policy.predict_action_chunk(batch).detach().float().cpu()

    # fused 표현 = 컴파일 경로가 head 에 주는 것. hook 없이 직접 재구성한다.
    cap: list = []
    h = (dec.forward_layers[-1].out_proj
         .register_forward_hook(lambda m, i, o: cap.append(o.detach().float())))
    try:
        with torch.no_grad():
            policy.reset()
            policy.predict_action_chunk(batch)
    finally:
        h.remove()
    fwd = cap[0][:, -K:, :]
    wdt = dec.norm.weight.dtype
    with torch.no_grad():
        fused = dec.norm((0.5 * (fwd + dec._bk.to(fwd.dtype))).to(wdt)).float()

    err = (pred - gt).abs().mean(dim=(0, 2))                   # (K,)
    s = separation(fused)
    return {"use_ratio": s["use_ratio"], "eff_rank": s["eff_rank"],
            "cos_adj_mean": s["cos_adj_mean"],
            "err": [float(x) for x in err], "err_mean": float(err.mean())}


# ── 실행 ───────────────────────────────────────────────────────────────────────

def run(tag: str = DEFAULT_TAG, seed: int = 0, step: int | None = 150_000,
        task: str = "libero_10", batch: int = 4, device: str | None = None,
        alphas=DEFAULT_ALPHAS, verbose: bool = True) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {"tag": tag, "ok": False}

    ckpt = resolve_ckpt(tag, seed, step, task)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    K, D = cfg.chunk_size, cfg.dim_model
    out.update(ckpt=str(ckpt), K=K, dim_model=D)
    if verbose:
        print(f"[ckpt] {ckpt}\n[cfg ] {cfg.type}  K={K}  D={D}")

    if not getattr(cfg, "use_bimamba_decoder", False):
        out["verdict"] = "not_bimamba"
        print("\n!! BiMamba 가 아니다. b_k 가 없다.")
        return out

    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}})

    if K not in _DS_CACHE:
        _DS_CACHE[K] = LeRobotDataset(
            REPO_ID, delta_timestamps=resolve_delta_timestamps(cfg, LeRobotDatasetMetadata(REPO_ID)))
    ds = _DS_CACHE[K]

    stride = max(1, len(ds) // (2 * batch + 2))
    bt = _make_batch(ds, [stride * (i + 1) for i in range(batch)], preprocessor)
    gt = bt.get("action")
    if gt is None:
        out["verdict"] = "no_ground_truth"
        print("\n!! batch 에 'action' 이 없다.")
        return out
    gt = gt.detach().float().cpu()

    b = extract_bk(policy, bt)
    variants = make_variants(b, alphas=alphas, seed=seed)
    if verbose:
        print(f"[b_k ] shape={tuple(b.shape)}  ||b|| 평균={b.norm(dim=-1).mean():.4f}")
        print(f"[변형] {len(variants)} 개")

    # 한 번만 컴파일하고 이후엔 _bk 만 갈아 끼운다.
    compile_decoder(policy, b)
    dec = policy.model.decoder
    res: dict = {}
    try:
        for name, tbl in variants.items():
            dec._bk = tbl.unsqueeze(0).to(b.device)
            res[name] = measure(policy, bt, gt, K)
            if verbose:
                r = res[name]
                print(f"   {name:<14} 활용률 {r['use_ratio']:.2f}  오차 {r['err_mean']:.4f}")
    finally:
        dec.__dict__.pop("forward", None)

    out["variants"] = res
    out["gt_scale"] = float(gt.abs().mean())
    out["ok"] = True
    if verbose:
        report(out)
    return out


def report(r: dict) -> None:
    v = r["variants"]
    K = r["K"]
    real, zero = v.get("real", {}), v.get("zero", {})
    span = zero.get("err_mean", 0) - real.get("err_mean", 0)   # b_k 가 버는 오차

    print("\n" + "=" * 70)
    print(f"{r['tag']}  (K={K})   정답 크기 {r['gt_scale']:.4f}")
    print("-" * 70)
    print(f"{'표':<14}{'활용률':>9}{'이웃cos':>10}{'오차':>10}{'real 대비':>11}{'회복률':>9}")
    print("-" * 70)
    named = [k for k in ("real", "zero", "mean", "shuffle", "random_ortho") if k in v]
    alph = sorted((k for k in v if k.startswith("alpha=")),
                  key=lambda k: float(k.split("=")[1]))
    for name in named + alph:
        d = v[name]
        rel = d["err_mean"] - real.get("err_mean", 0)
        # zero 를 0%, real 을 100% 로 두고 이 표가 얼마나 회복했는가
        rec = (zero.get("err_mean", 0) - d["err_mean"]) / span if span > 1e-9 else float("nan")
        print(f"{name:<14}{d['use_ratio']:>9.2f}{d['cos_adj_mean']:>10.3f}"
              f"{d['err_mean']:>10.4f}{rel:>+11.4f}{rec:>8.0%}")
        if name == "random_ortho":
            print("-" * 70)
    print("=" * 70)

    # ── 판정 ──────────────────────────────────────────────────────────────────
    mean_, sh, ro = v.get("mean"), v.get("shuffle"), v.get("random_ortho")
    if mean_ and span > 1e-9:
        rec_mean = (zero["err_mean"] - mean_["err_mean"]) / span
        print("\nmean — 크기·공통성분은 그대로, 위치별 변화만 제거")
        if rec_mean < 0.25:
            print(f"   회복률 {rec_mean:.0%}. 이득은 거의 전부 '위치마다 다르다' 에서 온다.")
            print("   큰 상수를 더하는 것 자체로는 설명되지 않는다.")
        elif rec_mean > 0.75:
            print(f"   회복률 {rec_mean:.0%}. 위치와 무관하게 큰 벡터를 더하는 것만으로 대부분 설명된다.")
            print("   '위치 분리' 해석을 다시 봐야 한다.")
        else:
            print(f"   회복률 {rec_mean:.0%}. 위치별 변화와 전역 성분이 둘 다 기여한다.")

    if sh and ro and span > 1e-9:
        print("\nshuffle vs random_ortho — 둘 다 모델이 못 본 표라 OOD 조건은 같다")
        d = ro["err_mean"] - sh["err_mean"]
        if d > 0.02 * max(r["gt_scale"], 1e-9):
            print(f"   shuffle 이 {d:.4f} 낮다 — 학습된 벡터 '집합' 자체에 값어치가 있다.")
        elif d < -0.02 * max(r["gt_scale"], 1e-9):
            print(f"   random_ortho 가 {-d:.4f} 낮다 — 분리만 되면 내용은 상관없다.")
        else:
            print(f"   차이 {abs(d):.4f} 로 사실상 같다 — 내용보다 '위치가 갈라진다' 가 본질이다.")
        if v.get("real") and sh["err_mean"] - real["err_mean"] > 0.5 * span:
            print("   단 real 과의 간격이 크다 — 어느 위치에 어느 벡터가 가는지도 중요하다.")

    if alph:
        best = min(alph + (["real"] if "real" in v else []), key=lambda k: v[k]["err_mean"])
        print(f"\n크기 스윕 — 오차 최소는 {best} ({v[best]['err_mean']:.4f})")
        if best == "real":
            print("   학습된 크기(α=1)가 최적이다. 우연히 큰 값이 아니라 맞춰진 값이다.")
        else:
            print(f"   α=1 이 최적이 아니다. 크기 자체는 덜 민감하다는 뜻이다.")

    print("\n⚠️ 변형은 전부 학습에 없던 표라 그 자체로 OOD 다.")
    print("   절대 수치가 아니라 변형끼리의 비교만 읽을 것.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=DEFAULT_TAG, help="쉼표로 여러 개, 또는 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=150_000, help="-1 이면 최신")
    ap.add_argument("--task", default="libero_10")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tags = list(K_TAGS.values()) if a.tags == "all" else [t.strip() for t in a.tags.split(",")]
    res: dict = {}
    for tag in tags:
        if len(tags) > 1:
            print("\n" + "#" * 70 + f"\n# {tag}\n" + "#" * 70)
        try:
            res[tag] = run(tag=tag, seed=a.seed, step=None if a.step < 0 else a.step,
                           task=a.task, batch=a.batch, device=a.device, verbose=True)
        except FileNotFoundError as e:
            print(f"건너뜀 — {e}")
            res[tag] = {"tag": tag, "ok": False, "verdict": "no_checkpoint", "error": str(e)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\n건너뜀 — {tag} 에서 {type(e).__name__}: {e}")
            res[tag] = {"tag": tag, "ok": False, "verdict": "error", "error": str(e)}

    if a.json:
        import json as _json
        p = Path(a.json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[json] {p}")

    return 0 if all(r.get("ok") for r in res.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
