"""diag2 — 순방향 스택이 위치 정체성을 잃고, b_k 가 그걸 복원하는지 확인한다.

diag0/diag1 이 확정한 것:
    out_k = LayerNorm( 0.5 * ( fwd_k + b_k ) )        비트 단위 항등식
    b_k 는 관측 무관 상수이고, 위치끼리 거의 직교한다 (코사인 ~ 0).

아직 가설인 것 — 이 스크립트가 재는 것:
    "순방향 스택의 쿼리 위치 출력 fwd_k 는 K 가 커질수록 서로 구분이 안 되게 된다.
     b_k 는 head 직전에 위치를 다시 분리해 준다."

왜 그럴 거라고 보는가:
    디코더 입력은 [C ; Q] 이고 쿼리는 zeros + pos_embed 다. 스캔이 쿼리 위치에
    도달할 즈음 상태는 이미 관측 토큰 M 개를 흡수한 뒤라, 쿼리 하나하나는 큰
    상태 위의 작은 섭동이다. 그래서 fwd_k ~ fwd_{k+1} 이 되고, action head 는
    "몇 번째 액션인지"를 구분하지 못한 채 서로 다른 값을 내놓아야 한다.

재는 것 — 네 가지 표현을 같은 자로 잰다:
    fwd      순방향 스택 출력 (쿼리 위치)
    b        역방향 스택이 주는 상수표
    fused    LayerNorm( 0.5*(fwd + b) )    ← head 가 실제로 보는 것
    ablated  LayerNorm( 0.5*(fwd + 0) )    ← b_k 만 끈 것. 같은 체크포인트, 같은 순방향.

    fused vs ablated 가 핵심 대조다. 체크포인트도 순방향도 그대로고 b_k 만
    토글하므로, 차이는 전부 b_k 탓이다.

자:
    cos_adj    이웃 위치 간 코사인. 1 에 가까울수록 "구분 안 됨"
    eff_rank   특이값의 participation ratio. K 개 위치가 실질적으로 몇 개의
               서로 다른 방향을 쓰는가. K 면 완전 분리, 1 이면 전부 같은 방향.
    use_ratio  eff_rank / K. K 끼리 비교하려면 이걸 본다.

예측:
    fwd 의 use_ratio 는 낮고, K 가 커질수록 더 낮아진다.
    b 의 use_ratio 는 1 에 가깝다.
    fused 는 ablated 보다 확실히 높다.

두 번째 절(--errors)은 위치별 예측 오차를 b_k 켜고/끄고 잰다.
diag1 의 compile_decoder 를 그대로 써서 b_k 를 0 으로 바꾼 모델을 만든다.

주의: 노트북 커널이 아니라 mamba_ssm 이 깔린 venv 로 돌려야 한다
(common_v23.py:43-44). diag2_position_collapse.ipynb 이 알아서 그 venv 로 부른다.

셸에서:
    PY=~/lerobot_project/lerobot_env/bin/python
    PYTHONPATH=src $PY notebooks/libero/diag2_position_collapse.py --tags all --json /tmp/diag2.json
    PYTHONPATH=src $PY notebooks/libero/diag2_position_collapse.py --tags all --acm2 --errors
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

# 단방향 기준선. exp5_tonight.py:137 규약 — K=100 은 접미사 없음.
ACM2_TAGS = {50: "acm2_k50", 100: "acm2", 150: "acm2_k150"}

_DS_CACHE: dict = {}


# ── 자 ─────────────────────────────────────────────────────────────────────────

def _eff_rank(M: torch.Tensor) -> float:
    """특이값의 participation ratio. (K, D) -> [1, min(K,D)].

    K 개 행이 서로 직교하고 크기가 같으면 K, 전부 같은 방향이면 1 이 나온다.
    "이 위치들이 실질적으로 몇 개의 서로 다른 방향을 쓰는가" 를 한 숫자로 준다.
    """
    s = torch.linalg.svdvals(M.float())
    s = s[s > 1e-12]
    if s.numel() == 0:
        return 0.0
    return float(s.sum().pow(2) / s.pow(2).sum())


def separation(X: torch.Tensor) -> dict:
    """위치끼리 얼마나 구분되는가. X 는 (K, D) 또는 (B, K, D).

    배치가 있으면 샘플마다 재고 평균한다 (샘플 간 평균이 아니라 지표의 평균 —
    표현을 먼저 평균해 버리면 관측 의존 성분이 지워져서 fwd 가 과소평가된다).
    """
    if X.dim() == 2:
        X = X.unsqueeze(0)
    X = X.float()
    B, K, _ = X.shape

    cos_adj = torch.zeros(K - 1)
    offdiag = 0.0
    er = 0.0
    er_c = 0.0
    for i in range(B):
        M = X[i]                                          # (K, D)
        U = torch.nn.functional.normalize(M, dim=-1)
        C = U @ U.T                                       # (K, K)
        cos_adj += torch.diagonal(C, offset=1).cpu()
        offdiag += float((C.sum() - C.diagonal().sum()) / (K * K - K))
        er += _eff_rank(M)
        er_c += _eff_rank(M - M.mean(0, keepdim=True))     # 공통 성분 제거 후

    cos_adj /= B
    return {
        "cos_adj": [float(x) for x in cos_adj],
        "cos_adj_mean": float(cos_adj.mean()),
        "cos_offdiag_mean": offdiag / B,
        "eff_rank": er / B,
        "eff_rank_centered": er_c / B,
        "use_ratio": (er / B) / K,
        "K": K,
    }


# ── 표현 뽑기 ──────────────────────────────────────────────────────────────────

def capture(policy, batch, K: int) -> dict:
    """한 번의 forward 에서 fwd / b / fused / ablated 를 전부 뽑는다."""
    dec = policy.model.decoder
    has_bwd = getattr(dec, "backward_layers", None) is not None

    cap: dict[str, list] = {"fwd": [], "bwd": []}
    handles = [dec.forward_layers[-1].out_proj.register_forward_hook(
        lambda m, i, o, k="fwd": cap[k].append(o.detach().float()))]
    if has_bwd:
        handles.append(dec.backward_layers[-1].out_proj.register_forward_hook(
            lambda m, i, o, k="bwd": cap[k].append(o.detach().float())))
    try:
        with torch.no_grad():
            policy.reset()
            policy.predict_action_chunk(batch)
    finally:
        for h in handles:
            h.remove()

    if not cap["fwd"]:
        raise RuntimeError("순방향 출력을 잡지 못했다. hook 지점을 확인할 것.")

    # 순방향 입력은 [C ; Q] → 쿼리는 뒤쪽 K개
    fwd = cap["fwd"][0][:, -K:, :]                        # (B, K, D)
    rep = {"fwd": fwd}

    # LayerNorm 가중치 dtype 에 맞춰서 넣고 float 로 받는다 (half 체크포인트 대비).
    norm = dec.norm
    wdt = norm.weight.dtype

    def _ln(x):
        with torch.no_grad():
            return norm(x.to(wdt)).float()

    if has_bwd and cap["bwd"]:
        # 역방향 입력은 flip([C;Q]) → 쿼리는 앞쪽 K개. 원래 순서로 되돌린다.
        # 배치 전체가 동일하다는 것은 diag0 에서 확정 — 0번만 쓴다.
        b = cap["bwd"][0][0, :K, :].flip(0).contiguous()  # (K, D)
        rep["b"] = b
        rep["fused"] = _ln(0.5 * (fwd + b.unsqueeze(0)))
        # LayerNorm 이 스케일을 지우므로 0.5*fwd 와 fwd 는 같다. 원본과 형태만 맞춘 것.
        rep["ablated"] = _ln(0.5 * fwd)
    else:
        rep["ablated"] = _ln(0.5 * fwd)

    return rep


# ── 위치별 예측 오차 ───────────────────────────────────────────────────────────

def position_errors(policy, batch, K: int) -> dict:
    """b_k 켜고/끈 두 모델의 위치별 예측 오차. 정답이 배치에 없으면 건너뛴다."""
    from diag1_bk_identity import compile_decoder, extract_bk

    gt = batch.get("action") if hasattr(batch, "get") else None
    if gt is None:
        return {"ok": False, "why": "batch 에 'action' 키가 없다"}
    gt = gt.detach().float().cpu()
    if gt.dim() != 3 or gt.shape[1] != K:
        return {"ok": False, "why": f"action 모양이 (B,K,A) 가 아니다: {tuple(gt.shape)}"}

    b = extract_bk(policy, batch)

    with torch.no_grad():
        policy.reset()
        pred_on = policy.predict_action_chunk(batch).detach().float().cpu()

    # b_k 를 0 으로 준 컴파일 디코더 = 역방향 기여만 제거. 순방향은 그대로.
    dec = policy.model.decoder
    compile_decoder(policy, torch.zeros_like(b))
    try:
        with torch.no_grad():
            policy.reset()
            pred_off = policy.predict_action_chunk(batch).detach().float().cpu()
    finally:
        # 인스턴스 속성을 지워서 클래스의 원래 forward 가 다시 보이게 한다.
        dec.__dict__.pop("forward", None)

    err_on = (pred_on - gt).abs().mean(dim=(0, 2))        # (K,)
    err_off = (pred_off - gt).abs().mean(dim=(0, 2))
    return {
        "ok": True,
        "err_on": [float(x) for x in err_on],
        "err_off": [float(x) for x in err_off],
        "err_on_mean": float(err_on.mean()),
        "err_off_mean": float(err_off.mean()),
        "gt_scale": float(gt.abs().mean()),
        # 정답과 예측이 같은 공간인지 눈으로 확인하라는 것 — 스케일이 크게 다르면
        # 정규화 공간이 어긋난 것이라 오차 숫자를 믿으면 안 된다.
        "pred_scale": float(pred_on.abs().mean()),
    }


# ── 실행 ───────────────────────────────────────────────────────────────────────

def run(tag: str = DEFAULT_TAG, seed: int = 0, step: int | None = 150_000,
        task: str = "libero_10", batch: int = 4, device: str | None = None,
        errors: bool = False, save_dir: str | Path | None = None,
        verbose: bool = True) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {"tag": tag, "ok": False}

    ckpt = resolve_ckpt(tag, seed, step, task)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    K, D = cfg.chunk_size, cfg.dim_model
    bimamba = bool(getattr(cfg, "use_bimamba_decoder", False))
    out.update(ckpt=str(ckpt), K=K, dim_model=D, bimamba=bimamba)
    if verbose:
        print(f"[ckpt] {ckpt}\n[cfg ] {cfg.type}  K={K}  D={D}  bimamba={bimamba}")

    policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=ckpt,
        preprocessor_overrides={"device_processor": {"device": device}})

    if getattr(policy.model.decoder, "use_action_self_attention", False):
        out["warn"] = "use_action_self_attention=True — fused/ablated 는 근사다"
        if verbose:
            print("  !! " + out["warn"])

    if K not in _DS_CACHE:
        _DS_CACHE[K] = LeRobotDataset(
            REPO_ID, delta_timestamps=resolve_delta_timestamps(cfg, LeRobotDatasetMetadata(REPO_ID)))
    ds = _DS_CACHE[K]

    stride = max(1, len(ds) // (2 * batch + 2))
    idx = [stride * (i + 1) for i in range(batch)]
    bt = _make_batch(ds, idx, preprocessor)

    rep = capture(policy, bt, K)
    out["sep"] = {name: separation(X) for name, X in rep.items()}

    if errors:
        if bimamba:
            out["errors"] = position_errors(policy, bt, K)
        else:
            out["errors"] = {"ok": False, "why": "BiMamba 가 아니라 b_k 가 없다"}

    if save_dir:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        import numpy as np
        p = save_dir / f"sep_{tag}.npz"
        np.savez_compressed(p, **{k: v.cpu().numpy() for k, v in rep.items()})
        out["npz"] = str(p)

    out["ok"] = True
    if verbose:
        report(out)
    return out


def report(r: dict) -> None:
    K = r["K"]
    sep = r["sep"]
    print("\n" + "=" * 74)
    print(f"{r['tag']}  (K={K}, D={r['dim_model']})")
    print("-" * 74)
    print(f"{'표현':<10}{'이웃 코사인':>12}{'비대각 평균':>13}{'eff_rank':>11}"
          f"{'/K':>8}{'중심제거':>11}")
    print("-" * 74)
    order = [k for k in ("fwd", "b", "ablated", "fused") if k in sep]
    label = {"fwd": "fwd", "b": "b_k", "ablated": "ablated", "fused": "fused"}
    for name in order:
        s = sep[name]
        print(f"{label[name]:<10}{s['cos_adj_mean']:>12.3f}{s['cos_offdiag_mean']:>13.3f}"
              f"{s['eff_rank']:>11.1f}{s['use_ratio']:>8.2f}{s['eff_rank_centered']:>11.1f}")
    print("=" * 74)

    if "fused" in sep and "ablated" in sep:
        f, a = sep["fused"], sep["ablated"]
        d_cos = a["cos_adj_mean"] - f["cos_adj_mean"]
        d_use = f["use_ratio"] - a["use_ratio"]
        print(f"\nb_k 토글 (같은 체크포인트, 순방향 동일)")
        print(f"   이웃 코사인   {a['cos_adj_mean']:.3f} → {f['cos_adj_mean']:.3f}  "
              f"({d_cos:+.3f}, 낮을수록 잘 구분됨)")
        print(f"   위치 활용률   {a['use_ratio']:.2f} → {f['use_ratio']:.2f}  ({d_use:+.2f})")
        if d_use > 0.05 and d_cos > 0.02:
            print("\n   확인 — b_k 가 위치 분리를 실제로 복원한다.")
            print("          '긴 스캔이 위치를 지우고 역방향이 head 직전에 되살린다' 가설과 맞는다.")
        elif abs(d_use) <= 0.05:
            print("\n   미확인 — b_k 를 껐는데도 분리도가 별로 안 변한다.")
            print("            8.2 점의 원인을 위치 분리 말고 다른 데서 찾아야 한다.")

    f_ = sep.get("fwd")
    if f_ and f_["use_ratio"] < 0.5:
        print(f"\n   fwd 의 위치 활용률이 {f_['use_ratio']:.2f} 다 — "
              f"K={K} 위치가 실질 {f_['eff_rank']:.0f} 개 방향만 쓴다.")
        # 중심 제거 후에도 낮으면 위치 정보가 정말 없는 것이고,
        # 높으면 있긴 한데 공통 성분에 묻힌 것이다. 둘은 다른 얘기다.
        if f_["eff_rank_centered"] > 0.5 * K:
            print(f"        단 위치별 평균을 빼면 {f_['eff_rank_centered']:.0f} 로 올라간다 —")
            print(f"        위치 정보가 사라진 게 아니라 공통 성분에 묻힌 것이다.")
            print(f"        LayerNorm 은 토큰마다 D 축으로만 정규화하므로 이 공통 성분은")
            print(f"        head 까지 그대로 가고, b_k 가 위치별로 갈라 주는 역할을 한다.")
        else:
            print(f"        중심 제거 후에도 {f_['eff_rank_centered']:.0f} 다 — "
                  f"위치 정보 자체가 남아있지 않다.")

    e = r.get("errors")
    if e and e.get("ok"):
        n = len(e["err_on"])
        q = max(1, n // 4)
        print(f"\n위치별 예측 오차 (정답 대비 |Δ| 평균)")
        print(f"   정답 크기 {e['gt_scale']:.4f} / 예측 크기 {e['pred_scale']:.4f}"
              f"{'   !! 스케일이 크게 다르다 — 정규화 공간 확인 필요' if max(e['gt_scale'], e['pred_scale']) > 5 * max(min(e['gt_scale'], e['pred_scale']), 1e-9) else ''}")
        print(f"   {'구간':<12}{'b_k 켬':>10}{'b_k 끔':>10}{'차이':>10}")
        for i, lo in enumerate(range(0, n, q)):
            hi = min(lo + q, n)
            on = sum(e["err_on"][lo:hi]) / (hi - lo)
            off = sum(e["err_off"][lo:hi]) / (hi - lo)
            print(f"   k={lo+1:>3}–{hi:<7}{on:>10.4f}{off:>10.4f}{off - on:>+10.4f}")
        print(f"   {'전체':<12}{e['err_on_mean']:>10.4f}{e['err_off_mean']:>10.4f}"
              f"{e['err_off_mean'] - e['err_on_mean']:>+10.4f}")
    elif e:
        print(f"\n위치별 오차 건너뜀 — {e.get('why')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=DEFAULT_TAG, help="쉼표로 여러 개, 또는 'all'")
    ap.add_argument("--acm2", action="store_true", help="단방향 기준선도 같이 잰다")
    ap.add_argument("--errors", action="store_true", help="위치별 예측 오차도 잰다")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=150_000, help="-1 이면 최신")
    ap.add_argument("--task", default="libero_10")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    if a.tags == "all":
        tags = list(K_TAGS.values())
        if a.acm2:
            tags += list(ACM2_TAGS.values())
    else:
        tags = [t.strip() for t in a.tags.split(",")]

    res: dict = {}
    for tag in tags:
        if len(tags) > 1:
            print("\n" + "#" * 74 + f"\n# {tag}\n" + "#" * 74)
        try:
            res[tag] = run(tag=tag, seed=a.seed, step=None if a.step < 0 else a.step,
                           task=a.task, batch=a.batch, device=a.device,
                           errors=a.errors, save_dir=a.save_dir, verbose=True)
        except FileNotFoundError as e:
            print(f"건너뜀 — {e}")
            res[tag] = {"tag": tag, "ok": False, "verdict": "no_checkpoint", "error": str(e)}

    if a.json:
        import json as _json
        p = Path(a.json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[json] {p}")

    return 0 if all(r.get("ok") for r in res.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
