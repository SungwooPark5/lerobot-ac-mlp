"""diag1 — BiMamba ≡ 순방향 스택 + 위치별 상수표 b_k 임을 증명한다.

diag0 이 역방향 브랜치의 쿼리 위치 출력이 관측과 무관한 상수임을 확정했다.
상수라면 한 번 뽑아서 표로 저장할 수 있고, 역방향 스캔을 통째로 지워도
출력이 똑같아야 한다. 근사가 아니라 항등식이다.

하는 일:
  1. b_k 추출          — forward pass 한 번. (K, D) 상수표
  2. 디코더 컴파일      — backward_layers 를 b_k lookup 으로 대체
  3. 항등식 검증        — 원본 vs 컴파일 출력의 max|Δ| (0 이어야 함)
  4. 파라미터·지연 비교  — 얼마나 줄어드는가
  5. b_k 구조 분석      — 위치별 norm, 위치 간 코사인 유사도 행렬
                          ("위치 정체성" 가설의 직접 증거)

좌표 주의: 역방향 스택은 flip 된 시퀀스를 먹으므로 flip 좌표의 앞쪽 K개가
쿼리다. 원래 순서로 되돌리려면 뒤집어야 한다 — b = bwd_flipped[:K].flip(0).

주의: 노트북 커널이 아니라 mamba_ssm 이 깔린 venv 로 돌려야 한다
(common_v23.py:43-44). diag1_bk_identity.ipynb 이 알아서 그 venv 로 부른다.

셸에서:
    PY=~/lerobot_project/lerobot_env/bin/python
    PYTHONPATH=src $PY notebooks/libero/diag1_bk_identity.py --tags all --json /tmp/diag1.json
"""

from __future__ import annotations

import argparse
import time
import types
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.acm2_sscp_literal.modeling_acm2_sscp_literal import mamba2_stateful_forward
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

from diag0_backward_const import DEFAULT_TAG, K_TAGS, REPO_ID, _make_batch, resolve_ckpt

_DS_CACHE: dict = {}


# ── 1. b_k 추출 ────────────────────────────────────────────────────────────────

def extract_bk(policy, batch) -> torch.Tensor:
    """역방향 브랜치가 쿼리 위치에 내놓는 상수벡터를 (K, D) 로 뽑는다."""
    dec = policy.model.decoder
    K = policy.config.chunk_size
    cap: list[torch.Tensor] = []
    h = dec.backward_layers[-1].out_proj.register_forward_hook(
        lambda m, i, o: cap.append(o.detach()))
    try:
        with torch.no_grad():
            policy.reset()
            policy.predict_action_chunk(batch)
    finally:
        h.remove()
    if not cap:
        raise RuntimeError("역방향 출력을 잡지 못했다. hook 지점을 확인할 것.")
    # flip 좌표의 앞쪽 K개가 쿼리. 원래 순서로 되돌린다.
    # 배치 전체가 동일하므로(diag0 확정) 0번만 쓴다.
    return cap[0][0, :K, :].flip(0).contiguous()


# ── 2. 디코더 컴파일 ───────────────────────────────────────────────────────────

def compile_decoder(policy, b: torch.Tensor) -> None:
    """decoder.forward 를 역방향 스캔 없는 버전으로 갈아끼운다 (제자리 수정)."""
    dec = policy.model.decoder
    dec._bk = b.unsqueeze(0)  # (1, K, D)

    def compiled_forward(self, x, encoder_out, decoder_pos_embed=None,
                         encoder_pos_embed=None, carry=None):
        if decoder_pos_embed is not None:
            x = x + decoder_pos_embed
        if encoder_pos_embed is not None:
            encoder_out = encoder_out + encoder_pos_embed

        x = x.transpose(0, 1)                      # (B, K, D)
        encoder_out = encoder_out.transpose(0, 1)  # (B, T, D)
        seq = torch.cat([encoder_out, x], dim=1)

        new_states = []
        for i, layer in enumerate(self.forward_layers):
            init = carry[i] if carry is not None else None
            seq, st = mamba2_stateful_forward(layer, seq, initial_state=init, return_state=True)
            new_states.append(st)

        K = x.shape[1]
        # 원본: combined = 0.5*(fwd + bwd) 후 마지막 K개 슬라이스.
        # 여기서는 마지막 K개만 쓰면 되므로 먼저 자른다 (같은 연산).
        out = 0.5 * (seq[:, -K:, :] + self._bk.to(seq.dtype))

        if self.use_action_self_attention:  # 기본 off. 켜져 있으면 원본 그대로 재현.
            out_t = out.transpose(0, 1)
            residual = out_t
            out_norm = self.action_self_attn_norm(out_t)
            attn_delta = self.action_self_attn(out_norm, out_norm, value=out_norm,
                                               need_weights=False)[0]
            attn_delta = self.action_self_attn_dropout(attn_delta)
            if self.action_self_attention_use_gate:
                out_t = residual + torch.tanh(self.action_self_attn_gamma) * attn_delta
            else:
                out_t = residual + attn_delta
            out = out_t.transpose(0, 1)

        out = self.norm(out)
        return out.transpose(0, 1), new_states

    dec._orig_forward = dec.forward
    dec.forward = types.MethodType(compiled_forward, dec)


# ── 4. 지연 측정 ───────────────────────────────────────────────────────────────

def timeit(policy, batch, n_warmup: int = 5, n_iter: int = 20) -> float:
    """predict_action_chunk 한 번의 평균 시간(ms)."""
    cuda = next(policy.parameters()).is_cuda
    with torch.no_grad():
        for _ in range(n_warmup):
            policy.reset()
            policy.predict_action_chunk(batch)
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            policy.reset()
            policy.predict_action_chunk(batch)
        if cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e3


# ── 실행 ───────────────────────────────────────────────────────────────────────

def run(tag: str = DEFAULT_TAG, seed: int = 0, step: int | None = 150_000,
        task: str = "libero_10", batch: int = 4, device: str | None = None,
        save_dir: str | Path | None = None, verbose: bool = True) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {"tag": tag, "ok": False}

    ckpt = resolve_ckpt(tag, seed, step, task)
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    K, D = cfg.chunk_size, cfg.dim_model
    out.update(ckpt=str(ckpt), K=K, dim_model=D)
    if verbose:
        print(f"[ckpt] {ckpt}\n[cfg ] {cfg.type}  K={K}  D={D}  "
              f"bimamba={getattr(cfg, 'use_bimamba_decoder', None)}")

    if not getattr(cfg, "use_bimamba_decoder", False):
        out["verdict"] = "not_bimamba"
        print("\n!! BiMamba 가 아니다. tag 확인.")
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
    idx_a = [stride * (i + 1) for i in range(batch)]
    idx_b = [stride * (batch + i + 2) for i in range(batch)]
    batch_a = _make_batch(ds, idx_a, preprocessor)
    batch_b = _make_batch(ds, idx_b, preprocessor)

    # ── 1) b_k 추출 (batch_a 로) ────────────────────────────────────────────────
    b = extract_bk(policy, batch_a)
    if verbose:
        print(f"[b_k ] shape={tuple(b.shape)}  ||b|| 평균={b.norm(dim=-1).mean():.4f}")

    # ── 원본 출력 확보 ─────────────────────────────────────────────────────────
    with torch.no_grad():
        policy.reset(); act_a0 = policy.predict_action_chunk(batch_a).detach().float().cpu()
        policy.reset(); act_b0 = policy.predict_action_chunk(batch_b).detach().float().cpu()
    t_orig = timeit(policy, batch_a)

    # ── 2) 컴파일 후 재측정 ────────────────────────────────────────────────────
    compile_decoder(policy, b)
    with torch.no_grad():
        policy.reset(); act_a1 = policy.predict_action_chunk(batch_a).detach().float().cpu()
        policy.reset(); act_b1 = policy.predict_action_chunk(batch_b).detach().float().cpu()
    t_comp = timeit(policy, batch_a)

    # ── 3) 항등식 ──────────────────────────────────────────────────────────────
    # b_k 를 뽑은 batch_a 뿐 아니라, 한 번도 안 쓴 batch_b 에서도 같아야 한다.
    d_a = (act_a0 - act_a1).abs().max().item()
    d_b = (act_b0 - act_b1).abs().max().item()
    scale = act_a0.abs().mean().item()
    out.update(identity_max_diff_seen=d_a, identity_max_diff_unseen=d_b,
               action_scale=scale, identity_rel=max(d_a, d_b) / max(scale, 1e-12))

    # ── 4) 파라미터·지연 ───────────────────────────────────────────────────────
    dec = policy.model.decoder
    n_bwd = sum(p.numel() for p in dec.backward_layers.parameters())
    n_fwd = sum(p.numel() for p in dec.forward_layers.parameters())
    out.update(n_params_backward=n_bwd, n_params_forward=n_fwd, n_params_table=K * D,
               latency_orig_ms=t_orig, latency_compiled_ms=t_comp,
               speedup=t_orig / t_comp if t_comp else float("nan"))

    # ── 5) b_k 구조 ────────────────────────────────────────────────────────────
    bn = b.norm(dim=-1)
    bu = torch.nn.functional.normalize(b.float(), dim=-1)
    cos = (bu @ bu.T).cpu()                       # (K, K)
    adj = torch.diagonal(cos, offset=1)           # 이웃 위치 간 유사도
    far = cos[0, -1].item()                       # 첫 위치 vs 마지막 위치
    out.update(b_norm=[float(x) for x in bn.cpu()],
               cos_adjacent=[float(x) for x in adj],
               cos_first_last=far,
               cos_offdiag_mean=float((cos.sum() - cos.diagonal().sum()) / (len(cos) ** 2 - len(cos))))

    if save_dir:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / f"bk_{tag}.npz"
        import numpy as np
        np.savez_compressed(p, b=b.float().cpu().numpy(), cos=cos.numpy())
        out["npz"] = str(p)

    out["ok"] = max(d_a, d_b) == 0.0
    out["verdict"] = "identical" if out["ok"] else (
        "numerically_equal" if out["identity_rel"] < 1e-5 else "different")

    if verbose:
        report(out)
    return out


def report(r: dict) -> None:
    K = r["K"]
    print("\n" + "=" * 70)
    print(f"{r['tag']}  (K={K}, D={r['dim_model']})")
    print("-" * 70)
    print("항등식  원본 vs 컴파일 (역방향 스캔 삭제, b_k 표로 대체)")
    print(f"   b_k 를 뽑은 배치      max|Δ| = {r['identity_max_diff_seen']:.3e}")
    print(f"   처음 보는 배치        max|Δ| = {r['identity_max_diff_unseen']:.3e}")
    print(f"   (action 평균 크기 {r['action_scale']:.4f},  상대 {r['identity_rel']:.3e})")
    print("-" * 70)
    print("파라미터")
    print(f"   역방향 스택 {r['n_params_backward']:>12,d}")
    print(f"   b_k 표      {r['n_params_table']:>12,d}   "
          f"({r['n_params_backward'] / max(r['n_params_table'], 1):.0f}배 감소)")
    print(f"   순방향 스택 {r['n_params_forward']:>12,d}  (그대로)")
    print("-" * 70)
    print(f"지연  원본 {r['latency_orig_ms']:.2f} ms → 컴파일 {r['latency_compiled_ms']:.2f} ms "
          f"({r['speedup']:.2f}배)")
    print("-" * 70)
    bn = r["b_norm"]
    adj = r["cos_adjacent"]
    print("b_k 구조")
    print(f"   ||b_k||   평균 {sum(bn)/len(bn):.3f}   최소 {min(bn):.3f}   최대 {max(bn):.3f}")
    print(f"   이웃 위치 코사인  평균 {sum(adj)/len(adj):.3f}   최소 {min(adj):.3f}")
    print(f"   첫↔마지막 위치    {r['cos_first_last']:.3f}")
    print(f"   전체 비대각 평균  {r['cos_offdiag_mean']:.3f}")
    print("=" * 70)

    v = r["verdict"]
    if v == "identical":
        print("\n확정 — 출력이 비트 단위로 같다.")
        print("       BiMamba = 자기 순방향 스택 + 고정 b_k 표. 근사가 아니라 항등식이다.")
        print("       역방향 Mamba-2 스캔은 추론에서 삭제 가능하다.")
    elif v == "numerically_equal":
        print(f"\n확정(수치) — 상대오차 {r['identity_rel']:.1e}. 부동소수점 오차 수준이다.")
        print("       결론은 같다: 역방향 스캔은 b_k 표로 대체 가능하다.")
    else:
        print(f"\n불일치 — 상대오차 {r['identity_rel']:.1e}. 컴파일 경로가 원본과 다르다.")
        print("       use_action_self_attention 이나 carry 설정을 확인할 것.")

    if min(adj) > 0.9 and r["cos_first_last"] < min(adj):
        print("\n       b_k 는 이웃끼리 비슷하고 멀수록 달라진다 — 매끄러운 위치 부호다.")
        print("       '긴 스캔이 잃은 위치 정체성을 head 직전에 복원한다' 가설과 맞는다.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default=DEFAULT_TAG, help="쉼표로 여러 개, 또는 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--step", type=int, default=150_000, help="-1 이면 최신")
    ap.add_argument("--task", default="libero_10")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dir", default=None, help="b_k 를 npz 로 저장할 디렉토리")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    tags = list(K_TAGS.values()) if a.tags == "all" else [t.strip() for t in a.tags.split(",")]
    res: dict = {}
    for tag in tags:
        if len(tags) > 1:
            print("\n" + "#" * 70 + f"\n# {tag}\n" + "#" * 70)
        try:
            res[tag] = run(tag=tag, seed=a.seed, step=None if a.step < 0 else a.step,
                           task=a.task, batch=a.batch, device=a.device,
                           save_dir=a.save_dir, verbose=True)
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
