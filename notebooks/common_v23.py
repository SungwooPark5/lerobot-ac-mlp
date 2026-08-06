"""common_v23.py — v23 = smooth 변형 5종 (S3/S2/S7=MOSAIC/S6/S5) on eunji ACM2 literal-carry.

v21 literal-carry family(`acm2_sscp_literal[_smooth]`) 위에 얹은 5개 신규 정책. base 코드는
무수정, 전부 순수 subclass. 자세한 설계 = v23/IMPLEMENTATION_PLAN.md.

  literal   acm2_sscp_literal                    — literal (conv, ssm) 상태 carry (baseline)
  smooth    acm2_sscp_literal_smooth             — + GT-매칭 경계 FD 손실 (C1) (v21 baseline)
  s3_state  acm2_sscp_literal_smooth_state       — S3: carry-vs-fresh 상태연속성 손실 ⭐⭐
  s2_spec   acm2_sscp_literal_smooth_spectral    — S2: 주파수도메인(고주파 초과분) 경계 손실 ⭐
  mosaic    acm2_sscp_literal_smooth_overlap     — MOSAIC(옛 S7): overlap-add crossfade (추론만) ⭐
  s6_blend  acm2_sscp_literal_smooth_blend       — S6: α·carry 선형 상태보간 handoff
  s5_vel    acm2_sscp_literal_smooth_velint      — S5: 증분예측+proprioception 적분 (구조적 C0)

flag(신규):
  --policy.sscp_state_weight
  --policy.sscp_spectral_weight / sscp_spectral_halfwin / sscp_spectral_high_frac
  --policy.sscp_overlap / sscp_overlap_window
  --policy.sscp_blend_alpha / sscp_blend_learnable / sscp_blend_conv
  --policy.sscp_velint_anchor
+ v21 공통: sscp_enabled / sscp_p_carry / sscp_detach / sscp_smooth_weight / sscp_smooth_order ...

OUTPUT_BASE = outputs/v23.  ⚠️ import only ONE common_vXX per kernel.
"""
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# 이 파일은 레포 안(<repo>/notebooks/)에 있음 → 레포 루트를 파일 위치에서 유도(어디에 clone 하든 동작).
# 환경변수로 덮어쓸 수 있음:  LEROBOT_REPO / LEROBOT_PYTHON / LEROBOT_OUTPUT
HOME         = Path.home()
_ENV_REPO    = os.environ.get("LEROBOT_REPO")
PROJECT_ROOT = Path(_ENV_REPO) if _ENV_REPO else Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "src" / "lerobot").is_dir():      # 폴백: 옛 클러스터 레이아웃
    PROJECT_ROOT = HOME / "lerobot_project" / "lerobot-ac-mlp"
SRC_DIR      = PROJECT_ROOT / "src"
OUTPUT_BASE  = Path(os.environ.get("LEROBOT_OUTPUT", HOME / "lerobot_project" / "outputs")) / "v23"

# 학습/eval subprocess 가 쓸 python (= mamba_ssm 이 깔린 venv). 노트북 커널 python 과 다를 수 있음.
PYTHON = os.environ.get("LEROBOT_PYTHON") or str(HOME / "lerobot_project" / "lerobot_env" / "bin" / "python")
_site = glob.glob(str(HOME / "lerobot_project" / "lerobot_env" / "lib" / "python3.*" / "site-packages"))
SITE_DIR = _site[0] if _site else None
for _p in [str(SRC_DIR), SITE_DIR]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# ── Tasks / seeds / hyper-params ──────────────────────────────────────────────
TASKS: dict[str, tuple[str, str, str]] = {
    "transfer":  ("lerobot/aloha_sim_transfer_cube_human", "aloha", "AlohaTransferCube-v0"),
    "insertion": ("lerobot/aloha_sim_insertion_human",     "aloha", "AlohaInsertion-v0"),
}
PRIMARY_TASK = "transfer"
SEEDS = [0, 1, 2]
PRIMARY_SEED = 0
FPS = 50.0                       # ALOHA control freq (SPARC)

STEPS = 200_000
BATCH_SIZE = 8
EVAL_FREQ = 10_000        # eval at 100k,110k,...,200k
EVAL_START = 100_000      # 학습중 eval은 100k부터
TRAIN_EVAL_N = 500        # 각 학습중 eval 500 에피소드
SAVE_FREQ = 10_000
EVAL_N_EPISODES = 500
EVAL_BATCH_SIZE = 50
# LIBERO 는 task 10개 → 동시 env = eval.batch_size × 10. 크게 두면 렌더 컨텍스트가 VRAM 을 먹어 OOM.
LIBERO_EVAL_BATCH = 5


def _gpu_env(gpu_id):
    """모든 학습/eval 커맨드 앞에 붙는 per-run 환경변수.

    · MUJOCO_EGL_DEVICE_ID={gpu} : robosuite 는 렌더 디바이스가 CUDA_VISIBLE_DEVICES 안에
      있어야 assert 통과. _render_prefix 의 고정값 0 을 여기서 덮어씀(뒤에 오는 게 이김).
    · MPLBACKEND=Agg : subprocess 가 Jupyter 커널의 inline 백엔드를 상속하면 libero 가 죽음.
    """
    return ["MPLBACKEND=Agg", "HF_HUB_DISABLE_XET=1",
            f"MUJOCO_EGL_DEVICE_ID={gpu_id}", f"CUDA_VISIBLE_DEVICES={gpu_id}"]


def _eval_batch(task, n_episodes):
    return LIBERO_EVAL_BATCH if str(task).startswith("libero") else min(n_episodes, EVAL_BATCH_SIZE)

LR_CARRY = 1e-5   # (2026-07-12) 전 모델 lr 통일 → 옛 스윕도 1e-5

# ── Common flags ──────────────────────────────────────────────────────────────
_CARRY_ON  = "--policy.sscp_enabled=true"       # literal state carry on
_CARRY_OFF = "--policy.sscp_enabled=false"      # carry off → plain acm2 (zero-carry)
_SW  = lambda w: f"--policy.sscp_smooth_weight={w}"      # base GT-매칭 FD seam λ
_SO  = lambda o: f"--policy.sscp_smooth_order={o}"       # 1=C1 / 2=C1+C2
# ── v23 신규 변형 flag ──
_STATE = lambda w: f"--policy.sscp_state_weight={w}"                 # S3
_SPEC  = lambda w: f"--policy.sscp_spectral_weight={w}"              # S2
_SPECHW = lambda n: f"--policy.sscp_spectral_halfwin={n}"
_SPECHF = lambda f: f"--policy.sscp_spectral_high_frac={f}"
_OVL   = lambda n: f"--policy.sscp_overlap={n}"                      # MOSAIC
_OVLW  = lambda s: f"--policy.sscp_overlap_window={s}"
_OLAW  = lambda w: f"--policy.sscp_overlap_train_weight={w}"         # MOSAIC-trained (overlap-consistency)
_BIMAMBA_ON = "--policy.use_bimamba_decoder=true"
_BLEND = lambda a: f"--policy.sscp_blend_alpha={a}"                  # S6
_BLEARN = "--policy.sscp_blend_learnable=true"
_VEL   = lambda a: f"--policy.sscp_velint_anchor={a}"                # S5

# LR (2026-07-12 지시): ACT/ACM/ACM2/ours 전부 **1e-5 통일**. (이전: mamba계열만 3e-5)
# → 같은 lr 에서 비교해야 백본 차이가 lr 차이로 오염되지 않음.
# ⚠️ Diffusion/SmolVLA 는 논문 기본값(1e-4)을 그대로 씀 — 남의 방법을 우리 lr 로 깎으면
#    baseline 이 불공정해져 리뷰어가 문제 삼음. 아래 BASELINE_LR 참고.
LR = 1e-5
BASELINE_LR = 1e-4   # diffusion / smolvla 원 논문·lerobot 기본값
_OV = 10   # overlap steps (K=100 → hop=90)

# ── Model configs: (policy_type, lr, K, extra, use_chunk_pairs) ───────────────
# The 8-model v23 set. mosaic family (infer/train/+s3/+bimamba) is ONE overlap policy + flags.
MODEL_CONFIGS: dict[str, tuple[str, float, int, list[str], bool]] = {
    # ── baselines ──
    "act":  ("act",              LR,   100, [],           False),  # Transformer decoder reference
    "acm2": ("acm2_sscp_literal", LR,  100, [_CARRY_OFF], False),  # carry-off control (= literal minus carry)
    # ── 외부 정책 baselines (2026-07-12 추가; 각자 원 논문 lr=1e-4 유지) ──
    # diffusion: chunk_size 없음 → horizon(=K) 사용. horizon % 8 == 0 필수(UNet downsample).
    #   published DP 설정: horizon 16 / n_action_steps 8 / n_obs_steps 2.
    "diffusion": ("diffusion", BASELINE_LR, 16,
                  ["--policy.n_action_steps=8", "--policy.n_obs_steps=2"], False),
    # smolvla: VLA. 스크래치 학습은 비현실적 → smolvla_base 파인튜닝(PRETRAINED_INIT).
    "smolvla":   ("smolvla",   BASELINE_LR, 100, [], False),
    # ── mosaic family (overlap policy; carry on) ──
    "mosaic_infer": ("acm2_sscp_literal_smooth_overlap", LR, 100,
                 [_CARRY_ON, _OVL(_OV), _OVLW("hann")], True),                       # overlap at inference only
    "mosaic_train": ("acm2_sscp_literal_smooth_overlap", LR, 100,
                 [_CARRY_ON, _OVL(_OV), _OVLW("hann"), _OLAW(0.1)], True),           # + overlap-consistency training
    "mosaic_s3":    ("acm2_sscp_literal_smooth_overlap", LR, 100,
                 [_CARRY_ON, _OVL(_OV), _OVLW("hann"), _STATE(0.1)], True),          # overlap infer + S3 state loss
    # ── 실험2: MOSAIC 구성요소 분리 (ACM2 기준, carry ⊕ overlap). acm2=둘다 off ──
    "acm2_carry":   ("acm2_sscp_literal", LR, 100, [_CARRY_ON], True),               # carry만 (overlap X)
    "acm2_overlap": ("acm2_sscp_literal_smooth_overlap", LR, 100,
                 [_CARRY_OFF, _OVL(_OV), _OVLW("hann")], False),                     # overlap만 (carry X) — 신규
    "acm2_mosaic":  ("acm2_sscp_literal_smooth_overlap", LR, 100,
                 [_CARRY_ON, _OVL(_OV), _OVLW("hann")], True),                       # carry+overlap = MOSAIC(=mosaic_infer)
    # ── bimamba family ──
    "bimamba":       ("acm2_sscp_literal_bimamba", LR, 100, [_CARRY_ON], True),      # literal carry + BiMamba
    "bimamba_mosaic":    ("acm2_sscp_literal_smooth_overlap", LR, 100,
                      [_CARRY_ON, _OVL(_OV), _OVLW("hann"), _BIMAMBA_ON], True),      # BiMamba + mosaic infer
    "bimamba_mosaic_s3": ("acm2_sscp_literal_smooth_overlap", LR, 100,
                      [_CARRY_ON, _OVL(_OV), _OVLW("hann"), _STATE(0.1), _BIMAMBA_ON], True),  # BiMamba + mosaic + s3
    # ── ★acm (Mamba-1) family — 현재 메인. acm2 보다 acm SR 이 좋아 스택을 Mamba-1 로 포팅 ──
    #    ablation 사다리: acm → +carry → +BiMamba → +MOSAIC(=ours). 각 단계가 논문 기여 1개.
    "acm":            ("acm_sscp_literal", LR, 100, [_CARRY_OFF], False),        # 바닥(plain Mamba dec)
    "acm_carry":      ("acm_sscp_literal", LR, 100, [_CARRY_ON], True),          # +carry(SSCP) 단독
    "acm_mosaic":         ("acm_sscp_literal_smooth_overlap", LR, 100,
                       [_CARRY_ON, _OVL(_OV), _OVLW("hann")], True),             # +carry+MOSAIC (BiMamba 없음)
    "acm_bimamba":    ("acm_sscp_literal_bimamba", LR, 100, [_CARRY_ON], True),  # +carry+BiMamba
    "ours":           ("acm_sscp_literal_smooth_overlap", LR, 100,
                       [_CARRY_ON, _OVL(_OV), _OVLW("hann"), _BIMAMBA_ON], True),  # ★최종 = carry+BiMamba+MOSAIC
    # ── legacy/optional (not in CORE) ──
    "literal": ("acm2_sscp_literal",        LR, 100, [_CARRY_ON], True),
    "smooth":  ("acm2_sscp_literal_smooth", LR, 100, [_CARRY_ON, _SW(0.1), _SO(1)], True),
    "literal_carryoff": ("acm2_sscp_literal", LR, 100, [_CARRY_OFF], False),
}

MODEL_LABELS = {
    "act":  "ACT (Transformer dec, baseline)",
    "acm2": "acm2 (no carry, baseline)",
    "mosaic_infer": "mosaic overlap (infer only)",
    "mosaic_train": "mosaic overlap (trained) ⭐",
    "mosaic_s3":    "mosaic overlap + s3 state",
    "bimamba":  "BiMamba (literal carry)",
    "bimamba_mosaic":    "BiMamba + mosaic",
    "bimamba_mosaic_s3": "BiMamba + mosaic + s3 ⭐",
    "diffusion": "Diffusion Policy (baseline)",
    "smolvla":   "SmolVLA (baseline)",
    "acm":            "ACM (Mamba dec, no carry)",
    "acm_carry":      "ACM + carry",
    "acm_mosaic":         "ACM + carry + MOSAIC",
    "acm_bimamba":    "ACM + carry + BiMamba",
    "ours":           "Ours (carry + BiMamba + MOSAIC) ★",
    "literal":  "Literal carry",
    "smooth":   "Smooth C1",
    "literal_carryoff": "Zero-carry control (= acm2)",
}

# ── Phase A 스윕 태그 (seed 0, transfer): λ*/α* 탐색 ─────────────────────────────
#   각 정책의 핵심 파라미터 1D 그리드. 경계값(weight=0 / α=1 / overlap=0)은 baseline
#   (headline 태그)이 이미 담당하므로 여기선 생략. mosaic은 학습 1회(mosaic_base) + eval-time override.
_pw = lambda x: f"{x:g}".replace(".", "p")     # 0.03→'0p03', 0.1→'0p1', 1.0→'1'


def _mk_sweep_a():
    sw = {}
    for w in (0.03, 0.1, 0.3, 1.0):            # S3 state_weight
        sw[f"s3_w{_pw(w)}"] = ("acm2_sscp_literal_smooth_state", LR_CARRY, 100,
                               [_CARRY_ON, _STATE(w)], True)
    for w in (0.03, 0.1, 0.3, 1.0):            # S2 spectral_weight (halfwin 8, high_frac 0.5)
        sw[f"s2_w{_pw(w)}"] = ("acm2_sscp_literal_smooth_spectral", LR_CARRY, 100,
                               [_CARRY_ON, _SPEC(w), _SPECHW(8), _SPECHF(0.5)], True)
    for a in (0.9, 0.7, 0.5, 0.3):             # S6 blend alpha (fixed)
        sw[f"s6_a{_pw(a)}"] = ("acm2_sscp_literal_smooth_blend", LR_CARRY, 100,
                               [_CARRY_ON, _BLEND(a)], True)
    sw["s6_learn"] = ("acm2_sscp_literal_smooth_blend", LR_CARRY, 100,   # per-layer 학습 α (init 0.9)
                      [_CARRY_ON, _BLEND(0.9), _BLEARN], True)
    sw["s5_state"] = ("acm2_sscp_literal_smooth_velint", LR_CARRY, 100,  # anchor=proprioception
                      [_CARRY_ON, _VEL("state")], True)
    sw["s5_zero"] = ("acm2_sscp_literal_smooth_velint", LR_CARRY, 100,   # anchor=zero (ablation)
                     [_CARRY_ON, _VEL("zero")], True)
    sw["s5_state_c1"] = ("acm2_sscp_literal_smooth_velint", LR_CARRY, 100,  # 구조 C0 + 명시 C1 seam
                         [_CARRY_ON, _VEL("state"), _SW(0.1), _SO(1)], True)
    sw["mosaic_base"] = ("acm2_sscp_literal_smooth_overlap", LR_CARRY, 100,  # 학습 1회; overlap은 eval override
                     [_CARRY_ON, _OVL(0)], True)
    return sw


_SWEEP_A = _mk_sweep_a()
MODEL_CONFIGS.update(_SWEEP_A)


def _sweep_label(tag):
    if tag.startswith("s3_w"):
        return f"S3 state λ={tag[4:].replace('p', '.')}"
    if tag.startswith("s2_w"):
        return f"S2 spectral λ={tag[4:].replace('p', '.')}"
    if tag.startswith("s6_a"):
        return f"S6 blend α={tag[4:].replace('p', '.')}"
    return {
        "s6_learn": "S6 blend α (learnable)",
        "s5_state": "S5 velint (anchor=state)",
        "s5_zero": "S5 velint (anchor=zero, ablation)",
        "s5_state_c1": "S5 velint + C1 seam",
        "mosaic_base": "MOSAIC overlap base (eval-time sweep)",
    }.get(tag, tag)


MODEL_LABELS.update({t: _sweep_label(t) for t in _SWEEP_A})

# Phase A 학습 태그 (정책별 그룹 순서). mosaic_base는 학습 1회(overlap은 eval에서 스윕).
SWEEP_A_TAGS = (
    [t for t in _SWEEP_A if t.startswith("s3_")]
    + [t for t in _SWEEP_A if t.startswith("s2_")]
    + [t for t in _SWEEP_A if t.startswith("s6_")]
    + [t for t in _SWEEP_A if t.startswith("s5_")]
    + [t for t in _SWEEP_A if t.startswith("mosaic_")]
)


def sweep_tags(policy=None):
    """Phase A 스윕 학습 태그 리스트. policy in {'s3','s2','s5','s6','mosaic'} 로 필터(None=전체).

    사용: v23.sweep_tags()      → 전체 Phase A (17 태그: s3×4·s2×4·s6×5·s5×3·mosaic_base×1)
          v23.sweep_tags('s3')  → ['s3_w0p03','s3_w0p1','s3_w0p3','s3_w1']
    ⚠ mosaic_base는 학습 1회만; overlap 스윕은 make_mosaic_sweep_eval_cmds()로 eval 단계에서.
    """
    if policy is None:
        return list(SWEEP_A_TAGS)
    return [t for t in SWEEP_A_TAGS if t.startswith(policy)]


MODEL_DIR_NAMES = {t: t for t in MODEL_CONFIGS}

# ── 정책별 커맨드 특수 처리 ──────────────────────────────────────────────────
# chunk_size 를 안 받는 정책(=diffusion 은 horizon 사용). MODEL_CONFIGS 의 K 를 그 필드로 보냄.
CHUNK_FIELD = {"diffusion": "horizon"}     # 나머지는 전부 chunk_size
# --policy.type 대신 사전학습 체크포인트에서 파인튜닝하는 정책.
PRETRAINED_INIT = {"smolvla": "lerobot/smolvla_base"}

# v23 = 8-model comparison (legacy). lr 은 이제 전 모델 1e-5 통일.
CORE_TAGS = ["act", "acm2", "mosaic_infer", "mosaic_train", "mosaic_s3",
             "bimamba", "bimamba_mosaic", "bimamba_mosaic_s3"]
EVAL_TAGS = CORE_TAGS

# Training split across two notebooks (4 + 4) for separate GPU sessions.
TRAIN_TAGS_A = ["act", "acm2", "mosaic_infer", "mosaic_train"]          # 11a — baselines + core mosaic
TRAIN_TAGS_B = ["mosaic_s3", "bimamba", "bimamba_mosaic", "bimamba_mosaic_s3"]  # 11b — mosaic+s3 + BiMamba family


# ── GPU allocation guard ──────────────────────────────────────────────────────
def available_gpus(requested=None):
    """Return the subset of `requested` GPU ids that actually exist on this node.

    Asserts CUDA is visible — the mamba_ssm SSD Triton kernel is CUDA-ONLY, so a CPU
    fallback silently crashes deep in the kernel. Falls back to [0] if `requested` has no
    valid id. Pass requested=None to use every visible GPU.
    """
    import torch

    assert torch.cuda.is_available(), (
        "CUDA not visible — mamba_ssm SSD kernel is CUDA-only. Run on a GPU node "
        "(check `nvidia-smi -L` and `torch.cuda.is_available()`); a CPU fallback crashes "
        "inside the Triton kernel."
    )
    n = torch.cuda.device_count()
    req = list(range(n)) if requested is None else [g for g in requested if 0 <= g < n]
    if not req:
        req = [0]
    return req


def cuda_report():
    """One-line CUDA visibility report (for notebook preflight)."""
    import torch

    print(
        f"CUDA available: {torch.cuda.is_available()} | device_count: {torch.cuda.device_count()} "
        f"| CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )


# ── OSMesa/EGL render env ─────────────────────────────────────────────────────
def _render_prefix():
    rb = os.environ.get("RENDER_BACKEND", "egl")
    if rb == "osmesa":
        return f"PYTHONPATH={SRC_DIR} MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa"
    icd = os.path.expanduser("~/icd/10_nvidia.json")
    gldir = os.path.expanduser("~/nvidia-gl/lib")
    pre = (f"LD_LIBRARY_PATH={gldir}:$LD_LIBRARY_PATH " if os.path.isdir(gldir) else "")
    pre += (f"__EGL_VENDOR_LIBRARY_FILENAMES={icd} " if os.path.exists(icd) else "")
    return f"PYTHONPATH={SRC_DIR} {pre}MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0"


# ── Paths / checkpoints ───────────────────────────────────────────────────────
def train_dir(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    return OUTPUT_BASE / "train" / task / MODEL_DIR_NAMES[tag] / f"seed{seed}"


def eval_clean_dir(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    return OUTPUT_BASE / "eval_clean" / task / MODEL_DIR_NAMES[tag] / f"seed{seed}"


def run_label(tag, seed, task):
    return f"{tag}/{task}/seed{seed}"


LAST_CHECKPOINT_LINK = "last"     # lerobot: <out>/checkpoints/last -> <step>


def _ckpts(root: Path):
    if not root.is_dir():
        return []
    return sorted([d for d in root.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda d: int(d.name))


def last_ckpt_step(out: Path):
    """이 출력 디렉토리가 도달한 마지막 체크포인트 step. 없으면 None."""
    cks = _ckpts(Path(out) / "checkpoints")
    return int(cks[-1].name) if cks else None


def last_train_config(out: Path):
    """resume 에 넘길 train_config.json 경로 (= 마지막 체크포인트 안). 없으면 None.

    lerobot 은 이 파일을 <out>/checkpoints/<step>/pretrained_model/ 에만 저장한다.
    checkpoints/last 는 그 step 을 가리키는 심볼릭 링크.
    """
    out = Path(out)
    for cand in (out / "checkpoints" / LAST_CHECKPOINT_LINK, *reversed(_ckpts(out / "checkpoints"))):
        tc = cand / "pretrained_model" / "train_config.json"
        if tc.exists():
            return tc
    return None


def _pretrained(ckpt: Path) -> Path:
    pm = ckpt / "pretrained_model"
    return pm if pm.exists() else ckpt


def get_last_checkpoint(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    c = _ckpts(train_dir(tag, seed, task) / "checkpoints")
    return c[-1] if c else None


def _eval_curve_dir(eval_dir) -> dict[int, float]:
    import re
    out = {}
    for p in glob.glob(str(Path(eval_dir) / "eval_step_*.json")):
        m = re.search(r"eval_step_(\d+)\.json", p)
        if not m:
            continue
        try:
            d = json.loads(Path(p).read_text())
        except Exception:
            continue
        sr = d.get("overall", {}).get("pc_success")
        if sr is not None:
            out[int(m.group(1))] = sr
    return dict(sorted(out.items()))


def eval_curve(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    return _eval_curve_dir(train_dir(tag, seed, task) / "eval")


def best_ckpt_dir(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK, how="last"):
    ckpts = {int(c.name): c for c in _ckpts(train_dir(tag, seed, task) / "checkpoints")}
    if not ckpts:
        return None
    if isinstance(how, int):
        return ckpts.get(how) or ckpts[min(ckpts, key=lambda s: abs(s - how))]
    if how == "last":
        return ckpts[max(ckpts)]
    curve = eval_curve(tag, seed, task)
    steps = [s for s in sorted(curve) if s in ckpts]
    if not steps:
        return ckpts[max(ckpts)]
    srs = [curve[s] for s in steps]
    if how == "robust":
        sm = [sum(srs[max(0, i - 1):i + 2]) / len(srs[max(0, i - 1):i + 2]) for i in range(len(srs))]
        return ckpts[steps[max(range(len(steps)), key=lambda i: (sm[i], steps[i]))]]
    return ckpts[steps[max(range(len(steps)), key=lambda i: (srs[i], steps[i]))]]


def show_eval_curve(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    curve = eval_curve(tag, seed, task)
    cks = {int(c.name) for c in _ckpts(train_dir(tag, seed, task) / "checkpoints")}
    if not curve:
        print(f"{tag}: eval 곡선 없음"); return
    print(f"{tag}  (★=ckpt)")
    for s in sorted(curve):
        print(f"   {s:>8,}  {curve[s]:5.1f}%  {'★' if s in cks else ''}")


# ── Training command (from-scratch, resume-aware) ─────────────────────────────
def make_train_cmd(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK, gpu_id=0,
                   resume=None, lr_override=None,
                   steps=None, out_dir=None, eval_start=None):
    policy_type, lr, K, extra, use_cp = MODEL_CONFIGS[tag]
    if lr_override is not None:
        lr = lr_override
    dataset, env_type, env_task = TASKS[task]
    out = Path(out_dir) if out_dir is not None else train_dir(tag, seed, task)
    parts = _gpu_env(gpu_id) + [f"{PYTHON} -m lerobot.scripts.lerobot_train"]
    # smolvla 처럼 사전학습 가중치에서 출발하는 정책은 --policy.type 대신 --policy.path.
    init = PRETRAINED_INIT.get(tag)
    parts.append(f"--policy.path={init}" if init else f"--policy.type={policy_type}")
    # diffusion 은 chunk_size 가 없고 horizon 을 씀 (CHUNK_FIELD).
    parts.append(f"--policy.{CHUNK_FIELD.get(policy_type, 'chunk_size')}={K}")
    # n_action_steps: temporal ensembling은 매 스텝 쿼리 → 1. 그 외엔 K (청크 경계마다 carry).
    if not any("n_action_steps" in e for e in extra):
        n_act = 1 if any("temporal_ensemble_coeff" in e for e in extra) else K
        parts.append(f"--policy.n_action_steps={n_act}")
    parts += [
        f"--policy.optimizer_lr={lr}", f"--dataset.repo_id={dataset}", "--dataset.video_backend=pyav",
        f"--env.type={env_type}", f"--env.task={env_task}",
        f"--batch_size={BATCH_SIZE}", f"--steps={steps or STEPS}",
        f"--eval_freq={EVAL_FREQ}", f"--eval_start_step={eval_start or EVAL_START}",
        f"--eval.n_episodes={TRAIN_EVAL_N}",
        f"--save_freq={SAVE_FREQ}", f"--seed={seed}", f"--output_dir={out}", "--policy.push_to_hub=false",
    ]
    # ── resume ────────────────────────────────────────────────────────────────
    # lerobot 은 train_config.json 을 **체크포인트 안**에만 쓴다:
    #   <out>/checkpoints/<step>/pretrained_model/train_config.json   (+ checkpoints/last 심볼릭 링크)
    # 예전엔 <out>/train_config.json 을 찾았는데 그런 파일은 존재하지 않는다 →
    #   · resume 이 **한 번도 안 걸려** 재실행할 때마다 step 0 부터 다시 학습했고
    #   · "config 없음 = 빈 디렉토리" 로 오판해 체크포인트째로 지울 뻔했다.
    tc = last_train_config(out)
    if resume is None:
        resume = tc is not None
    if resume:
        if tc is None:
            raise FileNotFoundError(
                f"resume 하려는데 체크포인트가 없다: {out}/checkpoints/  → 처음부터 학습해야 함"
            )
        parts += ["--resume=true", f"--config_path={tc}"]
    if use_cp:
        parts.append("--use_chunk_pairs=true")
    if str(task).startswith("libero"):        # LIBERO 는 동시 env = batch × 10 → 작게
        parts.append(f"--eval.batch_size={LIBERO_EVAL_BATCH}")
    # 기본은 이전 방식 그대로(config num_workers=4). 단, /dev/shm 없는 노드에서만
    #   export LEROBOT_NUM_WORKERS=0 하면 워커 끄기(shm 크래시 회피). 안 하면 안 붙음.
    _nw = os.environ.get("LEROBOT_NUM_WORKERS")
    if _nw is not None:
        parts.append(f"--num_workers={_nw}")
    parts.extend(extra)
    return " ".join(parts)


# ── Clean eval (official SR + actions 기록 → jerk) ────────────────────────────
def make_eval_cmd(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK, gpu_id=0,
                  n_episodes=None, select="last", extra_policy=None, out_dir=None):
    cd = best_ckpt_dir(tag, seed, task, how=select)
    if cd is None:
        raise FileNotFoundError(f"No checkpoint for {run_label(tag, seed, task)}")
    ckpt = _pretrained(cd)
    _ds, env_type, env_task = TASKS[task]
    n = n_episodes or EVAL_N_EPISODES
    out = Path(out_dir) if out_dir is not None else eval_clean_dir(tag, seed, task)
    out.mkdir(parents=True, exist_ok=True)
    # 표준 lerobot_eval 직접 호출(외란 X, 외부 스크립트 의존 X).
    # action(.pt) 은 fork 의 eval 루프가 RECORD_DIR 환경변수로 기록 → jerk 측정용.
    parts = _gpu_env(gpu_id) + [
        f"RECORD_DIR={out / 'actions'}",
        f"{PYTHON} -m lerobot.scripts.lerobot_eval", f"--policy.path={ckpt}",
        f"--env.type={env_type}", f"--env.task={env_task}",
        f"--eval.n_episodes={n}", f"--eval.batch_size={_eval_batch(task, n)}", f"--output_dir={out}",
    ]
    if extra_policy:
        parts.extend(extra_policy)          # e.g. eval-time --policy.sscp_overlap=... override
    return " ".join(parts)


# ── MOSAIC overlap: eval-time 스윕 (재학습 X — overlap은 추론 전용 파라미터) ──────────
def mosaic_overlap_grid():
    """(overlap, window) 조합. overlap=0=hard switch(baseline). 총 9 조합."""
    combos = [(0, "linear")]
    for ov in (5, 10, 20, 40):
        for win in ("linear", "hann"):
            combos.append((ov, win))
    return combos


def make_mosaic_sweep_eval_cmds(seed=PRIMARY_SEED, task=PRIMARY_TASK, gpu_id=0,
                            tag="mosaic_base", n_episodes=None, select="last"):
    """mosaic_base 체크포인트 1개를 overlap/window 조합마다 override해 평가(조합별 out 디렉토리).

    ⚠ run_perturb_eval가 --policy.* override(pretrained config 위 덮어쓰기)를 수용해야 함.
    무시되면 폴백: overlap별 학습 태그 등록(학습 동일하나 GPU 낭비) 또는 eval 스크립트에서
    로드 후 policy.config.sscp_overlap 직접 설정.
    """
    base = eval_clean_dir(tag, seed, task)
    labeled = []
    for ov, win in mosaic_overlap_grid():
        out = base / f"ov{ov}_{win}"
        extra = [f"--policy.sscp_overlap={ov}", f"--policy.sscp_overlap_window={win}"]
        labeled.append((f"{tag}/ov{ov}_{win}",
                        make_eval_cmd(tag, seed, task, gpu_id, n_episodes, select,
                                      extra_policy=extra, out_dir=out)))
    return labeled


def mosaic_sweep_jerk(seed=PRIMARY_SEED, task=PRIMARY_TASK, tag="mosaic_base"):
    """mosaic overlap 스윕의 조합별 매끄러움 지표 표(recorded actions)."""
    import smooth_metrics as sm
    base = eval_clean_dir(tag, seed, task)
    K = MODEL_CONFIGS[tag][2]
    print(f"== {task} / MOSAIC overlap 스윕 매끄러움 ==")
    print(f"{'combo':<14} {'b-jerk':>8} {'i-jerk':>8} {'contrast':>9} {'SPARC':>9} {'n_ep':>5}")
    print("-" * 60)
    for ov, win in mosaic_overlap_grid():
        trajs = _load_action_trajs(base / f"ov{ov}_{win}" / "actions")
        r = sm.aggregate_smoothness(trajs, K, fs=FPS)
        combo = f"ov{ov}/{win}"
        if not r or "boundary_jerk_contrast_mean" not in r:
            print(f"{combo:<14}  (actions 없음)"); continue
        print(f"{combo:<14} {r['boundary_jerk_mean']:>8.4f} {r['interior_jerk_mean']:>8.4f} "
              f"{r['boundary_jerk_contrast_mean']:>9.4f} {r['sparc_mean']:>9.3f} {r['n_episodes']:>5}")


# ── Launcher (EGL) ────────────────────────────────────────────────────────────
def launch_cmds_live(labeled_cmds, log_tag="job"):
    try:
        from IPython import get_ipython
        ip = get_ipython()
    except ImportError:
        ip = None
    env_prefix = _render_prefix()
    log_dir = OUTPUT_BASE / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for label, cmd in labeled_cmds:
        safe = label.replace("/", "__")
        log_path = log_dir / f"{log_tag}__{safe}.log"
        print(f"[{label}]  (log: {log_path})")
        parts.append(f"({env_prefix} {cmd} >> {log_path} 2>&1 && echo '[{label}] 완료 ✅' || echo '[{label}] 실패 ❌')")
    if not parts:
        print("실행할 명령 없음."); return
    pc = " & ".join(parts) + " & wait"
    print(f"\n{len(parts)}개 병렬 실행\n")
    (ip.system if ip is not None else lambda c: subprocess.run(c, shell=True, executable="/bin/bash"))(pc)


def clean_if_no_ckpt(out: Path):
    """체크포인트가 **하나도 없는** 출력 디렉토리만 정리한다(크래시 잔재).

    ⚠️ 체크포인트가 있으면 절대 지우지 않는다 — 예전 코드는 <out>/train_config.json 이
    없으면(그 파일은 애초에 거기 안 생긴다) 디렉토리를 통째로 rmtree 해서, 120k 까지 간
    학습도 날려버릴 수 있었다.
    """
    out = Path(out)
    if out.exists() and last_ckpt_step(out) is None:
        import shutil
        shutil.rmtree(out)
        return True
    return False


def launch_training_live(jobs, gpu_offset=0, skip_done=True):
    labeled = []
    for i, (tag, seed, task) in enumerate(jobs):
        if skip_done and get_training_status(tag, seed, task)["done"]:
            print(f"  [skip] {run_label(tag, seed, task)} done"); continue
        clean_if_no_ckpt(train_dir(tag, seed, task))
        labeled.append((run_label(tag, seed, task), make_train_cmd(tag, seed, task, gpu_offset + i)))
    launch_cmds_live(labeled, log_tag="train")


# ── Status ────────────────────────────────────────────────────────────────────
def get_training_status(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    out = train_dir(tag, seed, task)
    jsonl = out / "training_log.jsonl"
    st = {"tag": tag, "seed": seed, "task": task, "step": 0, "best_sr": None, "done": False}
    if jsonl.exists():
        last, best = 0, None
        for line in jsonl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            last = max(last, rec.get("step", 0))
            for k in ("pc_success", "eval/pc_success"):
                if k in rec:
                    v = rec[k] / 100.0
                    best = v if best is None else max(best, v)
        st.update(step=last, best_sr=best, done=last >= STEPS)
    return st


def get_eval_status(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    info = eval_clean_dir(tag, seed, task) / "eval_info.json"
    st = {"tag": tag, "seed": seed, "task": task, "sr": None, "n_done": 0, "done": False}
    if info.exists():
        overall = json.loads(info.read_text()).get("overall", {})
        pc = overall.get("pc_success")
        st.update(sr=(pc / 100.0 if pc is not None else None),
                  n_done=overall.get("n_episodes", EVAL_N_EPISODES), done=True)
    return st


def print_training_status(jobs):
    print(f"{'RUN':<30} {'STEP':>8} {'BEST_SR':>8} {'DONE':>6}")
    print("-" * 56)
    for tag, seed, task in jobs:
        s = get_training_status(tag, seed, task)
        sr = f"{s['best_sr']:.3f}" if s["best_sr"] is not None else "  ─"
        print(f"{run_label(tag, seed, task):<30} {s['step']:>8,} {sr:>8} {'YES' if s['done'] else 'no':>6}")


# ── SR aggregation (Wilson CI) ────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / denom
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, c - h), min(1.0, c + h)


def aggregate_seeds(tag, task=PRIMARY_TASK, seeds=None):
    import numpy as np
    seeds = seeds if seeds is not None else SEEDS
    srs, pk, pn, per = [], 0, 0, {}
    for seed in seeds:
        s = get_eval_status(tag, seed, task)
        per[seed] = s["sr"]
        if s["sr"] is not None and s["done"]:
            srs.append(s["sr"]); n = int(s["n_done"]) or EVAL_N_EPISODES
            pk += int(round(s["sr"] * n)); pn += n
    out = {"tag": tag, "label": MODEL_LABELS.get(tag, tag), "n_seeds": len(srs), "per_seed": per,
           "sr_mean": float(np.mean(srs)) if srs else None,
           "sr_std": float(np.std(srs, ddof=1)) if len(srs) > 1 else 0.0}
    if pn > 0:
        lo, hi = wilson_ci(pk, pn)
        out.update(pooled_n=pn, pooled_sr=pk / pn, pooled_ci_lo=lo, pooled_ci_hi=hi)
    return out


def print_aggregate_table(tags, task=PRIMARY_TASK):
    print(f"== {task} / clean SR ==  (mean±std over seeds)")
    print(f"{'RUN':<30} {'SR mean':>9} {'± std':>7} {'seeds':>6} {'pooled95%CI':>18}")
    print("-" * 74)
    for t in tags:
        r = aggregate_seeds(t, task)
        m = f"{r['sr_mean']:.3f}" if r["sr_mean"] is not None else "  ─"
        sd = f"{r['sr_std']:.3f}" if r["sr_mean"] is not None else "  ─"
        ci = f"[{r['pooled_ci_lo']:.3f},{r['pooled_ci_hi']:.3f}]" if "pooled_ci_lo" in r else "—"
        print(f"{r['label']:<30} {m:>9} {sd:>7} {r['n_seeds']:>6} {ci:>18}")


# ── Jerk/SPARC report (recorded eval actions → smooth_metrics) ─────────────────
def _load_action_trajs(actions_dir: Path):
    import numpy as np
    trajs = []
    if not actions_dir.is_dir():
        return trajs
    for p in sorted(glob.glob(str(actions_dir / "*.np[yz]"))):
        try:
            d = np.load(p, allow_pickle=True)
        except Exception:
            continue
        arr = None
        if isinstance(d, np.lib.npyio.NpzFile):
            for key in ("actions", "action", "arr_0"):
                if key in d:
                    arr = d[key]; break
            if arr is None and len(d.files):
                arr = d[d.files[0]]
        else:
            arr = d
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.ndim == 3:
            for i in range(arr.shape[0]):
                trajs.append(arr[i])
        elif arr.ndim == 2:
            trajs.append(arr)
    # Per-episode .pt files under action_logs/ (training-time eval / clean-eval via RECORD_DIR).
    pt_dir = actions_dir / "action_logs"
    if pt_dir.is_dir():
        import torch
        for p in sorted(glob.glob(str(pt_dir / "*.pt"))):
            try:
                t = torch.load(p, map_location="cpu")
            except Exception:
                continue
            if hasattr(t, "detach"):
                t = t.detach().cpu().numpy()
            arr = np.asarray(t)
            if arr.ndim == 3:
                for i in range(arr.shape[0]):
                    trajs.append(arr[i])
            elif arr.ndim == 2:
                trajs.append(arr)
    return trajs


def jerk_report(tag, seed=PRIMARY_SEED, task=PRIMARY_TASK, chunk_size=None):
    import smooth_metrics as sm
    K = chunk_size or MODEL_CONFIGS[tag][2]
    trajs = _load_action_trajs(eval_clean_dir(tag, seed, task) / "actions")
    agg = sm.aggregate_smoothness(trajs, K, fs=FPS)
    agg["tag"], agg["label"] = tag, MODEL_LABELS.get(tag, tag)
    return agg


def print_jerk_table(tags, seed=PRIMARY_SEED, task=PRIMARY_TASK):
    print(f"== {task} / 매끄러움 (recorded actions) ==")
    print(f"{'RUN':<30} {'b-jerk':>8} {'i-jerk':>8} {'contrast':>9} {'SPARC':>9} {'n_ep':>5}")
    print("-" * 74)
    for t in tags:
        r = jerk_report(t, seed, task)
        if not r or "boundary_jerk_contrast_mean" not in r:
            print(f"{MODEL_LABELS.get(t, t):<30}  (actions 없음 — eval 먼저)"); continue
        print(f"{r['label']:<30} {r['boundary_jerk_mean']:>8.4f} {r['interior_jerk_mean']:>8.4f} "
              f"{r['boundary_jerk_contrast_mean']:>9.4f} {r['sparc_mean']:>9.3f} {r['n_episodes']:>5}")
    print("\n★ contrast(경계-내부 jerk)가 낮을수록 매끄러움. SPARC는 0에 가까울수록 매끄러움.")
