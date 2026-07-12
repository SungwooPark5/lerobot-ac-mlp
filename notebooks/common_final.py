"""common_final.py — 논문(KBS) 최종 실험 드라이버.  <repo>/notebooks/ 안에서 self-contained.

  학습  : 150k step · seed 4개 · lr sweep 없음 (전 모델 고정 lr).
  eval  : 150k 체크포인트 1개를 5회 반복(rep 마다 env seed 변경) → mean±std.
  모델  : baseline 6 (act·act_te·diffusion·smolvla·acm2·acm) + ours(carry+BiMamba+MOSAIC).

같은 폴더의 common_v23.py(커맨드 빌더/런처/집계) + smooth_metrics.py(jerk·SPARC) 만 필요.
외부 폴더(~/lerobot_project/v23, v9) 의존 없음.

경로 오버라이드(환경변수): LEROBOT_REPO / LEROBOT_PYTHON / LEROBOT_OUTPUT — common_v23 참고.
"""
import os
import sys
from pathlib import Path

# ── HuggingFaceVLA/libero 데이터셋은 Xet 저장 → hf-xet 가 meta/episodes/*.parquet 를 로컬에 안 풀어
#    'does not contain any parquet file' 로 학습이 죽음. Xet 끄면 일반 HTTP 로 받아 정상.
#    (이 kernel 에서 launch 되는 train subprocess 가 이 env 를 상속함.)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# ── 학습 subprocess 가 Jupyter 커널의 inline matplotlib 백엔드를 상속하면 libero(matplotlib import)가
#    'invalid backend' 로 죽음. 커널 env 를 headless(Agg)로 덮어써서 상속 자체를 막는다(override).
os.environ["MPLBACKEND"] = "Agg"

_HERE = Path(__file__).resolve().parent          # <repo>/notebooks
if str(_HERE) not in sys.path:                   # 노트북을 어디서 열든 옆 모듈을 import 할 수 있게
    sys.path.insert(0, str(_HERE))

from common_v23 import *        # noqa: F401,F403  make_train_cmd/launch_*/eval_curve/report helpers/TASKS/...
import common_v23 as v23        # noqa: F401

# ── 출력은 outputs/final 로 (v23 와 분리) ──────────────────────────────────────
# v23 의 경로 헬퍼(train_dir/eval_clean_dir/status/curve/report/_logs)는 모두 common_v23 모듈의
# OUTPUT_BASE 를 호출시점에 참조하므로, 그 전역을 재지정하면 train/eval/log/report 전부 outputs/final 로 감.
# LEROBOT_OUTPUT 로 위치 변경 가능 (기본 ~/lerobot_project/outputs). ⚠️ 레포 안에 쓰지 말 것(git 오염).
OUTPUT_BASE = v23.OUTPUT_BASE.parent / "final"
v23.OUTPUT_BASE = OUTPUT_BASE

# ── robosuite(LIBERO) EGL 디바이스 ────────────────────────────────────────────
# robosuite 는 MUJOCO_EGL_DEVICE_ID 가 CUDA_VISIBLE_DEVICES 안의 값이어야 assert 통과.
# v23 _render_prefix 는 MUJOCO_EGL_DEVICE_ID=0 고정 → gpu_id!=0 잡에서 AssertionError.
# make_train_cmd/make_eval_cmd 맨 앞에 MUJOCO_EGL_DEVICE_ID={gpu_id} 주입(render_prefix 뒤에 와서 0을 덮어씀).
# + MPLBACKEND=Agg : 학습 subprocess 가 Jupyter 커널의 inline matplotlib 백엔드를 상속하면
#   libero(matplotlib.cm import) 가 'invalid backend' 로 죽음 → Agg(headless)로 덮어씀.
# (common_final 을 import 한 프로세스에만 적용 → 은지님 v23/dro 런은 영향 없음.)
_EGL = lambda g: f"MPLBACKEND=Agg MUJOCO_EGL_DEVICE_ID={g} "

# LIBERO eval: task 가 10개라 동시 env 수 = eval.batch_size × 10. eval env 는 학습 시작 전에 전부 미리 생성돼
# 학습 내내 GPU VRAM(MuJoCo 렌더 컨텍스트)을 점유함 → 정책 로드 시 VRAM 초과 = CUDA OOM(진짜 원인).
# (EGL_NOT_INITIALIZED 홍수는 OOM 으로 죽은 뒤 env teardown 노이즈일 뿐.) → batch = VRAM knob. 4 = 동시 40 env.
# seed1~3(EVAL_FREQ=0)은 eval env 미생성이라 OOM 안 남. 50 env 도 OOM 이면 3 → 2 로. GPU 공유(overcommit)도 확인.
LIBERO_EVAL_BATCH = 5
def _libero_eval(cmd, task):
    return cmd + f" --eval.batch_size={LIBERO_EVAL_BATCH}" if str(task).startswith("libero") else cmd

_orig_make_train_cmd = v23.make_train_cmd
def make_train_cmd(tag, seed=v23.PRIMARY_SEED, task=v23.PRIMARY_TASK, gpu_id=0, **kw):
    return _EGL(gpu_id) + _libero_eval(_orig_make_train_cmd(tag, seed, task, gpu_id, **kw), task)
v23.make_train_cmd = make_train_cmd

_orig_make_eval_cmd = v23.make_eval_cmd
def make_eval_cmd(tag, seed=v23.PRIMARY_SEED, task=v23.PRIMARY_TASK, gpu_id=0, **kw):
    return _EGL(gpu_id) + _libero_eval(_orig_make_eval_cmd(tag, seed, task, gpu_id, **kw), task)
v23.make_eval_cmd = make_eval_cmd

# ══════════════════════════════════════════════════════════════════════════════
# 학습 / eval 프로토콜 (2026-07-12 확정)
#   학습  : **150k step**, **seed 4개**, lr sweep 없음(전 모델 고정 lr — v23.LR=1e-5).
#   eval  : **150k 체크포인트 1개**를 **5회 반복**(rep 마다 env seed 를 바꿔 재평가).
#           → 같은 정책의 평가 분산(env 랜덤성)을 seed 분산과 분리해서 볼 수 있음.
# ══════════════════════════════════════════════════════════════════════════════
v23.STEPS = 150_000             # make_train_cmd 가 --steps 로 읽음 (150k 까지만)
STEPS = v23.STEPS
CKPT_STEP = 150_000             # eval 대상 체크포인트 (고정)

# 학습중 eval: 50k~150k (10k마다). SR-vs-step 곡선용. 끄려면 v23.EVAL_FREQ = 0.
v23.EVAL_START = 50_000
v23.TRAIN_EVAL_N = 50           # 학습중 eval 에피소드(가볍게). 논문 수치는 아래 반복 eval 로.
EVAL_START = v23.EVAL_START

# 반복 eval (150k ckpt 재평가)
EVAL_REPEATS = 5                # 반복 횟수
EVAL_SEED0 = 1000               # rep r → --seed = EVAL_SEED0 + 100*r  (env 초기상태가 rep 마다 달라짐)
EVAL_N_EP = 50                  # 반복 1회당 에피소드 수 (총 = 5 × 50 = 250/seed)

# ══════════════════════════════════════════════════════════════════════════════
# 논문 모델 세트 (2026-07-12 기준)
#   백본 피벗: acm2(Mamba-2) 보다 acm(Mamba-1) 이 SR 이 좋아 우리 스택을 acm 으로 포팅.
#   → ours = acm + carry(SSCP) + BiMamba + MOSAIC
#   lr: 전 모델 1e-5 통일 (diffusion/smolvla 만 각자 논문 기본값 1e-4 — 공정성).
# ══════════════════════════════════════════════════════════════════════════════
# act_te = ACT + temporal ensembling (ACT 자체 smoothing). TE는 추론 전용(학습 불변)
#   → act 체크포인트에 eval-time 적용(재학습 불필요). coeff=0.01(ACT 원본), n_action_steps=1.
v23.MODEL_CONFIGS["act_te"] = ("act", v23.LR, 100, ["--policy.temporal_ensemble_coeff=0.01"], False)
v23.MODEL_DIR_NAMES["act_te"] = "act_te"
v23.MODEL_LABELS["act_te"] = "ACT+TE (temporal ensemble)"

# 메인 표 = baseline 5 + ours
BASELINE_TAGS = ["act", "act_te", "diffusion", "smolvla", "acm2", "acm"]
OURS = "ours"                                  # acm + carry + BiMamba + MOSAIC
FINAL_TAGS = BASELINE_TAGS + [OURS]
FINAL_LABELS = {
    "act":       "ACT (Transformer dec)",
    "act_te":    "ACT + TE (ACT 자체 smoothing)",
    "diffusion": "Diffusion Policy",
    "smolvla":   "SmolVLA",
    "acm2":      "ACM2 (Mamba-2 dec, no carry)",
    "acm":       "ACM (Mamba-1 dec, no carry)",   # ours 의 백본 = 우리 기여를 뺀 바닥
    "ours":      "Ours (carry + BiMamba + MOSAIC) ★",
}
# 학습 대상 (act_te 는 학습 X — act ckpt 에 eval-time TE)
TRAIN_TAGS = ["act", "diffusion", "smolvla", "acm2", "acm", "ours"]

# ── 그룹 (노트북 1개 = 그룹 1개) ──────────────────────────────────────────────
#   OURS  = 우리 모델 + 직접 대조군(acm: 같은 백본 carry off).
#           "plain Mamba 대비 개선"이 헤드라인 주장 → acm 없이는 그 수치를 못 씀.
GROUP_OURS     = ["acm", OURS]                              # 01/02  (2 x 4seed = 8잡)
GROUP_BASELINE = ["act", "diffusion", "smolvla", "acm2"]    # 03/04  (16잡) + act_te(eval only)
GROUP_ABLATION = ["acm_carry", "acm_bimamba", "acm_s7"]     # 05/06  (12잡)
TRAIN_ALL      = TRAIN_TAGS                                 # 전부   (24잡)

# ── Ablation 사다리 (논문 필수: 기여 3개를 각각 분리) ──────────────────────────
#   acm(바닥) → +carry → +BiMamba → +MOSAIC(=ours).  acm_s7 = MOSAIC 을 BiMamba 없이(직교성 확인)
ABLATION = ["acm", "acm_carry", "acm_bimamba", "acm_s7", "ours"]
ABLATION_TRAIN = ["acm_carry", "acm_bimamba", "acm_s7"]   # FINAL_TAGS 밖의 추가 학습분 (= GROUP_ABLATION)
DECISION_PAIR = ("acm_bimamba", "ours")       # carry+BiMamba  vs  +MOSAIC (떨림 제거가 SR 손해 없는가)

FINAL_SEEDS = [0, 1, 2, 3]                   # = MAIN_SEEDS (학습 seed 4개)

# ── Sim task 축: transfer(짧은 앵커) + LIBERO-10(긴 horizon) ──────────────────
# LIBERO는 레포 env(libero)로 지원. 공식 데이터셋 = HuggingFaceVLA/libero (전 suite 포함),
# eval suite는 env.task로 선택. 문서: huggingface.co/docs/lerobot/libero
# ⚠️ LIBERO **eval**은 LIBERO 시뮬(robosuite/libero) 설치 필요(mamba_ssm처럼 클러스터 의존성).
#    train은 데이터셋만 있으면 됨. fps=30(aloha 50과 다름 → jerk/SPARC는 fps_of() 사용).
FPS_BY_TASK = {"transfer": 50.0, "insertion": 50.0}
# LIBERO suite = horizon 축(같은 벤치마크·로봇 → 한 그림). 전부 HuggingFaceVLA/libero 데이터셋, env.task만 다름.
#   길이(스텝): libero_spatial 280 < libero_goal 300 < libero_10(LONG) 520
for _s in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
    v23.TASKS[_s] = ("HuggingFaceVLA/libero", "libero", _s)
    FPS_BY_TASK[_s] = 30.0

# ── 방향(2026-07-07 완전최종): 메인 = 실제로봇(so-101) + aloha insertion / 보조 = LIBERO_10(표만) ──
MAIN_SIM     = "insertion"       # aloha AlohaInsertion-v0 (300스텝, 집기+삽입 = aloha 중 가장 긴 horizon), fps50
SUPPORT_SIM  = "libero_10"       # 보조(표만, 추가분석 X — 교수님). LIBERO-LONG 520스텝, fps30
MAIN_SEEDS   = [0, 1, 2, 3]      # 학습 seed 4개 (해준 0-1 / 은지 2-3 로 나눠 pooled 가능)
SUPPORT_SEEDS = [0, 1, 2]        # 보조 3 seed (지금 돌던 것 살림)
# ⚠️ transfer(300)는 과거 ACT한테 진 지점 → insertion에서 우세 확인 필수(안 되면 real-robot 메인 캐리 + LIBERO 강조).

PRIMARY_SIM = SUPPORT_SIM                        # 하위호환: 기존 LIBERO 노트북(01_train/02b/03c)이 참조 = libero_10
LIBERO_HORIZON = ["libero_spatial", "libero_goal", "libero_10"]   # 짧→긴 (보조 horizon, 여유 시만)
SIM_TASKS = ["insertion", "libero_10"]          # 메인 aloha insertion + 보조 libero_10


def fps_of(task):
    return FPS_BY_TASK.get(task, v23.FPS)

# 그룹 색 (report용): baseline=검정/회색 계열, mamba 사다리=파랑→초록(ours)
COLOR = {"act": "#000000", "act_te": "#9467bd", "diffusion": "#d62728", "smolvla": "#ff7f0e",
         "acm2": "#bbbbbb", "acm": "#888888",
         "acm_carry": "#9ecae1", "acm_s7": "#6baed6", "acm_bimamba": "#1f77b4", "ours": "#2ca02c"}


def train_jobs(seeds, tags=None, task=None):
    """launch_training_live 용 (tag, seed, task) job 리스트. 기본 tags=TRAIN_TAGS(act_te 제외), task=MAIN_SIM."""
    tags = tags or TRAIN_TAGS          # act_te 는 학습 안 함(act ckpt에 eval-time TE)
    task = task or MAIN_SIM
    return [(t, s, task) for s in seeds for t in tags]


def act_te_eval_cmd(seed, task, gpu_id=0, n_episodes=None, select="last"):
    """act_te = act 체크포인트에 temporal ensembling 을 eval-time 적용(재학습 X).

    act ckpt 로드 + TE 오버라이드(coeff=0.01, n_action_steps=1), 결과는 act_te eval 디렉토리에 기록
    → 03_report/03c 가 자동으로 act_te 로 집계. TE=ACT 자체 smoothing → 매끄러움 공정 비교.
    """
    out = v23.eval_clean_dir("act_te", seed, task)
    return make_eval_cmd("act", seed=seed, task=task, gpu_id=gpu_id, n_episodes=n_episodes,
                         select=select, out_dir=out,
                         extra_policy=["--policy.temporal_ensemble_coeff=0.01", "--policy.n_action_steps=1"])


# ══════════════════════════════════════════════════════════════════════════════
# 반복 eval — 150k 체크포인트 1개를 EVAL_REPEATS 회 재평가 (rep 마다 env seed 변경)
#   · 외란 X (run_perturb 아님). 표준 lerobot_eval 직접 호출.
#   · rep 은 **학습 seed 와 다른 축**: 학습 seed = 모델 분산 / rep = 평가(env) 분산.
#   · action(.pt) 은 fork eval 루프가 RECORD_DIR 로 기록 → jerk 측정 가능.
# ══════════════════════════════════════════════════════════════════════════════
TE_FLAGS = ["--policy.temporal_ensemble_coeff=0.01", "--policy.n_action_steps=1"]


def eval_rep_dir(tag, seed, task, rep):
    """rep 별 eval 출력 디렉토리. rep=None 이면 기존 단일 eval 디렉토리."""
    base = v23.eval_clean_dir(tag, seed, task)
    return base if rep is None else base / f"rep{rep}"


def repeat_eval_cmd(tag, seed, rep, task=None, gpu_id=0, n_episodes=None, step=None):
    """150k(=CKPT_STEP) 체크포인트를 rep 번째 env seed 로 평가하는 커맨드.

    tag='act_te' 는 **act 체크포인트**를 재사용하고 TE 플래그만 eval-time 으로 얹음(재학습 X).
    """
    task = task or MAIN_SIM
    step = CKPT_STEP if step is None else step
    n = n_episodes or EVAL_N_EP
    is_te = tag == "act_te"
    src = "act" if is_te else tag                      # ckpt 를 가져올 학습 태그

    cd = v23.best_ckpt_dir(src, seed, task, how=step)  # int → 그 step (없으면 최근접)
    if cd is None:
        raise FileNotFoundError(f"No {step:,} checkpoint for {src}/{task}/seed{seed}")
    ckpt = v23._pretrained(cd)

    out = eval_rep_dir(tag, seed, task, rep)
    out.mkdir(parents=True, exist_ok=True)
    _ds, env_type, env_task = v23.TASKS[task]
    batch = LIBERO_EVAL_BATCH if str(task).startswith("libero") else min(n, v23.EVAL_BATCH_SIZE)

    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}", f"MUJOCO_EGL_DEVICE_ID={gpu_id}", "MPLBACKEND=Agg",
        f"RECORD_DIR={out / 'actions'}",
        f"{v23.PYTHON} -m lerobot.scripts.lerobot_eval", f"--policy.path={ckpt}",
        f"--env.type={env_type}", f"--env.task={env_task}",
        f"--eval.n_episodes={n}", f"--eval.batch_size={batch}",
        f"--seed={EVAL_SEED0 + 100 * rep}",            # ★ rep 마다 다른 env 초기상태
        f"--output_dir={out}",
    ]
    if is_te:
        parts.extend(TE_FLAGS)
    return " ".join(parts)


def repeat_eval_jobs(tags=None, seeds=None, reps=None, task=None, n_episodes=None):
    """(tag, seed, rep) 전체 조합 → [(label, cmd)] . gpu_id 는 호출측에서 청크로 배정."""
    tags = tags or FINAL_TAGS
    seeds = seeds or MAIN_SEEDS
    reps = range(EVAL_REPEATS) if reps is None else reps
    jobs = []
    for s in seeds:
        for t in tags:
            for r in reps:
                jobs.append((t, s, r))
    return jobs


def rep_sr(tag, seed, task, rep):
    """rep 하나의 SR(%) — 없으면 None."""
    info = eval_rep_dir(tag, seed, task, rep) / "eval_info.json"
    if not info.exists():
        return None
    import json as _json
    pc = _json.loads(info.read_text()).get("overall", {}).get("pc_success")
    return None if pc is None else float(pc)


def sr_over_reps(tag, task=None, seeds=None, reps=None):
    """반복 eval 집계: {seed: [rep별 SR]}, 전체 평균/표준편차/n.

    반환: dict(per_seed={seed: mean}, all=[모든 rep SR], mean, std, n_runs, n_seed)
    """
    import statistics as _st
    task = task or MAIN_SIM
    seeds = seeds or MAIN_SEEDS
    reps = range(EVAL_REPEATS) if reps is None else reps
    per_seed, allv = {}, []
    for s in seeds:
        vals = [v for v in (rep_sr(tag, s, task, r) for r in reps) if v is not None]
        if vals:
            per_seed[s] = _st.fmean(vals)
            allv += vals
    return {
        "per_seed": per_seed, "all": allv,
        "mean": _st.fmean(allv) if allv else None,
        "std": _st.pstdev(allv) if len(allv) > 1 else 0.0,
        "n_runs": len(allv), "n_seed": len(per_seed),
    }


# ── 그룹 실행 (노트북에서 한 줄) ──────────────────────────────────────────────
def run_training(tags, seeds=None, task=None, ngpu=8):
    """tags x seeds 를 ngpu 청크로 학습(각 청크 끝날 때까지 대기). resume 자동."""
    seeds = seeds or MAIN_SEEDS
    task = task or MAIN_SIM
    jobs = train_jobs(seeds, tags=tags, task=task)
    print(f"학습 {tags} x seed{seeds} = {len(jobs)}잡  ({STEPS:,} step, lr 고정)")
    for i in range(0, len(jobs), ngpu):
        chunk = jobs[i:i + ngpu]
        print(f"\n===== 청크 {i // ngpu + 1}/{-(-len(jobs) // ngpu)} ({len(chunk)}잡) =====")
        for j in chunk:
            print("  ", j)
        v23.launch_training_live(chunk)
    print("\n학습 완료:", tags, seeds)
    return jobs


def run_repeat_evals(tags, seeds=None, reps=None, task=None, ngpu=8, n_episodes=None):
    """150k 체크포인트 x rep 반복 eval. 이미 끝난 run 은 skip → 재실행 안전."""
    seeds = seeds or MAIN_SEEDS
    reps = list(range(EVAL_REPEATS)) if reps is None else list(reps)
    task = task or MAIN_SIM
    n = n_episodes or EVAL_N_EP

    runs = [(t, s, r) for s in seeds for t in tags for r in reps]
    todo = [x for x in runs if rep_sr(x[0], x[1], task, x[2]) is None]
    print(f"eval {tags} | {CKPT_STEP:,} ckpt x {len(reps)}rep x {len(seeds)}seed x {n}ep")
    print(f"  전체 {len(runs)} run / 남은 {len(todo)} (완료 {len(runs) - len(todo)})")

    for i in range(0, len(todo), ngpu):
        chunk = todo[i:i + ngpu]
        labeled = []
        for g, (t, s, r) in enumerate(chunk):
            try:
                labeled.append((f"{t}/seed{s}/rep{r}",
                                repeat_eval_cmd(t, s, r, task=task, gpu_id=g, n_episodes=n)))
            except FileNotFoundError as e:
                print("  skip:", e)
        if labeled:
            print(f"\n===== eval 청크 {i // ngpu + 1}/{-(-len(todo) // ngpu)} ({len(labeled)} run) =====")
            v23.launch_cmds_live(labeled)
    print("\neval 완료:", tags)


def print_ckpt_status(tags, seeds=None, task=None, step=None):
    """150k 체크포인트 존재 확인 (eval 전 필수 체크)."""
    seeds = seeds or MAIN_SEEDS
    task = task or MAIN_SIM
    step = CKPT_STEP if step is None else step
    missing = []
    print(f"{step:,} 체크포인트 ({task}):")
    for s in seeds:
        row = []
        for t in tags:
            cd = v23.best_ckpt_dir(t, s, task, how=step)
            if cd is None:
                row.append(f"{t}:X")
                missing.append((t, s))
            else:
                got = int(cd.name)
                row.append(f"{t}:{got // 1000}k" + ("" if got == step else "(!)"))
        print(f"  seed{s}  " + "  ".join(row))
    if missing:
        print("\n⚠ 체크포인트 없음:", missing, "→ 학습 먼저")
    return not missing


def sr_table(tags, seeds=None, reps=None, task=None, n_episodes=None, csv_path=None):
    """반복 eval 집계 표. mean±std(= seed x rep run) + pooled Wilson CI. rows 반환."""
    import csv as _csv
    seeds = seeds or MAIN_SEEDS
    reps = list(range(EVAL_REPEATS)) if reps is None else list(reps)
    task = task or MAIN_SIM
    n = n_episodes or EVAL_N_EP

    rows = []
    for t in tags:
        agg = sr_over_reps(t, task=task, seeds=seeds, reps=reps)
        if agg["mean"] is None:
            print(f"  {t:<12} (결과 없음)")
            continue
        n_ep = agg["n_runs"] * n
        lo, hi = v23.wilson_ci(int(round(agg["mean"] / 100 * n_ep)), n_ep)
        rows.append({"tag": t, "model": v23.MODEL_LABELS.get(t, t),
                     "SR_mean": round(agg["mean"], 2), "SR_std": round(agg["std"], 2),
                     "n_run": agg["n_runs"], "n_episodes": n_ep,
                     "CI_lo": round(lo * 100, 1), "CI_hi": round(hi * 100, 1),
                     "per_seed": {s: round(v, 1) for s, v in agg["per_seed"].items()}})
    if not rows:
        print("결과 없음 — eval 먼저")
        return rows

    hdr = f"{'MODEL':<34}{'SR (mean±std)':>18}{'95% CI':>16}{'runs':>6}   per-seed"
    print(hdr)
    print("-" * (len(hdr) + 8))
    for r in rows:
        sr = f"{r['SR_mean']:.1f} ± {r['SR_std']:.1f}"
        ci = f"[{r['CI_lo']:.1f}, {r['CI_hi']:.1f}]"
        print(f"{r['model']:<34}{sr:>18}{ci:>16}{r['n_run']:>6}   {r['per_seed']}")

    if csv_path:
        p = Path(csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("\n저장:", p)
    return rows


def action_trajs(tag, seed, task=None, reps=None):
    """action 궤적(.pt) 수집 — 반복 eval(rep*/actions) 우선, 없으면 단일 eval, 없으면 학습중 eval.

    04_report_jerk 가 이걸 통해 rep 5개를 전부 pool 해서 jerk 통계를 냄.
    """
    task = task or MAIN_SIM
    reps = range(EVAL_REPEATS) if reps is None else reps
    trajs = []
    for r in reps:                                   # 1) 반복 eval
        trajs += v23._load_action_trajs(eval_rep_dir(tag, seed, task, r) / "actions") or []
    if trajs:
        return trajs
    trajs = v23._load_action_trajs(v23.eval_clean_dir(tag, seed, task) / "actions") or []
    if trajs:
        return trajs
    edir = v23.train_dir(tag, seed, task) / "eval"    # 3) 학습중 eval (videos_step_*/action_logs)
    best = None
    if edir.is_dir():
        steps = [int(d.name.rsplit("_", 1)[-1]) for d in edir.glob("videos_step_*")
                 if d.name.rsplit("_", 1)[-1].isdigit()]
        if steps:
            best = edir / f"videos_step_{max(steps)}" / "action_logs"
    return (v23._load_action_trajs(best) or []) if best else []


def libero_suite_eval_cmd(tag, seed, suite, gpu_id=0, n_episodes=None, select="best"):
    """libero_10로 학습한 체크포인트를 다른 LIBERO suite로 평가(horizon 그림용).

    한 모델을 spatial/goal/10에 평가 → '짧→긴' SR 곡선. ⚠️ LIBERO 시뮬 필요.
    """
    cd = v23.best_ckpt_dir(tag, seed, PRIMARY_SIM, how=select)
    if cd is None:
        raise FileNotFoundError(f"No {PRIMARY_SIM} checkpoint for {tag} seed{seed}")
    ckpt = v23._pretrained(cd)
    out = v23.eval_clean_dir(tag, seed, PRIMARY_SIM) / f"suite_{suite}"
    out.mkdir(parents=True, exist_ok=True)
    n = n_episodes or v23.EVAL_N_EPISODES
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}", f"MUJOCO_EGL_DEVICE_ID={gpu_id}", "MPLBACKEND=Agg",
        f"RECORD_DIR={out / 'actions'}",
        f"{v23.PYTHON} -m lerobot.scripts.lerobot_eval", f"--policy.path={ckpt}",
        "--env.type=libero", f"--env.task={suite}",
        f"--eval.n_episodes={n}", f"--eval.batch_size={LIBERO_EVAL_BATCH}", f"--output_dir={out}",
    ]
    return " ".join(parts)


def libero_eval_cmd(tag, seed, gpu_id=0, n_episodes=100, step=150_000, suite=None, out_dir=None):
    """외란 없는 순수 LIBERO eval — 표준 lerobot_eval 직접 호출(run_perturb 안 씀).

    - step: 평가할 체크포인트 스텝(정수). best_ckpt_dir(how=step)=정확히 그 step(없으면 최근접).
    - action(.pt)은 fork eval 루프가 RECORD_DIR 환경변수로 기록 → jerk 측정 가능(run_perturb 불필요).
    - out = eval_clean_dir(tag,seed,suite) → 03_report의 get_eval_status/aggregate_seeds가 자동 pooled.
    """
    suite = suite or PRIMARY_SIM                      # 'libero_10'
    cd = v23.best_ckpt_dir(tag, seed, PRIMARY_SIM, how=step)   # step=int → 그 step(없으면 최근접 ckpt)
    if cd is None:
        raise FileNotFoundError(f"No checkpoint for {tag} seed{seed} ({PRIMARY_SIM})")
    ckpt = v23._pretrained(cd)
    out = Path(out_dir) if out_dir is not None else v23.eval_clean_dir(tag, seed, suite)
    out.mkdir(parents=True, exist_ok=True)
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}", f"MUJOCO_EGL_DEVICE_ID={gpu_id}", "MPLBACKEND=Agg",
        f"RECORD_DIR={out / 'actions'}",
        f"{v23.PYTHON} -m lerobot.scripts.lerobot_eval", f"--policy.path={ckpt}",
        "--env.type=libero", f"--env.task={suite}",
        f"--eval.n_episodes={n_episodes}", f"--eval.batch_size={LIBERO_EVAL_BATCH}",
        f"--output_dir={out}",
    ]
    return " ".join(parts)


def resolved_ckpt_step(tag, seed, step=150_000):
    """실제로 선택될 체크포인트 스텝(150k 존재 확인용). 없으면 None."""
    cd = v23.best_ckpt_dir(tag, seed, PRIMARY_SIM, how=step)
    return int(cd.name) if cd is not None else None


def check_tags():
    """FINAL_TAGS + ABLATION 이 전부 v23.MODEL_CONFIGS 에 있는지 확인 + 요약(정책/lr/K) 출력."""
    tags = FINAL_TAGS + [t for t in ABLATION if t not in FINAL_TAGS]
    print(f"=== 메인 표 {len(FINAL_TAGS)}모델 + ablation ({MAIN_SIM} / {len(MAIN_SEEDS)} seed) ===")
    for t in tags:
        cfg = v23.MODEL_CONFIGS.get(t)
        mark = "★" if t == OURS else (" " if t in FINAL_TAGS else "·")   # ·= ablation 전용
        if cfg is None:
            print(f"  !! {t:<12} -> MODEL_CONFIGS 에 없음")
            continue
        pol, lr, K, _extra, cp = cfg
        print(f"  OK {mark}{t:<12} lr={lr:<7} K={K:<4} pairs={str(cp):<5} {pol}")
    missing = [t for t in tags if t not in v23.MODEL_CONFIGS]
    if missing:
        print("⚠ MODEL_CONFIGS에 없음:", missing)
    return not missing
