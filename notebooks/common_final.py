"""common_final.py — 논문(KBS) 최종 실험 드라이버.  <repo>/notebooks/ 안에서 self-contained.

  학습  : 150k step · seed 4개 · lr sweep 없음 (전 모델 고정 lr).
  eval  : 150k 체크포인트 1개를 5회 반복(rep 마다 env seed 변경) → mean±std.
  모델  : baseline 6 (act·act_te·diffusion·smolvla·acm2·acm) + ours(carry+BiMamba+MOSAIC).

같은 폴더의 common_v23.py(커맨드 빌더/런처/집계) + smooth_metrics.py(jerk·SPARC) 만 필요.
외부 폴더(~/lerobot_project/v23, v9) 의존 없음.

경로 오버라이드(환경변수): LEROBOT_REPO / LEROBOT_PYTHON / LEROBOT_OUTPUT — common_v23 참고.
"""
import json
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

# ── 렌더 백엔드: EGL 고정 (학습/eval 전부) ────────────────────────────────────
# common_v23._render_prefix() 가 모든 커맨드 앞에 MUJOCO_GL=egl / PYOPENGL_PLATFORM=egl 을 붙인다.
# (각 run 은 그 뒤에 MUJOCO_EGL_DEVICE_ID={gpu_id} 를 다시 써서 자기 GPU 로 렌더 — robosuite assert 통과용)
# OSMesa 는 CPU 소프트웨어 렌더라 eval 이 몇 배로 느려짐 → RENDER_BACKEND=osmesa 를 export 하지 말 것.
os.environ.setdefault("RENDER_BACKEND", "egl")
if os.environ.get("RENDER_BACKEND") == "osmesa":
    print("⚠️ RENDER_BACKEND=osmesa — CPU 소프트웨어 렌더. eval 이 수 배 느려짐.\n"
          "   GPU(EGL) 로 돌리려면:  del os.environ['RENDER_BACKEND']  후 커널 재시작")

_HERE = Path(__file__).resolve().parent          # <repo>/notebooks
if str(_HERE) not in sys.path:                   # 노트북을 어디서 열든 옆 모듈을 import 할 수 있게
    sys.path.insert(0, str(_HERE))

import importlib as _il

import common_v23 as v23        # noqa: F401
# 노트북은 `importlib.reload(cf)` 를 부른다. 그때 common_v23 도 같이 새로 실행해야
# (a) 옛 세션에서 v23 전역에 남은 몽키패치/설정 잔재가 지워지고
# (b) 아래에서 다시 세팅하는 STEPS/EVAL_FREQ/... 가 항상 같은 출발점에서 적용된다.
# 이걸 안 하면 커널을 재시작하기 전까지 v23 의 옛 상태가 계속 살아남는다.
v23 = _il.reload(v23)
from common_v23 import *        # noqa: F401,F403,E402  make_train_cmd/launch_*/eval_curve/TASKS/...

# ── 출력은 outputs/final 로 (v23 와 분리) ──────────────────────────────────────
# v23 의 경로 헬퍼(train_dir/eval_clean_dir/status/curve/report/_logs)는 모두 common_v23 모듈의
# OUTPUT_BASE 를 호출시점에 참조하므로, 그 전역을 재지정하면 train/eval/log/report 전부 outputs/final 로 감.
# LEROBOT_OUTPUT 로 위치 변경 가능 (기본 ~/lerobot_project/outputs). ⚠️ 레포 안에 쓰지 말 것(git 오염).
OUTPUT_BASE = v23.OUTPUT_BASE.parent / "final"
v23.OUTPUT_BASE = OUTPUT_BASE

# ── EGL 디바이스 / LIBERO eval batch ─────────────────────────────────────────
# 예전엔 여기서 v23.make_train_cmd / make_eval_cmd 를 감싸 몽키패치했는데, 노트북이
# importlib.reload(cf) 를 부르면 래퍼를 다시 원본으로 잡아 **RecursionError** 가 났다.
# → 이제 그 주입은 common_v23.make_*_cmd 안(_gpu_env / _eval_batch)에서 직접 한다. 래핑 없음.
#   · MUJOCO_EGL_DEVICE_ID={gpu} : robosuite 는 렌더 디바이스가 CUDA_VISIBLE_DEVICES 안이어야 함
#   · MPLBACKEND=Agg             : subprocess 가 커널의 inline 백엔드를 상속하면 libero 가 죽음
#   · LIBERO 는 동시 env = batch × 10(task) → VRAM OOM 방지로 batch 를 작게
LIBERO_EVAL_BATCH = 5
v23.LIBERO_EVAL_BATCH = LIBERO_EVAL_BATCH

# ══════════════════════════════════════════════════════════════════════════════
# 학습 / eval 프로토콜 (2026-07-12 확정)
#   학습  : **150k step**, **seed 4개**, lr sweep 없음(전 모델 고정 lr — v23.LR=1e-5).
#   eval  : **150k 체크포인트 1개**를 **5회 반복**(rep 마다 env seed 를 바꿔 재평가).
#           → 같은 정책의 평가 분산(env 랜덤성)을 seed 분산과 분리해서 볼 수 있음.
# ══════════════════════════════════════════════════════════════════════════════
v23.STEPS = 150_000             # make_train_cmd 가 --steps 로 읽음 (150k 까지만)
STEPS = v23.STEPS
CKPT_STEP = 150_000             # eval 대상 체크포인트 (고정)
v23.SAVE_FREQ = 10_000          # 10k 마다 체크포인트 → 150k 반드시 존재

# ── 학습중 eval: OFF ──────────────────────────────────────────────────────────
# eval_freq=0 → 학습 프로세스가 sim env 를 아예 안 만든다.
#   · GPU/시간 절약 (150k 동안 eval 로 새는 시간 0)
#   · env 를 미리 다 만들며 VRAM 을 먹던 CUDA OOM 원인 자체가 사라짐
#   · 어차피 논문 수치는 150k 체크포인트 반복 eval 로만 낸다 (best-ckpt 안 고름)
# ⚠️ 대가: SR-vs-step 곡선이 안 나온다(09_report_sr 의 곡선 셀은 비어 있게 됨).
#    수렴 곡선이 필요하면 v23.EVAL_FREQ = 10_000 으로 되돌릴 것.
v23.EVAL_FREQ = 0               # ← 이 값 하나로 꺼짐 (lerobot_train: eval_freq>0 일 때만 env 생성/eval)
v23.EVAL_START = STEPS          # eval_freq=0 이라 무의미하지만 안전하게 끝값으로
# ⚠️ TRAIN_EVAL_N 은 0 으로 두면 안 됨: EvalConfig 가 batch_size(50) > n_episodes 를 에러로 막아
#    --eval.n_episodes=0 을 주는 순간 학습이 config 파싱에서 죽는다. eval_freq=0 이라 쓰이지 않는 값.
v23.TRAIN_EVAL_N = 50
EVAL_START = v23.EVAL_START

# ── 반복 eval (150k ckpt 재평가) ──────────────────────────────────────────────
EVAL_REPEATS = 5                # 반복 횟수
EVAL_SEED0 = 1000               # rep r → --seed = EVAL_SEED0 + 100*r  (env 초기상태가 rep 마다 달라짐)
EVAL_N_EP = 500                 # 반복 1회당 에피소드 (= 모델/seed 당 5 × 500 = 2,500 에피소드)

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
GROUP_OURS     = [OURS]                                     # 01/02  우리 모델      (4잡)
GROUP_ACM      = ["acm"]                                    # 03/04  ★직접 대조군   (4잡)
GROUP_BASELINE = ["act", "diffusion", "smolvla", "acm2"]    # 05/06  외부 baseline (16잡) + act_te(eval only)
GROUP_ABLATION = ["acm_carry", "acm_bimamba", "acm_s7"]     # 07/08  사다리 나머지 (12잡)
TRAIN_ALL      = TRAIN_TAGS                                 # 전부                 (24잡)
# ★ acm = 같은 백본에서 carry 를 끈 것. "plain Mamba 대비 개선"이 헤드라인 주장이라
#   이 수치가 없으면 그 주장에 분모가 없다 → ours 다음으로 우선순위 높음.

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
def n_gpu():
    """이 노드에서 실제로 보이는 GPU 수. 노트북의 NGPU 는 이 값을 쓴다(하드코딩 금지)."""
    try:
        return len(v23.available_gpus())
    except Exception as e:                      # CUDA 없음 등 — 리포트 전용 커널
        print("⚠️ GPU 감지 실패:", e)
        return 1


def prefetch_dataset(task=None, force=False):
    """데이터셋을 **한 프로세스로 먼저** 내려받아 HF 캐시를 데운다.

    ⚠️ 이걸 안 하면: 학습 잡 N개가 각자 snapshot_download 를 같은 캐시 폴더에 동시에 호출한다.
       먼저 시작한 잡이 meta/info.json·meta/episodes/*.parquet 를 아직 다 못 풀었는데 다른 잡이
       그 폴더를 읽어버려서
         FileNotFoundError: ... meta/info.json
         FileNotFoundError: Provided directory does not contain any parquet file: .../meta/episodes
       로 죽는다. 다운로드를 완주한 한 잡만 살아남음(= seed 하나만 성공하는 증상).
    캐시가 이미 차 있으면 즉시 리턴(초 단위).
    """
    import subprocess
    task = task or MAIN_SIM
    repo_id = v23.TASKS[task][0]
    # ⚠️ shell 을 거치지 않고 argv 로 넘긴다. shell=True + 따옴표로 감싸면 여러 줄 코드의
    #    개행이 리터럴 '\n' 으로 전달돼 SyntaxError 가 난다.
    py = (
        "from lerobot.datasets.lerobot_dataset import LeRobotDataset\n"
        f"ds = LeRobotDataset({repo_id!r})\n"
        f"print('PREFETCH_OK', {repo_id!r}, 'episodes', ds.num_episodes, 'frames', ds.num_frames)\n"
    )
    env = dict(os.environ, HF_HUB_DISABLE_XET="1", PYTHONPATH=str(v23.SRC_DIR), MPLBACKEND="Agg")
    print(f"[prefetch] {repo_id} — 캐시 데우는 중 (첫 실행은 몇 분)…")
    r = subprocess.run([v23.PYTHON, "-c", py], env=env, capture_output=True, text=True)
    ok = "PREFETCH_OK" in r.stdout
    if ok:
        print("[prefetch]", [l for l in r.stdout.splitlines() if "PREFETCH_OK" in l][0])
    else:
        print("[prefetch] ⚠️ 실패 — 아래 로그 확인 (이 상태로 병렬 학습을 띄우면 전부 죽는다)")
        print((r.stdout or "")[-800:])
        print((r.stderr or "")[-1500:])
        print("\n터미널에서 직접 받아보려면:")
        print(f"  HF_HUB_DISABLE_XET=1 PYTHONPATH={v23.SRC_DIR} {v23.PYTHON} - <<'PY'\n{py}PY")
    return ok


def run_training(tags, seeds=None, task=None, ngpu=None, prefetch=True):
    """tags x seeds 를 ngpu 청크로 학습(각 청크 끝날 때까지 대기). resume 자동.

    ngpu=None → 이 노드의 실제 GPU 수(n_gpu())를 씀.
    prefetch=True → 병렬 실행 전에 데이터셋을 한 번 받아둔다(동시 다운로드 레이스 방지).
    """
    seeds = seeds or MAIN_SEEDS
    task = task or MAIN_SIM
    ngpu = ngpu or n_gpu()
    if prefetch and not prefetch_dataset(task):
        raise RuntimeError("데이터셋 prefetch 실패 → 병렬 학습을 띄우면 전부 죽는다. 위 로그 확인.")
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


def run_repeat_evals(tags, seeds=None, reps=None, task=None, ngpu=None, n_episodes=None):
    """150k 체크포인트 x rep 반복 eval. 이미 끝난 run 은 skip → 재실행 안전.

    ngpu=None → 이 노드의 실제 GPU 수(n_gpu())를 씀.
    """
    ngpu = ngpu or n_gpu()
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
