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

# bimamba_te = BiMamba + temporal ensembling. act_te 와 같은 원리(추론 전용 TE) — bimamba ckpt 재사용.
#   실험1 대조군: "우리 overlap(MOSAIC) vs 기존 smoothing(TE)" 를 BiMamba 백본에서도 비교.
#   *_te 태그는 아래 eval 로직이 자동으로 src=tag[:-3] 체크포인트 + TE_FLAGS 로 처리.
v23.MODEL_CONFIGS["bimamba_te"] = ("acm2_sscp_literal_bimamba", v23.LR, 100,
                                   ["--policy.temporal_ensemble_coeff=0.01"], False)
v23.MODEL_DIR_NAMES["bimamba_te"] = "bimamba_te"
v23.MODEL_LABELS["bimamba_te"] = "BiMamba+TE (temporal ensemble)"

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
GROUP_ABLATION = ["acm_carry", "acm_bimamba", "acm_mosaic"]     # 07/08  사다리 나머지 (12잡)
TRAIN_ALL      = TRAIN_TAGS                                 # 전부                 (24잡)
# ★ acm = 같은 백본에서 carry 를 끈 것. "plain Mamba 대비 개선"이 헤드라인 주장이라
#   이 수치가 없으면 그 주장에 분모가 없다 → ours 다음으로 우선순위 높음.

# ── Ablation 사다리 (논문 필수: 기여 3개를 각각 분리) ──────────────────────────
#   acm(바닥) → +carry → +BiMamba → +MOSAIC(=ours).  acm_mosaic = MOSAIC 을 BiMamba 없이(직교성 확인)
ABLATION = ["acm", "acm_carry", "acm_bimamba", "acm_mosaic", "ours"]
ABLATION_TRAIN = ["acm_carry", "acm_bimamba", "acm_mosaic"]   # FINAL_TAGS 밖의 추가 학습분 (= GROUP_ABLATION)
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
# ── sim task 축 (horizon: 짧음 → 긺) ─────────────────────────────────────────
#   Intro 주장 = "짧은 task 는 ACT 와 대등, horizon 이 길어질수록 우리가 앞선다."
#   → 그 주장을 뒷받침하려면 **짧은 쪽 데이터포인트(transfer)도 있어야** 한다.
SHORT_SIM    = "transfer"        # AlohaTransferCube-v0 — 짧은 앵커. 과거 ACT 가 이겼던 판.
MAIN_SIM     = "insertion"       # AlohaInsertion-v0 — 집기+삽입 2단계, 더 긴/어려운 horizon (메인)
SUPPORT_SIM  = "libero_10"       # 보조(표만, 추가분석 X — 교수님). LIBERO-LONG 520스텝, fps30
MAIN_SEEDS   = [0, 1, 2, 3]      # 학습 seed 4개 (해준 0-1 / 은지 2-3 로 나눠 pooled 가능)
SUPPORT_SEEDS = [0, 1, 2]        # 보조 3 seed (지금 돌던 것 살림)
# ⚠️ transfer(300)는 과거 ACT한테 진 지점 → insertion에서 우세 확인 필수(안 되면 real-robot 메인 캐리 + LIBERO 강조).

PRIMARY_SIM = SUPPORT_SIM                        # 하위호환: 기존 LIBERO 노트북(01_train/02b/03c)이 참조 = libero_10
LIBERO_HORIZON = ["libero_spatial", "libero_goal", "libero_10"]   # 짧→긴 (보조 horizon, 여유 시만)
SIM_TASKS = [SHORT_SIM, MAIN_SIM, SUPPORT_SIM]  # horizon 축: transfer(짧) → insertion → libero_10(긺)


def fps_of(task):
    return FPS_BY_TASK.get(task, v23.FPS)

# 그룹 색 (report용): baseline=검정/회색 계열, mamba 사다리=파랑→초록(ours)
COLOR = {"act": "#000000", "act_te": "#9467bd", "diffusion": "#d62728", "smolvla": "#ff7f0e",
         "acm2": "#bbbbbb", "acm": "#888888",
         "acm_carry": "#9ecae1", "acm_mosaic": "#6baed6", "acm_bimamba": "#1f77b4", "ours": "#2ca02c"}


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
# 실험1·2 — overlap 재해석 (재학습 X). overlap-add 는 추론 전용(파라미터 0개 추가)이라,
#   base 체크포인트를 그대로 불러와 policy 클래스만 overlap 계열로 바꾸면 된다. base ckpt 를
#   복사 + config.json 의 type/sscp_overlap 만 패치해서 out_tag 의 정상 학습 경로에 놓으면
#   기존 eval 기계(best_ckpt_dir / run_libero_eval_jobs)가 그대로 평가한다.
#     act_overlap  ← act        (carry 없음; ACT 는 stateless → overlap 만)
#     acm2_overlap ← acm2       (carry off + overlap  = "overlap without carry" ablation)
#     acm2_mosaic  ← acm2_carry (carry on  + overlap  = MOSAIC; ≡ mosaic_infer)
# ══════════════════════════════════════════════════════════════════════════════
OVERLAP_SOURCE = {"act_overlap": "act", "acm2_overlap": "acm2", "acm2_mosaic": "acm2_carry"}


def materialize_overlap_ckpt(out_tag, seed, task, how=None, overlap=None, window="hann"):
    """out_tag 의 base 체크포인트를 복사 + config.json 패치해 overlap 클래스 ckpt 로 재해석.

    반환: 만들어진 pretrained_model 경로 (base 없으면 None → 호출측이 스킵/경고).
    재실행 안전(idempotent): 이미 out_tag type + sscp_overlap 이면 그대로 둔다.
    overlap 정책은 파라미터 0개 추가 subclass → base 가중치를 그대로 load(strict) 하므로 안전.
    (혹시 키가 어긋나면 eval 이 로드 단계에서 곧바로 에러 → 조용한 오염 없음.)
    """
    import json
    import shutil

    how = CKPT_STEP if how is None else how
    overlap = v23._OV if overlap is None else int(overlap)
    base = OVERLAP_SOURCE[out_tag]
    ptype = v23.MODEL_CONFIGS[out_tag][0]        # overlap 정책 타입 (예: acm2_sscp_literal_smooth_overlap)

    bcd = v23.best_ckpt_dir(base, seed, task, how=how)
    if bcd is None:
        return None                              # base 체크포인트 없음
    src_pm = v23._pretrained(bcd)
    dst = v23.train_dir(out_tag, seed, task) / "checkpoints" / bcd.name / "pretrained_model"

    if not (dst / "config.json").exists():       # 아직 복사 안 됨 → base 를 복사
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_pm, dst, dirs_exist_ok=True)

    cfgp = dst / "config.json"                   # config.json 패치 (idempotent)
    cfg = json.loads(cfgp.read_text())
    if cfg.get("type") != ptype or "sscp_overlap" not in cfg:
        cfg["type"] = ptype
        cfg["sscp_overlap"] = overlap
        cfg["sscp_overlap_window"] = window
        cfg["sscp_overlap_train_weight"] = 0.0
        cfgp.write_text(json.dumps(cfg, indent=2))
    return dst


def prepare_overlap_ckpts(tags, seeds, task):
    """overlap 태그들(OVERLAP_SOURCE 에 있는 것)을 base 로부터 재해석. [(tag,seed,경로|None)] 반환."""
    out = []
    for t in tags:
        if t not in OVERLAP_SOURCE:
            continue
        for s in seeds:
            pm = materialize_overlap_ckpt(t, s, task)
            src = OVERLAP_SOURCE[t]
            if pm is None:
                print(f"  ⚠️ {t}/seed{s}: base '{src}' 체크포인트 없음 → 생략 (먼저 {src} 확보 필요)")
            else:
                print(f"  ✓ {t}/seed{s} ← {src} 재해석: {pm}")
            out.append((t, s, pm))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 반복 eval — 150k 체크포인트 1개를 EVAL_REPEATS 회 재평가 (rep 마다 env seed 변경)
#   · 외란 X (run_perturb 아님). 표준 lerobot_eval 직접 호출.
#   · rep 은 **학습 seed 와 다른 축**: 학습 seed = 모델 분산 / rep = 평가(env) 분산.
#   · action(.pt) 은 fork eval 루프가 RECORD_DIR 로 기록 → jerk 측정 가능.
# ══════════════════════════════════════════════════════════════════════════════
TE_FLAGS = ["--policy.temporal_ensemble_coeff=0.01", "--policy.n_action_steps=1"]


def eval_rep_dir(tag, seed, task, rep, step=None):
    """rep 별 eval 출력 디렉토리.

    step=None 또는 CKPT_STEP(150k) → `rep{r}`  (기존 결과 경로 그대로)
    그 외 step(예: 100k)          → `{step//1000}k_rep{r}`  (150k 결과를 덮어쓰지 않게 분리)
    """
    base = v23.eval_clean_dir(tag, seed, task)
    if rep is None:
        return base
    if step is None or int(step) == CKPT_STEP:
        return base / f"rep{rep}"
    return base / f"{int(step) // 1000}k_rep{rep}"


def repeat_eval_cmd(tag, seed, rep, task=None, gpu_id=0, n_episodes=None, step=None, n_videos=0):
    """150k(=CKPT_STEP) 체크포인트를 rep 번째 env seed 로 평가하는 커맨드.

    tag='act_te' 는 **act 체크포인트**를 재사용하고 TE 플래그만 eval-time 으로 얹음(재학습 X).
    n_videos>0 이면 그 수만큼 mp4 를 <out>/videos/ 에 저장(RECORD_VIDEOS 환경변수).
    """
    task = task or MAIN_SIM
    step = CKPT_STEP if step is None else step
    n = n_episodes or EVAL_N_EP
    is_te = tag.endswith("_te")                        # act_te, bimamba_te, …
    src = tag[:-3] if is_te else tag                   # ckpt 를 가져올 학습 태그(TE 는 재학습 X)

    cd = v23.best_ckpt_dir(src, seed, task, how=step)  # int → 그 step (없으면 최근접)
    if cd is None:
        raise FileNotFoundError(f"No {step:,} checkpoint for {src}/{task}/seed{seed}")
    ckpt = v23._pretrained(cd)

    out = eval_rep_dir(tag, seed, task, rep, step)
    out.mkdir(parents=True, exist_ok=True)
    _ds, env_type, env_task = v23.TASKS[task]
    batch = LIBERO_EVAL_BATCH if str(task).startswith("libero") else min(n, v23.EVAL_BATCH_SIZE)

    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}", f"MUJOCO_EGL_DEVICE_ID={gpu_id}", "MPLBACKEND=Agg",
        f"RECORD_DIR={out / 'actions'}",
    ]
    if n_videos > 0:
        parts.append(f"RECORD_VIDEOS={int(n_videos)}")   # lerobot_eval: <out>/videos/ 에 mp4 저장
    parts += [
        f"{v23.PYTHON} -m lerobot.scripts.lerobot_eval", f"--policy.path={ckpt}",
        f"--env.type={env_type}", f"--env.task={env_task}",
        f"--eval.n_episodes={n}", f"--eval.batch_size={batch}",
        f"--seed={EVAL_SEED0 + 100 * rep}",            # ★ rep 마다 다른 env 초기상태
        f"--output_dir={out}",
    ]
    if is_te:
        parts.extend(TE_FLAGS)
    return " ".join(parts)


def record_videos_cmd(tag, seed, task=None, gpu_id=0, n_videos=5, n_episodes=None, step=None):
    """영상 저장 전용 eval — 150k 체크포인트로 **n_videos 개 에피소드 mp4** 를 저장.

    별도 폴더(eval_clean/<task>/<tag>/seed<N>/videos_rep/videos/)에 저장 → 기존 SR/떨림 결과와 분리.
    영상은 무겁고 느리므로 rep 반복 없이 **한 번, 적은 에피소드**만.
    """
    task = task or MAIN_SIM
    step = CKPT_STEP if step is None else step
    n = n_episodes or n_videos           # 영상 개수만큼만 돌리면 충분
    src = tag[:-3] if tag.endswith("_te") else tag
    cd = v23.best_ckpt_dir(src, seed, task, how=step)
    if cd is None:
        raise FileNotFoundError(f"No {step:,} checkpoint for {src}/{task}/seed{seed}")
    ckpt = v23._pretrained(cd)

    out = v23.eval_clean_dir(tag, seed, task) / "videos_rep"
    out.mkdir(parents=True, exist_ok=True)
    _ds, env_type, env_task = v23.TASKS[task]
    batch = LIBERO_EVAL_BATCH if str(task).startswith("libero") else min(n, v23.EVAL_BATCH_SIZE)
    parts = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}", f"MUJOCO_EGL_DEVICE_ID={gpu_id}", "MPLBACKEND=Agg",
        f"RECORD_VIDEOS={int(n_videos)}",
        f"{v23.PYTHON} -m lerobot.scripts.lerobot_eval", f"--policy.path={ckpt}",
        f"--env.type={env_type}", f"--env.task={env_task}",
        f"--eval.n_episodes={n}", f"--eval.batch_size={batch}",
        f"--seed={EVAL_SEED0}", f"--output_dir={out}",
    ]
    if tag.endswith("_te"):
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


def rep_sr(tag, seed, task, rep, step=None):
    """rep 하나의 SR(%) — 없으면 None."""
    info = eval_rep_dir(tag, seed, task, rep, step) / "eval_info.json"
    if not info.exists():
        return None
    import json as _json
    pc = _json.loads(info.read_text()).get("overall", {}).get("pc_success")
    return None if pc is None else float(pc)


def sr_over_reps(tag, task=None, seeds=None, reps=None, step=None):
    """반복 eval 집계: {seed: [rep별 SR]}, 전체 평균/표준편차/n.

    반환: dict(per_seed={seed: mean}, all=[모든 rep SR], mean, std, n_runs, n_seed)
    """
    import statistics as _st
    task = task or MAIN_SIM
    seeds = seeds or MAIN_SEEDS
    reps = range(EVAL_REPEATS) if reps is None else reps
    per_seed, allv = {}, []
    for s in seeds:
        vals = [v for v in (rep_sr(tag, s, task, r, step) for r in reps) if v is not None]
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


# ── 노트북 2개로 나눠 돌리기 (GPU 2장씩) ─────────────────────────────────────
# seed 만 나누고, **GPU 는 그 노드에 실제로 있는 것**을 0번부터 쓴다.
#   · 2-GPU 노드 두 대(A 노드 / B 노드) → 양쪽 다 GPU [0,1]. ★우리 클러스터가 이 경우.
#   · 한 4-GPU 노드에서 A·B 를 동시에 띄울 때만 서로 밟지 않게 GPU 를 직접 나눠줄 것:
#       SEEDS, GPUS = cf.part('B', gpus=[2, 3])
# ⚠️ 없는 GPU 를 지정하면 triton 이 `RuntimeError: 0 active drivers` 로 죽는다(장치가 안 보임).
PART_SEEDS = {"A": [0, 1], "B": [2, 3]}


def part(name, gpus=None):
    """('A'|'B'[, gpus]) → (seeds, gpus).  gpus 생략 시 이 노드의 GPU 를 0번부터 seed 수만큼."""
    seeds = list(PART_SEEDS[name.strip().upper()])
    if gpus is None:
        avail = v23.available_gpus()
        gpus = list(avail)[: len(seeds)]
    return seeds, list(gpus)


def _check_gpus(gpus):
    """없는 GPU 를 지정했으면 여기서 잡는다(안 그러면 triton 의 '0 active drivers' 로 죽음)."""
    avail = set(v23.available_gpus())
    bad = [g for g in gpus if g not in avail]
    if bad:
        raise RuntimeError(
            f"이 노드에 없는 GPU {bad} 를 지정했다 (보이는 GPU: {sorted(avail)}).\n"
            "  · 2-GPU 노드라면 cf.part('B') 를 그냥 쓰면 GPU 0,1 로 자동 배정된다.\n"
            "  · 한 노드에서 A/B 를 동시에 돌릴 때만 cf.part('B', gpus=[2,3]) 처럼 직접 나눌 것.\n"
            "  (없는 GPU 를 주면 학습 프로세스가 RuntimeError: 0 active drivers 로 죽는다)"
        )


def run_training(tags, seeds=None, task=None, ngpu=None, gpus=None, prefetch=True):
    """tags x seeds 를 GPU 수만큼 청크로 학습(각 청크 끝날 때까지 대기). resume 자동.

    gpus=[0,1] → **그 GPU 만** 사용(창 2개를 서로 다른 GPU 로 띄울 때).
    gpus=None  → ngpu(기본: 이 노드의 GPU 수)만큼 0번부터 사용.
    prefetch=True → 병렬 실행 전에 데이터셋을 한 번 받아둔다(동시 다운로드 레이스 방지).
    """
    seeds = seeds or MAIN_SEEDS
    task = task or MAIN_SIM
    gpus = list(gpus) if gpus else list(range(ngpu or n_gpu()))
    _check_gpus(gpus)                       # 없는 GPU → 여기서 즉시 실패(학습 띄운 뒤 죽지 않게)
    if prefetch and not prefetch_dataset(task):
        raise RuntimeError("데이터셋 prefetch 실패 → 병렬 학습을 띄우면 전부 죽는다. 위 로그 확인.")
    jobs = train_jobs(seeds, tags=tags, task=task)
    n = len(gpus)
    print(f"학습 {tags} x seed{seeds} = {len(jobs)}잡  (GPU {gpus}, {STEPS:,} step, lr 고정)")
    for i in range(0, len(jobs), n):
        chunk = jobs[i:i + n]
        print(f"\n===== 청크 {i // n + 1}/{-(-len(jobs) // n)} ({len(chunk)}잡) =====")
        labeled = []
        for g, (tag, seed, tk) in zip(gpus, chunk):
            out = v23.train_dir(tag, seed, tk)
            done_step = v23.last_ckpt_step(out)
            if done_step is not None and done_step >= CKPT_STEP:
                print(f"  [skip] {v23.run_label(tag, seed, tk)} — 이미 {done_step:,} 완료")
                continue
            v23.clean_if_no_ckpt(out)               # 체크포인트 없는 크래시 잔재만 정리
            mode = f"resume @{done_step:,}" if done_step else "새로 시작"
            print(f"   GPU{g}  {v23.run_label(tag, seed, tk):<28} {mode}")
            labeled.append((v23.run_label(tag, seed, tk), make_train_cmd(tag, seed, tk, g)))
        if labeled:
            v23.launch_cmds_live(labeled, log_tag="train")
    print("\n학습 완료:", tags, seeds)
    return jobs


def resume_status(tags, seeds=None, task=None):
    """seed 별 진행 상황: 마지막 체크포인트 step / 남은 step / resume 가능 여부."""
    seeds = seeds or MAIN_SEEDS
    task = task or MAIN_SIM
    print(f"{task} — 목표 {STEPS:,} step")
    print(f"{'RUN':<26}{'마지막 ckpt':>14}{'남음':>12}   상태")
    print("-" * 68)
    todo = []
    for t in tags:
        for s in seeds:
            out = v23.train_dir(t, s, task)
            step = v23.last_ckpt_step(out)
            tc = v23.last_train_config(out)
            if step is None:
                state = "체크포인트 없음 → 처음부터"
                left = STEPS
            elif step >= STEPS:
                state = "✅ 완료"
                left = 0
            elif tc is None:
                state = "⚠️ ckpt 는 있는데 train_config 없음 → 처음부터"
                left = STEPS
            else:
                state = "▶ resume 가능"
                left = STEPS - step
            if left:
                todo.append((t, s))
            print(f"{v23.run_label(t, s, task):<26}{(f'{step:,}' if step else '-'):>14}"
                  f"{left:>12,}   {state}")
    print(f"\n이어서 돌릴 것: {todo or '없음 (전부 완료)'}")
    return todo


def run_repeat_evals(tags, seeds=None, reps=None, task=None, ngpu=None, gpus=None, n_episodes=None,
                     step=None):
    """150k 체크포인트 x rep 반복 eval. 이미 끝난 run 은 skip → 재실행 안전.

    gpus=[0,1] → 그 GPU 만 사용(창 2개를 서로 다른 GPU 로 띄울 때).
    """
    gpus = list(gpus) if gpus else list(range(ngpu or n_gpu()))
    _check_gpus(gpus)
    seeds = seeds or MAIN_SEEDS
    reps = list(range(EVAL_REPEATS)) if reps is None else list(reps)
    task = task or MAIN_SIM
    n = n_episodes or EVAL_N_EP

    step = CKPT_STEP if step is None else int(step)
    runs = [(t, s, r) for s in seeds for t in tags for r in reps]
    todo = [x for x in runs if rep_sr(x[0], x[1], task, x[2], step) is None]
    print(f"eval {tags} | {step:,} ckpt x {len(reps)}rep x {len(seeds)}seed x {n}ep | GPU {gpus}")
    print(f"  전체 {len(runs)} run / 남은 {len(todo)} (완료 {len(runs) - len(todo)})")

    ng = len(gpus)
    for i in range(0, len(todo), ng):
        chunk = todo[i:i + ng]
        labeled = []
        for g, (t, s, r) in zip(gpus, chunk):
            try:
                labeled.append((f"{t}/seed{s}/rep{r}@{step // 1000}k",
                                repeat_eval_cmd(t, s, r, task=task, gpu_id=g, n_episodes=n, step=step)))
            except FileNotFoundError as e:
                print("  skip:", e)
        if labeled:
            print(f"\n===== eval 청크 {i // ng + 1}/{-(-len(todo) // ng)} ({len(labeled)} run) =====")
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


def sr_table(tags, seeds=None, reps=None, task=None, n_episodes=None, csv_path=None, step=None):
    """반복 eval 집계 표. mean±std(= seed x rep run) + pooled Wilson CI. rows 반환."""
    import csv as _csv
    seeds = seeds or MAIN_SEEDS
    reps = list(range(EVAL_REPEATS)) if reps is None else list(reps)
    task = task or MAIN_SIM
    n = n_episodes or EVAL_N_EP

    rows = []
    for t in tags:
        agg = sr_over_reps(t, task=task, seeds=seeds, reps=reps, step=step)
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


def sr_grid_table(tags, task=None, seeds=None, reps=None, step=None, png_path=None,
                  csv_path=None, model_avg_row=True, figsize_scale=1.0):
    """표 이미지: **행 = 모델 × seed**, **열 = eval 1..N + avg**.

        | model | seed |  1   |  2   |  3   |  4   |  5   |  avg  |
        | ours  |  0   | 60.8 | 62.0 | 64.2 | 61.4 | 63.0 | 62.28 |
        |       |  1   | ...                                       |
        |       | mean | ...                              | 62.10 |   <- 모델 전체 평균

    각 칸 = 그 (seed, eval) 한 run 의 SR(%). 결측은 빈칸, avg 는 있는 것만 평균.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    task = task or MAIN_SIM
    seeds = seeds or MAIN_SEEDS
    reps = list(range(EVAL_REPEATS)) if reps is None else list(reps)
    step = CKPT_STEP if step is None else int(step)

    rows, body, bold = [], [], []          # bold = 굵게 표시할 행 인덱스(모델 평균)
    for t in tags:
        got_any = False
        seed_rows = []
        for sd in seeds:
            cells = [rep_sr(t, sd, task, r, step) for r in reps]
            if all(v is None for v in cells):
                continue
            got_any = True
            vals = [np.nan if v is None else float(v) for v in cells]
            seed_rows.append((str(sd), vals, float(np.nanmean(vals))))
        if not got_any:
            continue
        for sd, vals, avg in seed_rows:
            rows.append((v23.MODEL_DIR_NAMES.get(t, t), sd))
            body.append(vals + [avg])
        if model_avg_row and len(seed_rows) > 1:
            allv = np.array([v for _, vals, _ in seed_rows for v in vals], dtype=float)
            per_rep = [float(np.nanmean([vals[i] for _, vals, _ in seed_rows]))
                       for i in range(len(reps))]
            rows.append((v23.MODEL_DIR_NAMES.get(t, t), "mean"))
            body.append(per_rep + [float(np.nanmean(allv))])
            bold.append(len(rows) - 1)

    if not rows:
        print("eval 결과 없음")
        return []

    cell_text = []
    for (mname, sname), vals in zip(rows, body):
        cells = [mname, sname]
        cells += [("" if np.isnan(v) else f"{v:.1f}") for v in vals[:-1]]
        cells.append("" if np.isnan(vals[-1]) else f"{vals[-1]:.2f}")
        cell_text.append(cells)

    ncol = len(reps) + 2
    fig, ax = plt.subplots(figsize=((2 + len(reps) * 1.25) * figsize_scale,
                                    (1 + len(rows) * 0.45) * figsize_scale))
    ax.axis("off")
    tab = ax.table(cellText=cell_text,
                   colLabels=["model", "seed"] + [str(r + 1) for r in reps] + ["avg"],
                   cellLoc="center", loc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(12)
    tab.scale(1, 1.55)

    prev_model = None
    for (i, j), cell in tab.get_celld().items():
        cell.set_edgecolor("#666")
        if i == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(weight="bold")
            continue
        r0 = i - 1
        if r0 in bold:                                   # 모델 평균 행
            cell.set_facecolor("#f4f4f4")
            cell.set_text_props(weight="bold")
        if j == ncol - 1 and r0 not in bold:             # avg 열
            cell.set_text_props(weight="bold")
    # 모델명은 그 모델의 첫 행에만 표시
    for r0, (mname, _) in enumerate(rows):
        if mname == prev_model:
            c = tab[r0 + 1, 0]
            c.get_text().set_text("")
        prev_model = mname

    ax.set_title(f"{task} — SR per eval run (%)   [{step // 1000}k checkpoint, {len(reps)} evals]",
                 fontweight="bold", pad=12)
    fig.tight_layout()
    if png_path:
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        print("표 이미지:", png_path)
    if csv_path:
        import csv as _csv
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["model", "seed"] + [f"eval{r + 1}" for r in reps] + ["avg"])
            w.writerows(cell_text)
        print("표 CSV   :", csv_path)
    return cell_text


def action_trajs(tag, seed, task=None, reps=None, step=None):
    """action 궤적(.pt) 수집 — 반복 eval(rep*/actions) 우선, 없으면 단일 eval, 없으면 학습중 eval.

    04_report_jerk 가 이걸 통해 rep 5개를 전부 pool 해서 jerk 통계를 냄.
    """
    task = task or MAIN_SIM
    reps = range(EVAL_REPEATS) if reps is None else reps
    trajs = []
    for r in reps:                                   # 1) 반복 eval
        trajs += v23._load_action_trajs(eval_rep_dir(tag, seed, task, r, step) / "actions") or []
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


# ══════════════════════════════════════════════════════════════════════════════
# LIBERO 멀티노드 분할 — 노드 3대(GPU 2 / 4 / 4 = 합 10)로 나눠 돌린다.
#   · 잡(tag,seed)은 서로 독립 → 어느 노드가 돌리든 결과 경로는 (tag,seed)로 고정 → 리포트가 자동 pooled.
#   · node_split 은 GPU 수 비(2:4:4)로 잡을 결정론적으로 3등분한다(A=적게, B/C=많게).
#   · 각 노드는 **자기 노드에 보이는 GPU 를 0번부터** 쓴다(하드코딩 금지). run_training_jobs 가 처리.
# ══════════════════════════════════════════════════════════════════════════════
LIBERO_SEEDS = [0, 1, 2, 3]                 # 메인 표와 동일 seed 4개
NODE_GPUS = {"A": 2, "B": 4, "C": 4}        # 우리 클러스터: 노드 3대 = GPU 2 / 4 / 4 (합 10)


def libero_all_jobs(tags=None, seeds=None, task=None):
    """LIBERO 전체 (tag, seed, task) 잡. 기본 = TRAIN_TAGS(6모델) × LIBERO_SEEDS(4) × libero_10 = 24잡."""
    tags = tags or TRAIN_TAGS
    seeds = seeds or LIBERO_SEEDS
    task = task or SUPPORT_SIM
    return [(t, s, task) for s in seeds for t in tags]


def node_split(jobs, weights=None):
    """잡 리스트를 GPU 가중치(기본 2:4:4)로 노드별 버킷으로 나눈다(결정론적). {'A':[...], 'B':[...], 'C':[...]}."""
    weights = weights or NODE_GPUS
    names = list(weights)
    total = sum(weights.values())
    n = len(jobs)
    out, start, acc = {}, 0, 0
    for i, name in enumerate(names):
        acc += weights[name]
        end = n if i == len(names) - 1 else round(n * acc / total)
        out[name] = jobs[start:end]
        start = end
    return out


def node_jobs(node, tags=None, seeds=None, task=None, weights=None):
    """이 노드(A/B/C)가 맡을 LIBERO 잡 리스트."""
    return node_split(libero_all_jobs(tags, seeds, task), weights)[node.strip().upper()]


def libero_trained_pairs(tags=None, seeds=None, task=None, step=None):
    """**실제로 체크포인트가 있는** (tag, seed) 만 반환 — eval-only 분배용.

    학습이 몇 seed 까지 됐든(3개/4개), 어떤 모델이 빠졌든 있는 것만 골라낸다.
    step 근처(기본 150k)의 체크포인트가 있으면 포함. best_ckpt_dir 가 없으면 None → 제외.
    ⚠️ 이 노드에서 보이는 경로만 스캔 → 노드 간 공유 FS 를 가정(대개 클러스터 표준).
    """
    tags = tags or TRAIN_TAGS
    seeds = seeds or LIBERO_SEEDS
    task = task or SUPPORT_SIM
    step = CKPT_STEP if step is None else step
    out = []
    for s in seeds:
        for t in tags:
            if v23.best_ckpt_dir(t, s, task, how=step) is not None:
                out.append((t, s))
    return out


def libero_eval_node_pairs(node, weights=None, tags=None, seeds=None, task=None, step=None):
    """eval-only: 학습된 (tag,seed) 를 탐색해 GPU 비(2:4:4)로 나눈 뒤 이 노드 몫을 반환."""
    pairs = libero_trained_pairs(tags, seeds, task, step)
    return node_split(pairs, weights)[node.strip().upper()]


def split_by_gpu(items, gpu_counts):
    """items 리스트를 gpu_counts 비율로 연속 3등분(결정론적). 예: 24개, [4,4,2] → [10, 9, 5].

    노드별 eval 노트북이 '내가 몇 번째 노드' 만 정하면 자기 몫을 이 함수로 뽑는다.
    """
    n = len(items)
    total = sum(gpu_counts)
    out, start, acc = [], 0, 0
    for i, g in enumerate(gpu_counts):
        acc += g
        end = n if i == len(gpu_counts) - 1 else round(n * acc / total)
        out.append(items[start:end])
        start = end
    return out


def run_training_jobs(jobs, gpus, prefetch_task=None, prefetch=True):
    """명시적 (tag,seed,task) 잡 리스트를 주어진 로컬 GPU 로 학습(resume/skip/청크 자동)."""
    jobs = list(jobs)
    gpus = list(gpus)
    _check_gpus(gpus)
    if prefetch and jobs:
        pt = prefetch_task or jobs[0][2]
        if not prefetch_dataset(pt):
            raise RuntimeError("데이터셋 prefetch 실패 → 병렬 학습을 띄우면 전부 죽는다. 위 로그 확인.")
    n = len(gpus)
    print(f"학습 {len(jobs)}잡  (GPU {gpus}, {STEPS:,} step, lr 고정)")
    for i in range(0, len(jobs), n):
        chunk = jobs[i:i + n]
        print(f"\n===== 청크 {i // n + 1}/{-(-len(jobs) // n)} ({len(chunk)}잡) =====")
        labeled = []
        for g, (tag, seed, tk) in zip(gpus, chunk):
            out = v23.train_dir(tag, seed, tk)
            done = v23.last_ckpt_step(out)
            if done is not None and done >= CKPT_STEP:
                print(f"  [skip] {v23.run_label(tag, seed, tk)} — 이미 {done:,} 완료")
                continue
            v23.clean_if_no_ckpt(out)
            mode = f"resume @{done:,}" if done else "새로 시작"
            print(f"   GPU{g}  {v23.run_label(tag, seed, tk):<28} {mode}")
            labeled.append((v23.run_label(tag, seed, tk), make_train_cmd(tag, seed, tk, g)))
        if labeled:
            v23.launch_cmds_live(labeled, log_tag="train")
    print("\n학습 완료:", len(jobs), "잡")
    return jobs


def run_libero_eval_jobs(pairs, gpus, task=None, n_episodes=None, reps=None, step=None, n_videos=0):
    """명시적 (tag,seed) 쌍을 LIBERO eval (150k 체크포인트, 기본 500ep, 이미 끝난 run 은 skip)."""
    pairs = list(pairs)
    gpus = list(gpus)
    _check_gpus(gpus)
    task = task or SUPPORT_SIM
    n = n_episodes or EVAL_N_EP
    reps = [0] if reps is None else list(reps)
    step = CKPT_STEP if step is None else int(step)

    # ★ 유효한 500ep(overall n_ep>=2500=500×10) eval 이 이미 있으면 skip. 옛 50ep(overall 500)·
    #   미완(eval_info 없음)은 skip 안 함 → 다시 돌리면 lerobot_eval 이 task 단위로 이어서 함.
    _min_valid = 10 * EVAL_N_EP // 2

    def _has_valid(t, s, r):
        info = eval_rep_dir(t, s, task, r, step) / "eval_info.json"
        if not info.exists():
            return False
        try:
            ov = json.loads(info.read_text()).get("overall", {})
            ne = ov.get("n_ep", ov.get("n_episodes")) or 0
        except Exception:
            return False
        return ne >= _min_valid

    runs = [(t, s, r) for (t, s) in pairs for r in reps]
    todo = [x for x in runs if not _has_valid(*x)]
    print(f"LIBERO eval {len(pairs)}쌍 × {len(reps)}rep × {n}ep | {step:,} ckpt | GPU {gpus}")
    print(f"  전체 {len(runs)} run / 남은 {len(todo)} (유효 500ep 완료 {len(runs) - len(todo)} skip)")
    ng = len(gpus)
    for i in range(0, len(todo), ng):
        chunk = todo[i:i + ng]
        labeled = []
        for g, (t, s, r) in zip(gpus, chunk):
            try:
                labeled.append((f"{t}/seed{s}/rep{r}",
                                repeat_eval_cmd(t, s, r, task=task, gpu_id=g,
                                                n_episodes=n, step=step, n_videos=n_videos)))
            except FileNotFoundError as e:
                print("  skip:", e)
        if labeled:
            print(f"\n===== eval 청크 {i // ng + 1}/{-(-len(todo) // ng)} ({len(labeled)} run) =====")
            v23.launch_cmds_live(labeled)
    print("\nLIBERO eval 완료:", len(pairs), "쌍")


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
