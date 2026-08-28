"""exp5_tonight.py — 2026-08-24 밤 실행 드라이버. 6 GPU = 3노드 x 2 GPU (A/B/C).

은지님 8/24 요청("ACT 대비 월등히 높게 나올 세팅 찾기, 세팅마다 act/bimamba only/carry only/bimos
4종 전부, 50ep") + 8/18 교수님 지시(overlap 제거, ACT 무너지는 지점 찾기)의 실행판.

노트북 3개(nA/nB/nC_tonight.ipynb)가 전부 이 모듈만 부른다. 잡 정의/우선순위/스킵/집계가 전부 여기.

--------------------------------------------------------------------------
승리 조건 (8/25 재설계 — 이게 이 파일의 우선순위를 결정한다)
--------------------------------------------------------------------------
"ACT 를 넘는다" 는 두 가지가 섞여 있었고 둘 다 명시적으로 노린다.

  W1. 동일 세팅 우위 — 같은 (K, stride, TE) 에서 ours > ACT.
      이미 있다: K=100 TE 에서 +18.8 (49.3 vs 30.5), K=100 s100 TE-off +10.1.
  W2. 토너먼트 우위 — 각 방법에 **같은 탐색 예산**(K x stride x TE on/off)을 주고
      방법별 best 끼리 비교했을 때 우리가 1등. 은지님 8/24 "67.6 을 넘을 값"이 이것.
      우리 best 후보만 골라 놓고 ACT 는 한 세팅만 재면 체리피킹이라 리뷰에서 깨진다.

W2 의 판정 셀 = act+TE @ K=10/15/20. 우리 최고 후보(BiMamba+TE 75.4/75.6/71.4,
은지님 K sweep)가 전부 짧은 K 에 있는데, 그 K 들의 act+TE 열이 통째로 비어 있었다.

[8/25 오후 판정] exp3_act_te_ksweep 이 act+TE 열을 채웠다 (ak{K}_te 태그, 100ep/task,
seed0, 동일 UBAI). act_k10/15 는 이미 학습돼 있었다:

    K        10    15    20    50   100   150
    act+TE  72.5  67.3  61.2  39.7  27.7  21.9   <- K 에 대해 단조 붕괴
    bimamba 75.4  75.6  71.4  64.1  49.3  26.8   (cpoff)
    +carry  74.3  76.4  71.9  63.6  50.2  27.1   (cpoff)

전 K 에서 우리가 위. max(ours)=76.4(K=15) vs max(ACT)=72.5(K=10) -> 100ep 기준 W2 승리.
남은 확정 작업 2개:
  (1) K=10/15/20 을 500ep 로 재측정 — bimamba 쪽이 n=100(SE ±4.3)이라 K=10/15 마진
      (+2.9/+9.1)이 오차 안이다. 이 파일의 te_* 셀이 그 확정 런이다 (exp3 는 ak{K}_te
      태그로 저장돼 exp5 가 te_* 로 다시 돈다 — 중복이 아니라 n 업그레이드).
  (2) 순수 bimamba K=10/15 학습+측정 — 현재 bimamba 값은 carry-학습 오염판(cpoff).
      critical 은 이제 이 2잡뿐이다 (act_k10/15 는 is_ready 로 자동 이탈).

잔존 리스크: 500ep 에서 K=10/15 마진이 뒤집히면 K=20/50 마진(+10.2/+24.4)으로 후퇴,
그것도 흔들리면 폴백 = W1(긴 chunk 크로스오버 + 효율) + real-world(SO-101).

--------------------------------------------------------------------------
설계 근거 3가지
--------------------------------------------------------------------------
1. TE 는 재학습이 필요 없다.  common_final.act_te_eval_cmd 가 하듯 ACT ckpt 에
   `--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1` 만 얹으면 된다.
   ACT ckpt 가 있는 K 는 eval 만 하면 된다.

2. TE 와 stride 는 같은 축에 못 놓는다. configuration_acm2.py 가
   `temporal_ensemble_coeff is not None and n_action_steps > 1` 를 NotImplementedError 로 막는다.
   -> TE 축(stride=1 고정, K 변화) 과 stride 축(TE off) 은 별개 잡군이다.
   따라서 stride 큰 레짐에서만 "추론 호출 K 배 절약" 효율 주장이 성립한다 — TE 는 매 스텝 추론.

3. 기존 `bimamba` 태그는 순수 BiMamba 가 아니다.  _CARRY_ON + use_chunk_pairs=True 로
   학습됐다(은지님 8/18 지적). 순수판 `bimamba_pure` 는 8/24 밤에 K=50/100/150 학습 완료.
   오염된 프록시는 `bimamba_cpoff` 라벨로 표에 그대로 남긴다. 은지님의 75.4(K=10)는
   은지님 환경 + 오염판 값이므로, UBAI 에서 순수판으로 재측정해야 논문에 쓴다.

4. 8/18 stride sweep 슬라이드는 aloha 가 섞였다 — 은지님 8/24 확인: "76 까지 나온 건
   libero 에는 없다, 67.6 이 최고". libero/aloha 슬라이드의 s=10/50/75 행이 동일하다.
   따라서 ACT 76.0(K=100 s50), BiMamba 75.0(K=100 s75) 등은 libero 근거로 못 쓰고,
   교차검증된 libero 행은 K=100 s=100 하나뿐(ACT 18.8 / ACM2 25.3 / BiMamba 31.9 /
   BiMOS 25.6). ACT 의 진짜 libero 최고 = 67.6(K=50 s10, TE off).
   -> 이 파일의 레짐 맵(rm_* 셀)이 그 실험을 UBAI 에서 처음부터 재구현하는 것이다.

--------------------------------------------------------------------------
예산 (8/24 밤 실측)
--------------------------------------------------------------------------
  eval 1셀 = LIBERO-10 x 50ep/task = 500ep ~= 2 GPU-h
  학습 1잡 = 150k step               ~=  8 GPU-h (4잡/2GPU/2배치 ≈ 16h 실측)
  노드 역할은 suggest() 가 정한다: critical 학습이 노드를 먼저 선점하고,
  나머지는 ready eval 큐를 포화시키고, 남으면 일반 학습.
"""
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent          # <repo>/notebooks/libero
_NB = _HERE.parent                               # <repo>/notebooks
for _p in (str(_NB), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

cf = None       # setup() 에서 채운다
v23 = None

# -- 프로토콜 (은지님 8/18 지정) ---------------------------------------------
TASK = "libero_10"
SEED = 0
N_EP = 50                 # task 당 50 -> LIBERO-10 overall 500 (은지님 지정)
MIN_TRAIN_STEP = 100_000  # 이보다 적으면 '학습중'으로 보고 eval 대상에서 뺀다
TE_FLAGS = ["--policy.temporal_ensemble_coeff=0.01", "--policy.n_action_steps=1"]

# -- 변형 5종 ----------------------------------------------------------------
# key: (dir base, policy type, 설명)
#   carry/bimamba/bimos 의 학습·eval 플래그는 _train_extra / _eval_extra 가 만든다
#   (ACT 에 --policy.sscp_* 를 주면 파싱 에러로 죽기 때문에 정책별로 갈라야 한다).
VARIANTS = {
    "act":           ("act",          "act",                       False, "ACT"),
    "acm2":          ("acm2",         "acm2",                      False, "ACM2 (plain)"),
    "carry":         ("acm2_carry",   "acm2_sscp_literal",         True,  "carry only"),
    "bimamba":       ("bimamba_pure", "acm2_sscp_literal_bimamba", False, "BiMamba only (pure)"),
    "bimos":         ("bimamba",      "acm2_sscp_literal_bimamba", True,  "BiMamba + carry"),
    # eval 전용 프록시: bimos ckpt 에 carry 만 꺼서 본다. 학습이 chunk-pair 라 순수 아님.
    "bimamba_cpoff": ("bimamba",      None,                        None,  "BiMamba (chunk-pair trained!)"),
}
#  변형 -> use_chunk_pairs (학습 시). 순수 BiMamba 는 False 여야 한다.
USE_CHUNK_PAIRS = {"act": False, "acm2": False, "carry": True, "bimamba": False, "bimos": True}
REUSE_CKPT_OF = {"bimamba_cpoff": "bimos"}      # 학습 안 함 -> 이 변형의 ckpt 사용
TRAINABLE = ("act", "acm2", "carry", "bimamba", "bimos")
#  sscp 플래그를 못 받는 정책(파싱 에러로 죽는다). ACT 와 plain ACM2.
NO_SSCP_FLAG = ("act", "acm2")

# K=10/15 는 8/25 추가. 우리 최고 후보(BiMamba+TE 75.4/75.6)가 거기 있는데 스윕에서
# 빠져 있었다 — 이길 가능성이 있는 유일한 절대값 지점을 안 재고 있었던 것.
K_ALL = [10, 15, 20, 50, 100, 150]


def _carry_flag(variant, on):
    """정책별로 유효한 플래그만. ACT / plain ACM2 에는 sscp 플래그 자체가 없다."""
    if variant in NO_SSCP_FLAG:
        return []
    return [v23._CARRY_ON if on else v23._CARRY_OFF]


def _eval_extra(variant):
    return _carry_flag(variant, variant in ("carry", "bimos"))


def _train_extra(variant):
    # 순수 BiMamba 는 carry off 로 학습(+ use_chunk_pairs=False).
    return _carry_flag(variant, variant in ("carry", "bimos"))


def tag_of(variant, K):
    """(변형, K) -> 학습/ckpt 디렉토리 태그.  K=100 은 기존 이름 그대로(ckpt 재사용)."""
    base = VARIANTS[REUSE_CKPT_OF.get(variant, variant)][0]
    return base if K == 100 else f"{base}_k{K}"


#  실행 stride 격자. 8/25 K=100 결과에서 s10 이 봉우리(BiMamba 65.0 vs ACT 42.5)인데
#  s5 / s15 / s25 가 비어 있어 봉우리를 감싸지 못했다 -> 5 와 15 를 추가.
def strides_for(K):
    return [s for s in (1, 5, 10, 15, 20, 25, 50, 75, 100, 150) if s <= K]


# stride sweep 우선순위: 봉우리(10) 양옆부터. 이미 측정된 10/50/100 은 skip 로직이 거른다.
STRIDE_SWEEP_ORDER = (5, 15, 25, 10, 50, 75, 1, 100)


# -- 부팅 / 태그 등록 --------------------------------------------------------
def setup(verbose=True):
    """common_final 로드 + 이 실험이 쓰는 태그 전부 등록. 노트북 첫 셀에서 부른다.

    주의: 노트북이 importlib.reload(cf) 를 부르면 MODEL_CONFIGS 가 초기화되므로,
    reload 이후에 반드시 이 함수를 다시 불러야 한다(노트북 셀 1이 그렇게 돼 있다).
    """
    global cf, v23
    import importlib
    import common_final as _cf
    _cf = importlib.reload(_cf)
    cf, v23 = _cf, _cf.v23
    _register_all()
    if verbose:
        v23.cuda_report()
        print(f"OUTPUT_BASE = {v23.OUTPUT_BASE}")
        print(f"TASK={TASK} SEED={SEED} N_EP={N_EP}(task당) CKPT_STEP={cf.CKPT_STEP:,}")
    return cf, v23


def _reg(tag, policy, K, extra, use_cp, label):
    v23.MODEL_CONFIGS[tag] = (policy, v23.LR, K, list(extra), use_cp)
    v23.MODEL_DIR_NAMES[tag] = tag
    v23.MODEL_LABELS[tag] = label


def _register_all():
    # 1) 학습 태그 (변형 x K)
    for variant in TRAINABLE:
        base, policy, _cp, desc = VARIANTS[variant]
        for K in K_ALL:
            t = tag_of(variant, K)
            if t not in v23.MODEL_CONFIGS:
                _reg(t, policy, K, _train_extra(variant), USE_CHUNK_PAIRS[variant],
                     f"{desc} (K={K})")
            else:
                # 기존 태그(act / bimamba / acm2_carry)는 dir 이름을 건드리지 않는다.
                v23.MODEL_DIR_NAMES.setdefault(t, t)
    # 2) eval 출력 태그 (eval_clean_dir 가 MODEL_DIR_NAMES 를 참조한다)
    for j in _all_jobs():
        v23.MODEL_DIR_NAMES.setdefault(j["out"], j["out"])
        v23.MODEL_LABELS.setdefault(j["out"], j["label"])


# -- 잡 정의 -----------------------------------------------------------------
def _rm_job(variant, K, s):
    """레짐 맵 셀 (TE off, 실행 stride = s)."""
    return {
        "kind": "rm", "variant": variant, "K": K, "stride": s,
        "src": tag_of(variant, K),
        "out": f"rm_{variant}_k{K}_s{s}",
        "flags": _eval_extra(variant) + [f"--policy.n_action_steps={s}"],
        "label": f"{variant} K={K} s={s}",
    }


def _te_job(variant, K):
    """TE 셀 (stride=1 강제 + coeff 0.01)."""
    return {
        "kind": "te", "variant": variant, "K": K, "stride": 1,
        "src": tag_of(variant, K),
        "out": f"te_{variant}_k{K}",
        "flags": _eval_extra(variant) + TE_FLAGS,
        "label": f"{variant}+TE K={K}",
    }


#  변형 풀. "bimamba"(순수)는 ckpt 가 생기는 대로 자동 편입된다(is_ready 게이트).
#  은지님 8/24 지정: "세팅 바꿀 때마다 act, bimamba only, carry only, bimos 다 돌려야".
#  거기에 plain ACM2 를 더한다 — 8/25 표의 기준선이고 스캔 ablation 의 분모다.
_ALL_V = ("act", "acm2", "bimamba", "bimamba_cpoff", "bimos", "carry")
MAIN_K = 100          # stride sweep 의 주 chunk. 8/25 에 여기서 +22.5 가 나왔다.
MAIN_STRIDE = 10      # 그 봉우리. K sweep 은 이 stride 로 고정해서 축을 하나만 움직인다.


def _all_jobs():
    """우선순위 = 이 리스트 순서. 앞쪽이 stride sweep(8/25 해준님이 은지님께 약속한 것)."""
    Q = []
    # P1 -- ★ K=100 stride sweep. 8/25 결과가 s10 에서 봉우리인데(BiMamba 65.0 vs
    #   ACT 42.5 = +22.5, n=500) 양옆 s5/s15/s25 가 비어 봉우리를 감싸지 못했다.
    #   ACT 도 s10 근처가 최적일 가능성이 크므로(K=50 에서 s1 47.3 < s10 67.6),
    #   여기서 우리가 전 stride 를 지배하는지가 "동일 추론 비용 우위"의 근거가 된다.
    #   이미 측정된 s10/50/100 은 _eval_done 이 자동으로 건너뛴다.
    for s in STRIDE_SWEEP_ORDER:
        for v in _ALL_V:
            Q.append(_rm_job(v, MAIN_K, s))
    # P2 -- K sweep @ stride 10 고정. 메인 그림(= 교수님의 "ACT 가 무너지는 지점").
    #   현재 이 축에 K=50(ACT 67.6 / BiMamba 57.8) 과 K=100(42.5 / 65.0) 두 점뿐이라
    #   크로스오버가 50~100 사이라는 것만 알고 곡선이 없다. ACT ckpt 는 전 K 에 있으므로
    #   재학습 없이 eval 만으로 채워진다.
    for K in (150, 20, 15, 10, 50):
        for v in _ALL_V:
            Q.append(_rm_job(v, K, MAIN_STRIDE))
    # P3 -- K=50 stride sweep: ACT 최고점(67.6, s10) 재현 + carry 모순 해소
    #   (8/17 수정표 carry 69.5 > ACT 61.1  vs  8/18 슬라이드 ACT 67.6 > carry 64.6.
    #    슬라이드 stride 행은 aloha 오염 가능성까지 있다(설계근거 4) -> UBAI 로 재측정.)
    for s in (25, 5, 50, 15, 1):
        for v in _ALL_V:
            Q.append(_rm_job(v, 50, s))
    # P4 -- TE 축 500ep 확정 런. act+TE 는 8/25 exp3(100ep)로 이미 측정됨
    #   (72.5/67.3/61.2/39.7/27.7/21.9) — 전 K 에서 우리가 위지만 n=100 이라 마진이
    #   오차 안이다. 단 K=100 에선 TE(stride 1) 49.3 < s10 무TE 65.0 이므로 메인은 P1/P2.
    for K in (15, 10, 20, 50, 100, 150):
        for v in _ALL_V:
            Q.append(_te_job(v, K))
    # P5 -- 잔여 격자 (여유 있을 때)
    for K in (20, 150):
        for s in strides_for(K):
            for v in _ALL_V:
                Q.append(_rm_job(v, K, s))
    seen, out = set(), []
    for j in Q:
        if j["out"] not in seen:
            seen.add(j["out"])
            out.append(j)
    return out


# -- 인벤토리 ----------------------------------------------------------------
def ckpt_step(tag):
    return v23.last_ckpt_step(v23.train_dir(tag, SEED, TASK))


def is_ready(tag):
    st = ckpt_step(tag)
    return st is not None and st >= MIN_TRAIN_STEP


def inventory(verbose=True):
    """이 실험이 참조하는 모든 학습 태그의 ckpt 상태. 서버 파일시스템을 실제로 스캔한다."""
    tags = []
    for variant in TRAINABLE:
        for K in K_ALL:
            t = tag_of(variant, K)
            if t not in tags:
                tags.append(t)
    rows = []
    for t in tags:
        st = ckpt_step(t)
        state = "OK" if (st and st >= MIN_TRAIN_STEP) else ("PART" if st else "MISS")
        rows.append((t, st, state))
    if verbose:
        print(f"{'tag':<24}{'ckpt':>10}  state")
        print("-" * 44)
        for t, st, state in rows:
            print(f"{t:<24}{(format(st, ',') if st else '-'):>10}  {state}")
        n = {"OK": 0, "PART": 0, "MISS": 0}
        for _t, _s, x in rows:
            n[x] += 1
        print(f"\nOK {n['OK']} / PART {n['PART']} / MISS {n['MISS']}  (총 {len(rows)})")
        print("  PART = 학습 중단됨. run_trains() 가 resume 한다.")
    return rows


# -- 큐 ----------------------------------------------------------------------
def _eval_done(job):
    info = v23.eval_clean_dir(job["out"], SEED, TASK) / "eval_info.json"
    if not info.exists():
        return False
    try:
        ov = json.loads(info.read_text()).get("overall", {})
        n = ov.get("n_ep", ov.get("n_episodes")) or 0
    except Exception:
        return False
    return n >= 10 * N_EP // 2       # 절반 이상 = 유효 완료로 본다


def eval_queue(only_ready=True, skip_done=True):
    Q = _all_jobs()
    if skip_done:
        Q = [j for j in Q if not _eval_done(j)]
    if only_ready:
        Q = [j for j in Q if is_ready(j["src"])]
    return Q


# critical = W2(토너먼트) 판정에 필수인 학습. suggest() 가 eval 큐 깊이와 무관하게
# 이 잡들에 노드를 선점시킨다. 전부 끝나면 자동으로 선점이 풀린다.
# 8/25 오후: act_k10/15 는 이미 학습돼 있었음이 확인됐다(exp3 학습 셀 전부 skip)
# -> is_ready 로 자동 이탈, 실질 critical = 순수 bimamba 2잡.
CRITICAL_TRAIN_JOBS = [
    ("acm2",    100),   # P1 stride sweep 의 ACM2 열 전체(8셀)를 막는다. 스캔 ablation 의 분모
    ("bimamba", 10),    # 순수 BiMamba K=10 — 75.4 를 UBAI/순수판으로 재측정
    ("bimamba", 15),    # 순수 BiMamba K=15 — 76.4(전체 최고) 자리의 순수판
]

# 학습 우선순위 -- critical 이 항상 앞. 이미 OK 인 태그는 train_queue() 에서 빠진다.
# P2(K sweep @ s10) 를 채우는 순서로: 곡선의 양 끝(150, 20) -> 짧은 K.
TRAIN_PRIORITY = CRITICAL_TRAIN_JOBS + [
    ("acm2",    150),   # K sweep 열 채우기
    ("acm2",     20),
    ("acm2",     50),
    ("acm2",     10),
    ("acm2",     15),
    ("carry",    50),   # 40k 에서 중단 — resume
    ("carry",   150),
    ("carry",    20),
    ("bimamba",  20),   # 순수 K=20 (cpoff 프록시 71.4 의 순수판)
    ("bimos",    10),   # 오염판 K=10/15 — 은지님 75.4/75.6 재현 대조용
    ("bimos",    15),
    ("bimamba", 100),   # 이하는 8/24 밤에 대부분 완료 — 남은 것만 잡힌다
    ("act",     150),
    ("act",      20),
    ("bimamba",  50),
    ("bimos",   150),
    ("bimamba", 150),
    ("act",      50),
    ("bimos",    20),
    ("bimos",    50),
]


def critical_pending():
    """아직 안 끝난 critical 학습 태그."""
    return [tag_of(v, K) for v, K in CRITICAL_TRAIN_JOBS if not is_ready(tag_of(v, K))]


def train_queue(verbose=False):
    """학습이 안 됐거나(MISS) 덜 된(PART) 태그를 우선순위대로. 이미 OK 면 빠진다."""
    blocked = {}
    for j in _all_jobs():
        if not is_ready(j["src"]):
            blocked[j["src"]] = blocked.get(j["src"], 0) + 1
    q, seen = [], set()
    for variant, K in TRAIN_PRIORITY:
        t = tag_of(variant, K)
        if t in seen or is_ready(t):
            continue
        seen.add(t)
        q.append(t)
    # 우선순위 목록에 없는데 eval 큐가 참조하는 태그도 뒤에 붙인다(= 자동 편입).
    for t, _n in sorted(blocked.items(), key=lambda kv: -kv[1]):
        if t not in seen:
            seen.add(t)
            q.append(t)
    if verbose:
        print(f"{'tag':<24}{'ckpt':>10}  막힌 eval 셀")
        print("-" * 46)
        for t in q:
            st = ckpt_step(t)
            print(f"{t:<24}{(format(st, ',') if st else '-'):>10}  {blocked.get(t, 0)}")
    return q


# -- 노드 분배 ---------------------------------------------------------------
NODES = ("A", "B", "C")
NODE_ROLE = {"A": "train", "B": "eval", "C": "eval"}   # 기본값
ROLE_OVERRIDE = {}          # 예: X.ROLE_OVERRIDE['C'] = 'train'  (suggest() 가 알려준다)

EVAL_CELLS_PER_NODE = 12    # 2 GPU x 12h / (2 GPU-h per cell)
TRAIN_JOBS_PER_NODE = 2     # 2 GPU = 동시 2잡


def role_of(node):
    node = node.strip().upper()
    return ROLE_OVERRIDE.get(node, NODE_ROLE[node])


def _nodes_with(role):
    return [n for n in NODES if role_of(n) == role]


EVAL_NODE_ORDER = ("B", "C", "A")   # eval 을 먼저 배정받는 순서 (A 는 기본이 학습 노드)


def suggest(verbose=True):
    """인벤토리를 보고 노드 역할을 추천한다.

    규칙 (8/25 재설계):
    1. **critical 학습이 노드를 먼저 선점한다.** W2(토너먼트) 판정에 필수인 ckpt
       (act/pure @ K=10/15)는 eval 큐가 아무리 깊어도 학습 노드를 확보한다 —
       판정 셀을 여는 학습은 큐 소화보다 가치가 높다. 전부 끝나면 선점이 풀린다.
    2. 남는 노드는 **ready eval 큐를 포화**시킨다.
    3. 그래도 남으면 일반 학습.
    """
    n_ev, n_tr = len(eval_queue()), len(train_queue())
    crit = critical_pending()
    reserve = min(len(NODES), -(-len(crit) // TRAIN_JOBS_PER_NODE)) if crit else 0
    # 단, 돌릴 eval 이 이미 넉넉하면 선점은 1노드까지만. (8/28: stride sweep 으로 ready
    # 큐가 150셀인데 critical 3잡이 2노드를 가져가 eval 이 1노드로 굶던 것을 고침.
    # 막힌 열은 1노드가 밤새 2잡씩 흘려보내면 따라잡힌다.)
    if reserve > 1 and n_ev >= 2 * EVAL_CELLS_PER_NODE:
        reserve = 1
    if n_tr == 0:
        n_eval_nodes = len(NODES)                      # 학습할 게 없으면 전부 eval
    elif n_ev == 0:
        n_eval_nodes = 0                               # 돌릴 eval 이 없으면 전부 학습
    else:
        n_eval_nodes = min(len(NODES), -(-n_ev // EVAL_CELLS_PER_NODE))
    n_eval_nodes = min(n_eval_nodes, len(NODES) - reserve)     # critical 선점분 제외
    roles = {n: "train" for n in NODES}
    for n in EVAL_NODE_ORDER[:n_eval_nodes]:
        roles[n] = "eval"
    if verbose:
        cap = n_eval_nodes * EVAL_CELLS_PER_NODE
        print(f"ready eval 큐 {n_ev}셀 (노드당 하룻밤 ≈ {EVAL_CELLS_PER_NODE}셀) / 학습 대기 {n_tr}잡")
        if crit:
            print(f"  critical 학습 {len(crit)}잡 -> 노드 {reserve}대 선점: {crit}")
        print(f"  -> eval 노드 {n_eval_nodes}대(처리량 {cap}셀) / 학습 노드 {len(NODES) - n_eval_nodes}대")
        print(f"추천 역할: {roles}")
        changed = {n: r for n, r in roles.items() if r != NODE_ROLE[n]}
        if changed:
            print("\n  기본값과 다르다. **세 노트북 모두**에서 부팅 셀 다음에 아래 한 줄을 실행할 것:")
            print(f"    X.ROLE_OVERRIDE = {changed!r}")
            print("  (세 노드가 같은 역할표를 봐야 잡이 안 겹친다)")
        else:
            print("  기본값 그대로 가면 된다 (ROLE_OVERRIDE 불필요).")
    return roles


# ── 정적 분할 ────────────────────────────────────────────────────────────────
# 노드별 몫을 **인벤토리와 무관한 고정 목록**에서 나눈다(_all_jobs / _all_train_tags 는
# ckpt 상태를 안 본다). 그래야 학습이 끝나 큐가 줄어도 노드 간 배정이 흔들리지 않는다.
#   이전 방식(= 매번 만든 ready 큐를 index 로 나누기)은 A 의 학습이 끝나 셀이 ready 로
#   편입되는 순간 B/C 의 index 가 밀려 **두 노드가 같은 셀을 동시에 돌 수 있었다**
#   (같은 out_dir 에 두 프로세스가 써서 결과가 깨진다).
def _node_index(node):
    node = node.strip().upper()
    return NODES.index(node) if node in NODES else 0


def _all_train_tags():
    """학습 대상 전체 태그의 결정론적 순서(ckpt 상태 무관). TRAIN_PRIORITY 먼저, 나머지는 정렬."""
    out, seen = [], set()
    for variant, K in TRAIN_PRIORITY:
        t = tag_of(variant, K)
        if t not in seen:
            seen.add(t)
            out.append(t)
    rest = []
    for variant in TRAINABLE:
        for K in K_ALL:
            t = tag_of(variant, K)
            if t not in seen:
                seen.add(t)
                rest.append(t)
    return out + sorted(rest)


def node_eval_jobs(node):
    """이 노드 몫의 eval 셀 — 고정 분할 후 (ckpt 있음 & 미완료) 만 남긴다."""
    i, k = _node_index(node), len(NODES)
    mine = _all_jobs()[i::k]
    return [j for j in mine if is_ready(j["src"]) and not _eval_done(j)]


def node_train_tags(node, limit=TRAIN_JOBS_PER_NODE):
    """이 노드 몫의 학습 태그 — 고정 분할 후 아직 안 된 것(MISS/PART)만.

    limit: 한 배치 크기(기본 = GPU 수). **limit=None 이면 남은 전부**.
    """
    i, k = _node_index(node), len(NODES)
    mine = [t for t in _all_train_tags()[i::k] if not is_ready(t)]
    return mine if limit is None else mine[:limit]


def node_blocked_evals(node):
    """이 노드 몫 중 **다른 노드가 학습 중이라** 아직 못 도는 셀 수."""
    i, k = _node_index(node), len(NODES)
    return sum(1 for j in _all_jobs()[i::k]
               if not is_ready(j["src"]) and not _eval_done(j))


def plan(node, gpus=None, verbose=True):
    """이 노드 몫 전체. 역할 구분 없이 **학습 먼저, 그 다음 eval** 로 다 돈다."""
    node = node.strip().upper()
    gpus = list(gpus) if gpus else v23.available_gpus()
    ev = node_eval_jobs(node)
    tr_now = node_train_tags(node)                 # 이번 배치(2잡)
    tr_all = node_train_tags(node, limit=None)     # 이 노드가 끝까지 해야 할 전부
    blocked = node_blocked_evals(node)
    if verbose:
        print(f"===== NODE {node}  GPU {gpus} =====")
        print(f"\n[학습] 남은 {len(tr_all)}잡 x ~8h  (이번 배치 {len(tr_now)}잡)")
        for t in tr_all:
            st = ckpt_step(t)
            mark = "  <- 이번 배치" if t in tr_now else ""
            print(f"   {t:<24} {'resume @' + format(st, ',') if st else '새로 시작':<18}{mark}")
        hrs_t = 8 * len(tr_all) / max(1, len(gpus))
        hrs_e = 2 * len(ev) / max(1, len(gpus))
        print(f"\n[eval] 지금 돌 수 있는 {len(ev)}셀 x ~2 GPU-h -> GPU {len(gpus)}대로 ~{hrs_e:.0f}시간")
        for i, j in enumerate(ev[:15]):
            print(f"   {i + 1:>2}. {j['out']:<28} <- {j['src']:<18} {' '.join(j['flags'])}")
        if len(ev) > 15:
            print(f"   ... 외 {len(ev) - 15}셀")
        if blocked:
            print(f"\n   ※ {blocked}셀은 아직 ckpt 가 없어 대기 중(내 학습분 + 다른 노드 학습분). "
                  f"학습이 끝나는 대로 자동 편입된다.")
        print(f"\n[총 예상] 학습 ~{hrs_t:.0f}h + eval ~{hrs_e:.0f}h "
              f"= ~{hrs_t + hrs_e:.0f}h (대기분 {blocked}셀 별도)")
    return {"node": node, "gpus": gpus, "eval": ev,
            "train": tr_now, "train_all": tr_all, "blocked": blocked}


# -- 실행 --------------------------------------------------------------------
def run_evals(jobs, gpus, n_ep=None, dry=False):
    n_ep = n_ep or N_EP
    gpus = list(gpus)
    cf._check_gpus(gpus)
    todo = [j for j in jobs if not _eval_done(j)]
    print(f"eval {len(jobs)}셀 중 남은 {len(todo)} (완료 {len(jobs) - len(todo)} skip) | GPU {gpus}")
    ng = max(1, len(gpus))
    for i in range(0, len(todo), ng):
        chunk = todo[i:i + ng]
        labeled = []
        for g, j in zip(gpus, chunk):
            try:
                cmd = v23.make_eval_cmd(
                    j["src"], seed=SEED, task=TASK, gpu_id=g, n_episodes=n_ep,
                    select=cf.CKPT_STEP, extra_policy=j["flags"],
                    out_dir=v23.eval_clean_dir(j["out"], SEED, TASK))
                labeled.append((j["out"], cmd))
            except FileNotFoundError as e:
                print("  skip(ckpt 없음):", e)
        if not labeled:
            continue
        print(f"\n===== eval 청크 {i // ng + 1}/{-(-len(todo) // ng)} ({len(labeled)}셀) =====")
        if dry:
            for lab, c in labeled:
                print(f"[{lab}]\n  {c}\n")
        else:
            v23.launch_cmds_live(labeled, log_tag="exp5")


def run_trains(tags, gpus, dry=False):
    gpus = list(gpus)
    cf._check_gpus(gpus)
    jobs = [(t, SEED, TASK) for t in tags]
    if dry:
        for t in tags:
            print(f"[{t}]\n  {v23.make_train_cmd(t, SEED, TASK, gpu_id=gpus[0])}\n")
        return jobs
    return cf.run_training_jobs(jobs, gpus, prefetch_task=TASK)


def run_trains_all(node, gpus=None, max_rounds=20, dry=False):
    """[학습 셀] 이 노드 몫에서 **아직 안 된 학습을 전부** 돈다 (GPU 수만큼 배치로).

    이미 150k 인 태그는 run_training_jobs 가 skip 하고, 중단된 것(PART)은 resume 한다.
    배치마다 인벤토리를 다시 읽으므로 중간에 멈췄다가 재실행해도 이어서 간다.
    """
    node = node.strip().upper()
    gpus = list(gpus) if gpus else v23.available_gpus()
    todo = node_train_tags(node, limit=None)
    if not todo:
        print(f"[{node}] 학습할 것 없음 — 이 노드 몫은 전부 학습돼 있다.")
        return []
    print(f"[{node}] 학습 대기 {len(todo)}잡: {todo}\n"
          f"        GPU {gpus} 로 {TRAIN_JOBS_PER_NODE}잡씩, 잡당 ~8h "
          f"(총 ~{8 * len(todo) / max(1, len(gpus)):.0f}h)")
    done = []
    for rnd in range(1, max_rounds + 1):
        batch = node_train_tags(node)
        if not batch:
            break
        print(f"\n----- [{node}] 학습 배치 {rnd} : {batch} -----")
        run_trains(batch, gpus, dry=dry)
        done += batch
        if dry:
            break
    if not dry:
        left = node_train_tags(node, limit=None)
        print(f"\n[{node}] 학습 종료. 남은 것: {left if left else '없음 ✅'}")
    return done


def run_evals_all(node, gpus=None, max_rounds=6, dry=False):
    """[eval 셀] 이 노드 몫에서 **지금 돌 수 있는 eval 을 전부** 주르륵 돈다.

    한 바퀴 끝나면 인벤토리를 다시 읽는다 — 그 사이 다른 노드가 학습을 끝냈으면
    막혀 있던 셀이 자동으로 편입돼 이어서 돈다. 완료분은 항상 skip.
    """
    node = node.strip().upper()
    gpus = list(gpus) if gpus else v23.available_gpus()
    total = 0
    for rnd in range(1, max_rounds + 1):
        jobs = node_eval_jobs(node)
        if not jobs:
            break
        print(f"\n----- [{node}] eval 바퀴 {rnd} : {len(jobs)}셀 "
              f"(~{2 * len(jobs) / max(1, len(gpus)):.0f}h) -----")
        run_evals(jobs, gpus, dry=dry)
        total += len(jobs)
        if dry:
            return total
    blocked = node_blocked_evals(node)
    if blocked:
        print(f"\n[{node}] 돌 수 있는 건 다 돌았다. 남은 {blocked}셀은 아직 ckpt 가 없다"
              f"(다른 노드가 학습 중). 그 노드가 끝난 뒤 이 셀을 다시 실행하면 이어서 돈다.")
    else:
        print(f"\n[{node}] ✅ 이 노드 몫 eval 전부 완료.")
    return total


def preflight(gpu=0, n_ep=5):
    """stride/TE override 가 lerobot_eval 에서 실제로 먹는지 ~12분짜리로 확인.

    12시간을 걸기 전에 이거 한 번. 여기서 죽으면 override 경로가 막힌 것이고,
    그때는 override 대신 조합별 학습이 필요하다(= 계획 전면 수정).
    """
    Q = eval_queue()
    if not Q:
        print("eval 큐가 비었다 - 학습부터.")
        return
    j = Q[0]
    tag = j["out"] + "_preflight"
    v23.MODEL_DIR_NAMES.setdefault(tag, tag)
    out = v23.eval_clean_dir(tag, SEED, TASK)
    cmd = v23.make_eval_cmd(j["src"], seed=SEED, task=TASK, gpu_id=gpu, n_episodes=n_ep,
                            select=cf.CKPT_STEP, extra_policy=j["flags"], out_dir=out)
    print(f"preflight: {j['out']}  (n_ep={n_ep}/task)\n  {cmd}\n")
    v23.launch_cmds_live([(tag, cmd)], log_tag="preflight")


# -- 집계 --------------------------------------------------------------------
def _sr(out_tag):
    if out_tag not in v23.MODEL_DIR_NAMES:
        return None
    info = v23.eval_clean_dir(out_tag, SEED, TASK) / "eval_info.json"
    if not info.exists():
        return None
    try:
        return json.loads(info.read_text()).get("overall", {}).get("pc_success")
    except Exception:
        return None


_TBL_COLS = ["act", "acm2", "carry", "bimamba_cpoff", "bimamba", "bimos"]


def _mark_best(df, cols):
    have = [c for c in cols if df[c].notna().any()]
    if have and df["act"].notna().any():
        df["best"] = df[have].idxmax(axis=1)
        df["best-act"] = (df[have].max(axis=1) - df["act"]).round(1)
    return df


def stride_table(K=None):
    """★ stride sweep. 행=실행 stride, 열=변형.  act 열보다 높은 칸이 우리가 이기는 지점.

    같은 stride = 같은 추론 호출 횟수이므로, 이 표의 우위는 '동일 비용 우위'다.
    """
    import pandas as pd
    K = MAIN_K if K is None else K
    rows = []
    for s in strides_for(K):
        r = {"K": K, "stride": s}
        for v in _TBL_COLS:
            r[v] = _sr(f"rm_{v}_k{K}_s{s}")
        rows.append(r)
    df = _mark_best(pd.DataFrame(rows), _TBL_COLS)
    print(f"== stride sweep  K={K} (TE off, {N_EP}ep/task, seed{SEED}) ==")
    print(df.to_string(index=False))
    if K == 100:
        print("\n참고(8/25 은지님, n=500, 100k):")
        print("  stride   ACT  ACM2  BiMamba  carry  BiMOS")
        print("     10   42.5  56.8   65.0    60.0   60.4   <- 봉우리, +22.5")
        print("     50   23.6  37.4   31.2    36.8   31.2")
        print("    100   18.4  18.6   21.6    27.8   19.6")
    return df


def k_table(stride=None):
    """메인 그림: 행=chunk K (실행 stride 고정), 열=변형.  ACT 가 무너지는 지점을 본다."""
    import pandas as pd
    stride = MAIN_STRIDE if stride is None else stride
    rows = []
    for K in sorted(K_ALL):
        if stride > K:
            continue
        r = {"K": K, "stride": stride}
        for v in _TBL_COLS:
            r[v] = _sr(f"rm_{v}_k{K}_s{stride}")
        rows.append(r)
    df = _mark_best(pd.DataFrame(rows), _TBL_COLS)
    print(f"== K sweep @ stride={stride} (TE off, {N_EP}ep/task, seed{SEED}) ==")
    print(df.to_string(index=False))
    print("\n참고(기존 측정): K=50 s10 ACT 67.6 / BiMamba 57.8  |  "
          "K=100 s10 ACT 42.5 / BiMamba 65.0")
    print("  -> 크로스오버가 K=50~100 사이. 이 표가 그 곡선을 채운다.")
    return df


def regime_table(K=100):
    """stride_table 의 옛 이름(호환용)."""
    return stride_table(K)


def te_table():
    """TE 축: 행=K, 열=변형.  act 열이 어디서 뒤집히는지 = 크로스오버 K."""
    import pandas as pd
    cols = _TBL_COLS
    rows = []
    for K in sorted(K_ALL):
        r = {"K": K}
        for v in cols:
            r[v] = _sr(f"te_{v}_k{K}")
        rows.append(r)
    df = pd.DataFrame(rows)
    for v in cols[1:]:
        if df[v].notna().any() and df["act"].notna().any():
            df[v + "-act"] = (df[v] - df["act"]).round(1)
    print(f"== TE 축 (stride=1, coeff=0.01, {N_EP}ep/task, seed{SEED}) ==")
    print(df.to_string(index=False))
    print("\n참고 — 8/25 exp3 측정(ak{K}_te, 100ep/task, cpoff 계열):")
    print("  K        10    15    20    50   100   150")
    print("  act+TE  72.5  67.3  61.2  39.7  27.7  21.9")
    print("  bimamba 75.4  75.6  71.4  64.1  49.3  26.8")
    print("  +carry  74.3  76.4  71.9  63.6  50.2  27.1")
    print("100ep 기준 max(ours)=76.4(K=15) vs max(ACT)=72.5(K=10) — W2 승리.")
    print("위의 te_* 500ep 값이 이걸 확정/대체한다. ACT TE-off 최고는 67.6(K=50 s10).")
    return df


def report():
    """stride sweep -> K sweep -> TE 축 순. 앞의 둘이 메인이다."""
    stride_table(MAIN_K)
    print()
    k_table(MAIN_STRIDE)
    print()
    for K in (50, 20, 150):
        try:
            stride_table(K)
            print()
        except Exception as e:
            print(f"(K={K} stride 표 스킵: {e})")
    te_table()
