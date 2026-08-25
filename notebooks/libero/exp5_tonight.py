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
    "carry":         ("acm2_carry",   "acm2_sscp_literal",         True,  "carry only"),
    "bimamba":       ("bimamba_pure", "acm2_sscp_literal_bimamba", False, "BiMamba only (pure)"),
    "bimos":         ("bimamba",      "acm2_sscp_literal_bimamba", True,  "BiMamba + carry"),
    # eval 전용 프록시: bimos ckpt 에 carry 만 꺼서 본다. 학습이 chunk-pair 라 순수 아님.
    "bimamba_cpoff": ("bimamba",      None,                        None,  "BiMamba (chunk-pair trained!)"),
}
#  변형 -> use_chunk_pairs (학습 시). 순수 BiMamba 는 False 여야 한다.
USE_CHUNK_PAIRS = {"act": False, "carry": True, "bimamba": False, "bimos": True}
REUSE_CKPT_OF = {"bimamba_cpoff": "bimos"}      # 학습 안 함 -> 이 변형의 ckpt 사용
TRAINABLE = ("act", "carry", "bimamba", "bimos")

# K=10/15 는 8/25 추가. 우리 최고 후보(BiMamba+TE 75.4/75.6)가 거기 있는데 스윕에서
# 빠져 있었다 — 이길 가능성이 있는 유일한 절대값 지점을 안 재고 있었던 것.
K_ALL = [10, 15, 20, 50, 100, 150]


def _carry_flag(variant, on):
    """정책별로 유효한 플래그만. ACT 에는 sscp 플래그 자체가 없다."""
    if variant == "act":
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


def strides_for(K):
    return [s for s in (1, 10, 15, 20, 25, 50, 75, 100, 150) if s <= K]


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
_ALL_V = ("act", "bimamba", "bimamba_cpoff", "bimos", "carry")


def _all_jobs():
    """우선순위 = 이 리스트 순서. 앞쪽이 W2(토너먼트) 판정 셀이다."""
    Q = []
    # P1 -- TE 토너먼트 축, 500ep 확정 런. act+TE 는 8/25 exp3(100ep)로 이미 측정됨:
    #   72.5/67.3/61.2/39.7/27.7/21.9 (K=10..150) — 전 K 에서 우리가 위.
    #   K 순서 = 마진이 얇아 500ep 확정이 급한 순: 10(+2.9), 15(+9.1이지만 max 자리),
    #   20(+10.2), 50, 100, 150. te_bimamba_*(순수)는 pure ckpt 가 생기는 대로 편입.
    for K in (10, 15, 20, 50, 100, 150):
        vs = ("act", "bimamba") if K in (10, 15) else _ALL_V
        for v in vs:
            Q.append(_te_job(v, K))
    # P2 -- K=50 단거리 레짐 행: ACT 최고점(67.6, s10)과 정면 승부 + carry 모순 해소
    #   (8/17 수정표: carry 69.5 > ACT 61.1  vs  8/18 슬라이드: ACT 67.6 > carry 64.6
    #    — 같은 세팅인데 두 표가 다르고, 슬라이드 stride 행들은 aloha 오염 가능성까지
    #    확인됐다(설계근거 4). UBAI 재측정 값만 논문에 쓴다.)
    for s in (10, 25):
        for v in _ALL_V:
            Q.append(_rm_job(v, 50, s))
    # P3 -- K=100 레짐 맵, 긴 stride 우선 (W1: 긴 chunk 에서 ACT 붕괴 + 효율 주장)
    for s in (75, 50, 25):
        for v in _ALL_V:
            Q.append(_rm_job(v, 100, s))
    # P4 -- 나머지 K=50 / K=20 행
    for K, ss in ((50, (50,)), (20, (10, 20))):
        for s in ss:
            for v in _ALL_V:
                Q.append(_rm_job(v, K, s))
    # P5 -- 잔여 (여유 있을 때)
    for K, ss in ((100, (10, 100)), (150, (150, 100, 50))):
        for s in ss:
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
    ("act",     10),    # 이미 OK — 자동 이탈
    ("act",     15),    # 이미 OK — 자동 이탈
    ("bimamba", 10),    # 순수 BiMamba K=10 — 75.4 를 UBAI/순수판으로 재측정
    ("bimamba", 15),    # 순수 BiMamba K=15 — 76.4(전체 최고) 자리의 순수판
]

# 학습 우선순위 -- critical 이 항상 앞. 이미 OK 인 태그는 train_queue() 에서 빠진다.
TRAIN_PRIORITY = CRITICAL_TRAIN_JOBS + [
    ("bimos",   10),    # 오염판 K=10/15 — 은지님 75.4/75.6 재현 대조용 (2차 배치)
    ("bimos",   15),
    ("bimamba", 20),    # 순수 K=20 (cpoff 프록시 71.4 의 순수판)
    ("carry",   50),    # 40k 에서 중단 — resume
    ("carry",   20),
    ("bimamba", 100),   # 이하는 8/24 밤에 대부분 완료 — 남은 것만 잡힌다
    ("act",    150),
    ("act",     20),
    ("bimamba", 50),
    ("bimos",  150),
    ("carry",  150),
    ("bimamba", 150),
    ("act",     50),
    ("bimos",   20),
    ("bimos",   50),
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


def node_eval_jobs(node):
    """eval 노드들이 우선순위 큐를 번갈아(round-robin) 나눠 갖는다 -> 둘 다 상위부터 돈다."""
    node = node.strip().upper()
    ev_nodes = _nodes_with("eval")
    if node not in ev_nodes:
        return []
    i, k = ev_nodes.index(node), len(ev_nodes)
    return eval_queue()[i::k]


def node_train_tags(node):
    """학습 노드들이 학습 큐를 2잡씩 순서대로 나눠 갖는다. eval 노드는 폴백 슬라이스."""
    node = node.strip().upper()
    q = train_queue()
    tr_nodes = _nodes_with("train")
    if node in tr_nodes:
        i = tr_nodes.index(node) * TRAIN_JOBS_PER_NODE
        return q[i:i + TRAIN_JOBS_PER_NODE]
    # eval 노드의 폴백: 학습 노드들이 가져간 뒤쪽에서 집는다(겹침 없음)
    base = len(tr_nodes) * TRAIN_JOBS_PER_NODE
    ev_nodes = _nodes_with("eval")
    i = base + (ev_nodes.index(node) if node in ev_nodes else 0) * TRAIN_JOBS_PER_NODE
    return q[i:i + TRAIN_JOBS_PER_NODE]


def plan(node, gpus=None, verbose=True):
    """이 노드가 오늘 밤 할 일. 실행 전에 반드시 눈으로 확인할 것."""
    node = node.strip().upper()
    gpus = list(gpus) if gpus else v23.available_gpus()
    ev, tr = node_eval_jobs(node), node_train_tags(node)
    if verbose:
        print(f"===== NODE {node} ({role_of(node)})  GPU {gpus} =====")
        if role_of(node) == "train":
            print(f"\n[학습] {len(tr)}잡 x ~15h  (밤새 점유, 내일 오후 완료)")
            for t in tr:
                st = ckpt_step(t)
                print(f"   {t:<24} {'resume @' + format(st, ',') if st else '새로 시작'}")
            print("\n  -- 전체 학습 큐 --")
            train_queue(verbose=True)
        else:
            hrs = 2 * len(ev) / max(1, len(gpus))
            print(f"\n[eval] {len(ev)}셀 x ~2 GPU-h -> GPU {len(gpus)}대로 ~{hrs:.0f}시간")
            for i, j in enumerate(ev):
                mark = "  <- 오늘 밤" if i < EVAL_CELLS_PER_NODE else ""
                print(f"   {i + 1:>2}. {j['out']:<28} <- {j['src']:<18} "
                      f"{' '.join(j['flags'])}{mark}")
            if hrs > 12:
                print(f"\n  ※ 12시간을 넘는다. 앞에서부터 도니까 아침에 같은 셀을 다시 실행하면"
                      f" 완료분은 skip 되고 남은 {len(ev) - EVAL_CELLS_PER_NODE}셀부터 이어서 돈다.")
            if tr:
                print(f"\n[폴백 학습] eval 큐가 마르면(만): {tr}")
    return {"node": node, "gpus": gpus, "eval": ev, "train": tr}


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


def regime_table(K=100):
    """행=stride, 열=변형, 셀=SR.  act 열보다 높은 칸이 'ACT 가 무너지는 지점'."""
    import pandas as pd
    cols = ["act", "carry", "bimamba_cpoff", "bimamba", "bimos"]
    rows = []
    for s in strides_for(K):
        r = {"K": K, "stride": s}
        for v in cols:
            r[v] = _sr(f"rm_{v}_k{K}_s{s}")
        rows.append(r)
    df = pd.DataFrame(rows)
    have = [c for c in cols if df[c].notna().any()]
    if have and df["act"].notna().any():
        df["best"] = df[have].idxmax(axis=1)
        df["best-act"] = (df[have].max(axis=1) - df["act"]).round(1)
    print(f"== 레짐 맵 K={K} (TE off, {N_EP}ep/task, seed{SEED}) ==")
    print(df.to_string(index=False))
    return df


def te_table():
    """TE 축: 행=K, 열=변형.  act 열이 어디서 뒤집히는지 = 크로스오버 K."""
    import pandas as pd
    cols = ["act", "bimamba_cpoff", "bimamba", "bimos", "carry"]
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
    te_table()
    print()
    for K in (100, 50, 150, 20):
        try:
            regime_table(K)
            print()
        except Exception as e:
            print(f"(K={K} 표 스킵: {e})")
