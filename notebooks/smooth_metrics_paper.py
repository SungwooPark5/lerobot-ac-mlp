"""smooth_metrics_paper.py — 팀원(은지)의 aloha 스무스니스 스크립트와 **동일한 계산식**.

aloha 와 libero 숫자를 비교 가능하게 하려면 두 task 가 같은 지표 코드를 써야 한다.
이 파일은 은지님이 보낸 aloha 스크립트의 지표 함수들을 **그대로** 옮긴 것이다(파일 IO/main 제외).
우리 기존 smooth_metrics.py 와는 계산식이 전부 다르므로 논문 표는 이 모듈을 쓴다.

task 별로 다른 것은 딱 두 개(방법 아님):
  · fs  : SPARC 의 주파수축에만 영향. aloha=50, libero=30(fps). 나머지 지표는 fs 무관.
  · boundary_stride : 청크 경계 위치. 두 task 모두 100(하드스위치 기준으로 공정 비교).

지표 방향:
  jerk_rms ↓ · boundary/interior jerk RMS ↓ · B/I ratio →1 · SPARC →0 · ldj_cost ↓ · sign_flip_rate ↓
"""
import numpy as np

# 기본 상수 (은지 스크립트와 동일). libero 는 fs=30 으로 넘겨 쓸 것.
CONTROL_HZ = 50.0
CHUNK_SIZE = 100
OVERLAP_SIZE = 10
BOUNDARY_RADIUS = 2.0


def safe_norm(x, axis=-1):
    return np.linalg.norm(x, axis=axis)


def safe_rms(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    return np.sqrt(np.mean(x ** 2))


# ── Jerk: 3차 forward 차분 ────────────────────────────────────────────────────
def third_forward_difference(actions):
    """Δ³a_t = a_{t+3} − 3a_{t+2} + 3a_{t+1} − a_t.  입력 [T,A] → 출력 [T-3,A]."""
    return np.diff(actions, n=3, axis=0)


def jerk_sample_centers(num_action_steps):
    """Δ³a_t 는 t..t+3 을 쓰므로 시간 중심 = t+1.5."""
    num = num_action_steps - 3
    if num <= 0:
        return np.asarray([], dtype=np.float64)
    return np.arange(num, dtype=np.float64) + 1.5


def chunk_transition_steps(num_action_steps, stride):
    """청크 전환 스텝(첫 청크는 0 에서 시작 → 첫 전환 = stride)."""
    return np.arange(stride, num_action_steps, stride, dtype=np.float64)


def nearest_transition_offsets(sample_centers, transitions):
    """각 jerk 샘플 중심에서 가장 가까운 전환까지의 부호 거리."""
    if len(sample_centers) == 0 or len(transitions) == 0:
        return np.full(len(sample_centers), np.nan, dtype=np.float64)
    distances = sample_centers[:, None] - transitions[None, :]
    nearest = np.argmin(np.abs(distances), axis=1)
    return distances[np.arange(len(sample_centers)), nearest]


def boundary_and_interior_jerk(actions, boundary_stride, radius=BOUNDARY_RADIUS, eps=1e-12):
    """경계(전환 ±radius 스텝) vs 내부 jerk RMS. 은지 스크립트 그대로."""
    actions = np.asarray(actions, dtype=np.float64)
    empty = {"boundary_jerk_rms": np.nan, "interior_jerk_rms": np.nan,
             "boundary_interior_difference": np.nan, "boundary_interior_ratio": np.nan,
             "boundary_jerk_mean": np.nan, "interior_jerk_mean": np.nan,
             "n_boundaries": 0, "n_boundary_samples": 0, "n_interior_samples": 0}
    if actions.ndim != 2 or actions.shape[0] < 4:
        return empty
    jerk = third_forward_difference(actions)
    jerk_l2 = safe_norm(jerk, axis=-1)
    centers = jerk_sample_centers(actions.shape[0])
    transitions = chunk_transition_steps(actions.shape[0], boundary_stride)
    if len(transitions) == 0:
        return empty
    offsets = nearest_transition_offsets(centers, transitions)
    boundary_mask = np.abs(offsets) <= radius
    interior_mask = ~boundary_mask
    bvals = jerk_l2[boundary_mask]
    ivals = jerk_l2[interior_mask]
    if len(bvals) == 0 or len(ivals) == 0:
        r = empty.copy()
        r.update({"n_boundaries": len(transitions), "n_boundary_samples": len(bvals),
                  "n_interior_samples": len(ivals)})
        return r
    b_rms = safe_rms(bvals)
    i_rms = safe_rms(ivals)
    return {"boundary_jerk_rms": b_rms, "interior_jerk_rms": i_rms,
            "boundary_interior_difference": b_rms - i_rms,
            "boundary_interior_ratio": b_rms / (i_rms + eps),
            "boundary_jerk_mean": np.mean(bvals), "interior_jerk_mean": np.mean(ivals),
            "n_boundaries": len(transitions), "n_boundary_samples": len(bvals),
            "n_interior_samples": len(ivals)}


# ── SPARC (speed profile) ─────────────────────────────────────────────────────
def sparc_1d(signal, fs, padlevel=4, fc=10.0, amp_th=0.05, eps=1e-12):
    """Spectral Arc Length (0 에 가까울수록 매끄러움). 은지 스크립트 그대로."""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError(f"SPARC signal must be 1-D, got {signal.shape}")
    n = len(signal)
    if n < 8:
        return np.nan
    if fs <= 0:
        raise ValueError(f"Sampling frequency must be positive, got {fs}")
    signal = signal - np.mean(signal)
    if np.max(np.abs(signal)) < eps:
        return 0.0
    base_nfft = 2 ** int(np.ceil(np.log2(n)))
    nfft = base_nfft * (2 ** padlevel)
    frequencies = np.fft.rfftfreq(nfft, d=1.0 / fs)
    magnitude = np.abs(np.fft.rfft(signal, n=nfft))
    max_mag = np.max(magnitude)
    if max_mag < eps:
        return 0.0
    magnitude = magnitude / (max_mag + eps)
    cutoff = min(fc, fs / 2.0)
    mask = frequencies <= cutoff
    frequencies = frequencies[mask]
    magnitude = magnitude[mask]
    above = np.where(magnitude >= amp_th)[0]
    if len(above) < 2:
        return 0.0
    last = above[-1] + 1
    frequencies = frequencies[:last]
    magnitude = magnitude[:last]
    if len(frequencies) < 2:
        return 0.0
    nf = frequencies / (frequencies[-1] + eps)
    df = np.diff(nf)
    dM = np.diff(magnitude)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))


def sparc_from_speed_profile(actions, fs=CONTROL_HZ, fc=10.0, amp_th=0.05):
    """s_t = ||a_{t+1} − a_t||₂ 의 SPARC. 은지 스크립트 그대로(T>=9 필요)."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 9:
        return np.nan
    speed = safe_norm(np.diff(actions, axis=0), axis=-1)
    return sparc_1d(speed, fs=fs, padlevel=4, fc=fc, amp_th=amp_th)


# ── LDJ (path-length 정규화, step 단위) ────────────────────────────────────────
def log_dimensionless_jerk_style(actions, eps=1e-12):
    """ldj: 클수록(덜 음수) 매끄러움 / ldj_cost=−ldj: 작을수록 매끄러움. 은지 스크립트 그대로."""
    actions = np.asarray(actions, dtype=np.float64)
    n = actions.shape[0]
    if n < 4:
        return np.nan
    velocity = np.diff(actions, axis=0)
    speed = safe_norm(velocity, axis=-1)
    amplitude = np.sum(speed) + eps                 # path length
    jerk = third_forward_difference(actions)
    jerk_sq_sum = np.sum(jerk ** 2)
    duration = float(n - 1)                          # steps (초 아님)
    value = (duration ** 5 / (amplitude ** 2 + eps)) * jerk_sq_sum + eps
    return -np.log(value)


def sign_flip_rate(actions):
    """1차차분 부호변화 빈도 = 시간·차원 평균 rate. 은지 스크립트 그대로."""
    diff = np.diff(actions, axis=0)
    if len(diff) < 2:
        return np.nan
    flips = (diff[1:] * diff[:-1]) < 0
    return np.mean(flips)


# ── 한 궤적의 전체 지표 ────────────────────────────────────────────────────────
def metrics_for_trajectory(actions, boundary_stride=CHUNK_SIZE, fs=CONTROL_HZ):
    """은지 스크립트 metrics_for_trajectory 와 동일(fs 만 인자화)."""
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[0] < 4:
        return None
    d1 = np.diff(actions, n=1, axis=0)
    d2 = np.diff(actions, n=2, axis=0)
    jerk = third_forward_difference(actions)
    jerk_l2 = safe_norm(jerk, axis=-1)
    ldj = log_dimensionless_jerk_style(actions)
    m = {
        "T": actions.shape[0], "D": actions.shape[1],
        "diff_abs_mean": np.mean(np.abs(d1)),
        "accel_abs_mean": np.mean(np.abs(d2)),
        "jerk_rms": np.sqrt(np.mean(jerk ** 2)),
        "jerk_l2_mean": np.mean(jerk_l2),
        "jerk_l2_rms": safe_rms(jerk_l2),
        "jerk_l2_p95": np.percentile(jerk_l2, 95),
        "sparc": sparc_from_speed_profile(actions, fs=fs, fc=10.0, amp_th=0.05),
        "ldj": ldj,
        "ldj_cost": -ldj if np.isfinite(ldj) else np.nan,
        "sign_flip_rate": sign_flip_rate(actions),
    }
    m.update(boundary_and_interior_jerk(actions, boundary_stride=boundary_stride, radius=BOUNDARY_RADIUS))
    return m


# ── 여러 궤적 집계 (우리 노트북용 헬퍼) ────────────────────────────────────────
# 논문 표 핵심 지표만 mean/std 로. 방향: jerk_rms↓ boundary/interior↓ ratio→1 sparc→0 ldj_cost↓ signflip↓
REPORT_KEYS = ["jerk_rms", "boundary_jerk_rms", "interior_jerk_rms",
               "boundary_interior_difference", "boundary_interior_ratio",
               "sparc", "ldj_cost", "sign_flip_rate"]


def aggregate_paper(trajs, boundary_stride=CHUNK_SIZE, fs=CONTROL_HZ):
    """궤적 리스트 → 핵심 지표 mean/std. libero 는 fs=30 으로 호출."""
    per = [metrics_for_trajectory(t, boundary_stride=boundary_stride, fs=fs) for t in trajs]
    per = [p for p in per if p is not None]
    out = {"n_traj": len(per)}
    for k in REPORT_KEYS:
        vals = np.array([p[k] for p in per if p.get(k) is not None and np.isfinite(p[k])], dtype=float)
        out[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return out
