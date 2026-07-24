"""smooth_metrics.py — v14 trajectory-smoothness metrics (EXP-2 headline).

Computes, from an executed action trajectory (T, A):
  • boundary vs interior JERK contrast  — the v14 headline (chunk-boundary jerk removal).
    jerk = 3rd time-derivative of position (central finite difference). At chunk boundaries
    (t = K, 2K, ...) open-loop chunking spikes jerk; we measure boundary-minus-interior.
  • SPARC (Spectral Arc Length, Balasubramanian et al. 2015) — global smoothness of the
    speed profile (less negative = smoother).
  • per-axis / aggregate jerk RMS.

Self-contained (numpy only). Used by common_v14.jerk_report over recorded eval actions.
"""
import numpy as np


# ── Jerk ──────────────────────────────────────────────────────────────────────

def finite_jerk(traj: np.ndarray) -> np.ndarray:
    """3rd derivative of (T, A) position via central difference (h=1). Returns (T, A); the
    2 boundary samples on each side are 0 (insufficient stencil)."""
    traj = np.asarray(traj, dtype=float)
    j = np.zeros_like(traj)
    if traj.shape[0] >= 5:
        # x'''[t] ≈ (x[t+2] - 2x[t+1] + 2x[t-1] - x[t-2]) / 2
        j[2:-2] = (traj[4:] - 2.0 * traj[3:-1] + 2.0 * traj[1:-3] - traj[:-4]) / 2.0
    return j


def jerk_magnitude(traj: np.ndarray) -> np.ndarray:
    """(T,) L2 norm of per-step jerk across action dims."""
    return np.linalg.norm(finite_jerk(traj), axis=-1)


def boundary_interior_jerk(traj: np.ndarray, chunk_size: int, window: int = 2) -> dict:
    """Mean jerk magnitude at chunk boundaries (t=nK ± window) vs interior.

    Returns {boundary, interior, contrast (b-i), ratio (b/i)}. Higher contrast = worse
    chunking artifact; the v14 method should drive it toward ~0.
    """
    jm = jerk_magnitude(traj)
    T = len(jm)
    valid = np.zeros(T, dtype=bool)
    valid[2:T - 2] = True                                  # finite-diff valid region
    boundary = np.zeros(T, dtype=bool)
    for b in range(chunk_size, T, chunk_size):
        boundary[max(0, b - window):min(T, b + window + 1)] = True
    bm = jm[boundary & valid]
    im = jm[(~boundary) & valid]
    b_mean = float(bm.mean()) if bm.size else 0.0
    i_mean = float(im.mean()) if im.size else 0.0
    return {"boundary": b_mean, "interior": i_mean,
            "contrast": b_mean - i_mean,
            "ratio": (b_mean / i_mean) if i_mean > 1e-9 else float("nan")}


def jerk_rms(traj: np.ndarray) -> float:
    """Overall RMS jerk magnitude (valid region)."""
    jm = jerk_magnitude(traj)
    jm = jm[2:len(jm) - 2] if len(jm) > 4 else jm
    return float(np.sqrt(np.mean(jm ** 2))) if jm.size else 0.0


# ── SPARC (spectral arc length) ───────────────────────────────────────────────

def sparc(speed: np.ndarray, fs: float = 50.0, padlevel: int = 4,
          fc: float = 10.0, amp_th: float = 0.05) -> float:
    """Spectral Arc Length smoothness of a 1-D speed profile (Balasubramanian 2015).
    Less negative (closer to 0) = smoother. Returns 0.0 for degenerate input."""
    speed = np.asarray(speed, dtype=float)
    if speed.size < 4 or np.allclose(speed, 0.0):
        return 0.0
    nfft = int(2 ** (np.ceil(np.log2(speed.size)) + padlevel))
    f = np.arange(0, fs, fs / nfft)
    Mf = np.abs(np.fft.fft(speed, nfft))
    Mf = Mf / (Mf.max() + 1e-12)
    sel = f <= fc
    f_s, Mf_s = f[sel], Mf[sel]
    inx = np.where(Mf_s >= amp_th)[0]
    if inx.size < 2:
        return 0.0
    f_s = f_s[inx[0]:inx[-1] + 1]
    Mf_s = Mf_s[inx[0]:inx[-1] + 1]
    df = np.diff(f_s) / (f_s[-1] - f_s[0] + 1e-12)
    dM = np.diff(Mf_s)
    return float(-np.sum(np.sqrt(df ** 2 + dM ** 2)))


def episode_sparc(traj: np.ndarray, fs: float = 50.0) -> float:
    """SPARC of the speed profile (||velocity||) of a (T, A) action trajectory."""
    traj = np.asarray(traj, dtype=float)
    if traj.shape[0] < 3:
        return 0.0
    speed = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    return sparc(speed, fs)


# ── LDJ (log dimensionless jerk) ──────────────────────────────────────────────

def ldj(traj: np.ndarray, fs: float = 50.0) -> float:
    """Log Dimensionless Jerk of a (T, A) action trajectory (Balasubramanian 2015).

    LDJ = -ln( (T^5 / L^2) * integral(||jerk||^2 dt) ),  scale-/duration-invariant.
    Larger (less negative) = smoother. Multi-dim: jerk energy summed over action dims,
    L^2 = peak speed^2 (dimensionless normalizer). Returns 0.0 for degenerate input.
    """
    traj = np.asarray(traj, dtype=float)
    T = traj.shape[0]
    if T < 5:
        return 0.0
    dt = 1.0 / fs
    duration = (T - 1) * dt
    speed = np.linalg.norm(np.diff(traj, axis=0), axis=-1)      # (T-1,)
    peak = float(speed.max())
    if peak < 1e-9 or duration < 1e-9:
        return 0.0
    jerk = finite_jerk(traj) * (fs ** 3)                        # per-step diff -> physical units
    jerk_sq = np.sum(jerk ** 2, axis=-1)                        # (T,) energy over dims
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))  # numpy 2.x 는 trapezoid
    integral = _trapz(jerk_sq, dx=dt)
    dimensionless = (duration ** 5 / peak ** 2) * integral
    if dimensionless <= 0:
        return 0.0
    return float(-np.log(dimensionless))


# ── Sign-flip count (직접 방향 반전 = 떨림) ────────────────────────────────────

def sign_flips(traj: np.ndarray, per_step: bool = True) -> float:
    """Number of velocity sign reversals, summed over action dims.

    velocity = diff(traj); a flip is where consecutive velocity signs differ (a
    direction reversal — the visible 'shaking'). per_step=True normalizes by length
    (flips per step) so episodes of different length are comparable.
    """
    traj = np.asarray(traj, dtype=float)
    if traj.shape[0] < 3:
        return 0.0
    vel = np.diff(traj, axis=0)                                 # (T-1, A)
    sgn = np.sign(vel)
    sgn[sgn == 0] = 0                                           # zeros don't count as a flip
    flips = ((sgn[1:] * sgn[:-1]) < 0).sum()                   # sign change per dim
    flips = float(flips)
    if per_step:
        n = max(vel.shape[0] - 1, 1)
        return flips / n
    return flips


# ── Episode summary ───────────────────────────────────────────────────────────

def trajectory_smoothness(traj: np.ndarray, chunk_size: int, fs: float = 50.0,
                          window: int = 2) -> dict:
    """All v14 smoothness metrics for one executed action trajectory (T, A)."""
    bij = boundary_interior_jerk(traj, chunk_size, window)
    return {
        "boundary_jerk": bij["boundary"],
        "interior_jerk": bij["interior"],
        "boundary_jerk_contrast": bij["contrast"],   # ★ headline
        "boundary_jerk_ratio": bij["ratio"],
        "jerk_rms": jerk_rms(traj),                  # ↓ smaller = smoother
        "ldj": ldj(traj, fs),                        # ↑ larger (less negative) = smoother
        "sparc": episode_sparc(traj, fs),            # ↑ less negative = smoother
        "sign_flips": sign_flips(traj),              # ↓ fewer = smoother (per step)
        "n_steps": int(traj.shape[0]),
    }


def aggregate_smoothness(trajs: list[np.ndarray], chunk_size: int, fs: float = 50.0,
                         window: int = 2) -> dict:
    """Mean ± std of each metric over a list of episode trajectories."""
    if not trajs:
        return {}
    per = [trajectory_smoothness(t, chunk_size, fs, window) for t in trajs]
    keys = [k for k in per[0] if k != "n_steps"]
    out = {"n_episodes": len(per)}
    for k in keys:
        vals = np.array([p[k] for p in per if np.isfinite(p[k])], dtype=float)
        out[f"{k}_mean"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{k}_std"] = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    return out
