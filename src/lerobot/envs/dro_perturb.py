"""dro_perturb.py — DRO disturbance benchmark: disturbance-injecting gym_aloha variants.

Importing this module registers, for each base task (TransferCube, Insertion):
    gym_aloha/AlohaTransferCubeDRO-v0
    gym_aloha/AlohaInsertionDRO-v0
each = the base gym_aloha env (inner TimeLimit stripped) + a DROWrapper that injects one
disturbance per episode, at a per-episode random timestep, with a per-episode rng derived
from the env reset seed (reproducible, and different across vectorized sub-envs).

Disturbance types (DRO_TYPE):
    none      passthrough — clean baseline through the identical wrapper stack
    push      one-shot qvel impulse of L2 norm `mag` on the ROBOT dofs (arm bumped)
    teleport  shift one random OBJECT free joint's xy by `mag` meters (object slips/moved)
    obsnoise  additive gaussian noise (sigma=`mag`) on obs["agent_pos"] for DRO_OBS_STEPS
              consecutive steps (sensor disturbance)
    mix       per EPISODE, pick one of push/teleport/obsnoise uniformly (mag from
              DRO_LEVEL for the chosen type) — used for training-time robust eval

Runtime toggle: `dro_set_enabled(bool)` (reachable through the wrapper chain, e.g.
vec_env.call("dro_set_enabled", False)) switches the wrapper clean/disturbed at the
NEXT reset — lerobot_train uses it to split one eval budget into clean + disturbed
halves on the same env instances (same seeds → paired comparison).

Robot vs object dofs are auto-detected from the MuJoCo model: free joints (mjJNT_FREE)
are objects, everything else is the robot. No per-task hardcoding.

Env vars:
    DRO_TYPE       none|push|teleport|obsnoise            (default none)
    DRO_LEVEL      1|2|3 → magnitude from DRO_LEVELS       (default 2)
    DRO_MAG        explicit magnitude, overrides DRO_LEVEL (optional)
    DRO_STEP       fixed disturbance step; -1 = random     (default -1)
    DRO_TMIN/TMAX  random-step window in absolute steps    (default 80 / 240, for T=400)
    DRO_OBS_STEPS  obsnoise duration in steps              (default 30)
    DRO_SEED       base seed mixed into the per-episode rng (default 0)
    DRO_LOG_DIR    if set, append one jsonl line per episode with the realized
                   disturbance (type, step, mag, reset seed) for recovery analysis

Used via lerobot_eval_dro.py so the WHOLE lerobot_eval pipeline (correct normalization,
official SR, videos) runs unchanged — only the env injects the disturbance.
"""

import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np

import gym_aloha  # noqa: F401  (registers the base gym_aloha/* envs)

# Calibrate on the cluster with 02_calibrate: level 2 should roughly halve ACT's SR.
DRO_LEVELS = {
    "push": {1: 0.6, 2: 1.2, 3: 2.4},  # qvel impulse L2 norm over robot dofs [rad/s]
    "teleport": {1: 0.02, 2: 0.04, 3: 0.08},  # object xy shift [m]
    "obsnoise": {1: 0.01, 2: 0.03, 3: 0.08},  # gaussian sigma on agent_pos [rad]
}

_MJ_JNT_FREE = 0  # mujoco.mjtJoint.mjJNT_FREE


def _find_physics(env):
    """Walk the gym_aloha unwrapped env to its dm_control Physics (has .model/.data/.forward)."""
    u = env.unwrapped
    for attr in ("_env", "task", "sim", "physics"):
        phys = getattr(u, attr, None)
        phys = getattr(phys, "physics", phys)
        if phys is not None and hasattr(phys, "data") and hasattr(phys.data, "qpos"):
            return phys
    if hasattr(u, "data") and hasattr(u.data, "qpos"):
        return u  # raw mujoco binding: .model/.data present, forward via mj_forward
    return None


def _forward(phys):
    if hasattr(phys, "forward"):
        phys.forward()
    else:
        import mujoco

        mujoco.mj_forward(phys.model, phys.data)


def _joint_layout(phys):
    """Split dofs/qpos addresses into robot joints and object free joints.

    Returns (robot_dof_idx, free_joints) where free_joints is a list of qpos start
    addresses (each free joint owns qpos[adr:adr+7] = xyz + quat).
    """
    model = phys.model
    jnt_type = np.asarray(model.jnt_type).ravel()
    jnt_qposadr = np.asarray(model.jnt_qposadr).ravel()
    jnt_dofadr = np.asarray(model.jnt_dofadr).ravel()
    nv = int(model.nv)

    free_qposadr = [int(jnt_qposadr[j]) for j in range(len(jnt_type)) if jnt_type[j] == _MJ_JNT_FREE]
    free_dofs = set()
    for j in range(len(jnt_type)):
        if jnt_type[j] == _MJ_JNT_FREE:
            free_dofs.update(range(int(jnt_dofadr[j]), int(jnt_dofadr[j]) + 6))
    robot_dof_idx = np.array([d for d in range(nv) if d not in free_dofs], dtype=int)
    return robot_dof_idx, free_qposadr


class DROWrapper(gym.Wrapper):
    """Inject one disturbance per episode at a per-episode random step (env-var driven)."""

    def __init__(self, env):
        super().__init__(env)
        self.dtype = os.environ.get("DRO_TYPE", "none").lower()
        if self.dtype not in ("none", "push", "teleport", "obsnoise", "mix"):
            raise ValueError(f"DRO_TYPE={self.dtype!r} not in none|push|teleport|obsnoise|mix")
        self.level = int(os.environ.get("DRO_LEVEL", "2"))
        self._mag_override = os.environ.get("DRO_MAG")
        self._enabled = True
        self.fixed_step = int(os.environ.get("DRO_STEP", "-1"))
        self.tmin = int(os.environ.get("DRO_TMIN", "80"))
        self.tmax = int(os.environ.get("DRO_TMAX", "240"))
        self.obs_steps = int(os.environ.get("DRO_OBS_STEPS", "30"))
        self.base_seed = int(os.environ.get("DRO_SEED", "0"))
        self.log_dir = os.environ.get("DRO_LOG_DIR")

        self._rng = np.random.default_rng(self.base_seed)
        self._episode = 0
        self._n = 0
        self._t_dist = -1
        self._runtime_type = None  # dro_set_mode override (None → env-var config)
        self._ep_type = "none"   # per-episode resolved type (mix picks one at reset)
        self._ep_mag = 0.0
        self._applied = False
        self._reset_seed = None
        self._warned = False

    def dro_set_enabled(self, enabled: bool) -> bool:
        """Runtime clean/disturbed toggle — takes effect at the NEXT reset."""
        self._enabled = bool(enabled)
        return self._enabled

    def dro_set_mode(self, mode) -> str:
        """Runtime type override (takes effect at the NEXT reset).

        mode: 'none'|'push'|'teleport'|'obsnoise'|'mix' — overrides DRO_TYPE;
              None restores the env-var config. lerobot_train uses this to split one
              eval budget into clean + per-type quarters on the same envs/seeds.
        """
        if mode is not None and mode not in ("none", "push", "teleport", "obsnoise", "mix"):
            raise ValueError(f"dro_set_mode({mode!r}) invalid")
        self._runtime_type = mode
        return mode if mode is not None else self.dtype

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, **kwargs):
        self._n = 0
        self._applied = False
        self._reset_seed = seed
        # Per-episode rng: reproducible from (base seed, reset seed | episode counter).
        # Reset seed comes per sub-env from lerobot's rollout (env.reset(seed=seeds)),
        # so vectorized sub-envs get DIFFERENT disturbance draws — unlike v9's shared rng(0).
        mix = seed if seed is not None else self._episode
        self._rng = np.random.default_rng([self.base_seed, int(mix)])
        self._episode += 1

        cfg_type = self._runtime_type if self._runtime_type is not None else self.dtype
        if cfg_type == "none" or not self._enabled:
            self._ep_type, self._ep_mag, self._t_dist = "none", 0.0, -1
        else:
            if cfg_type == "mix":
                self._ep_type = str(self._rng.choice(["push", "teleport", "obsnoise"]))
            else:
                self._ep_type = cfg_type
            if self._mag_override is not None:
                self._ep_mag = float(self._mag_override)
            else:
                self._ep_mag = DRO_LEVELS[self._ep_type][self.level]
            if self.fixed_step >= 0:
                self._t_dist = self.fixed_step
            else:
                self._t_dist = int(self._rng.integers(self.tmin, self.tmax + 1))
        return self.env.reset(seed=seed, **kwargs)

    # ------------------------------------------------------------------- step
    def step(self, action):
        # Physics disturbances fire BEFORE the sim step at t_dist so the policy first
        # sees the disturbed state in the observation returned for this step.
        if self._ep_type in ("push", "teleport") and not self._applied and self._n == self._t_dist:
            ok = self._apply_physics_disturbance()
            self._applied = True
            self._log_episode(ok)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._ep_type == "obsnoise" and self._t_dist >= 0 and self._n >= self._t_dist:
            if not self._applied:
                self._applied = True
                self._log_episode(True)
            if self._n < self._t_dist + self.obs_steps:
                obs = self._noisy_obs(obs)

        info["dro_type"] = self._ep_type
        info["dro_step"] = self._t_dist
        info["dro_active"] = bool(self._applied and self._n >= self._t_dist)
        self._n += 1
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------ disturbances
    def _apply_physics_disturbance(self) -> bool:
        phys = _find_physics(self.env)
        if phys is None:
            if not self._warned:
                print("[DROWrapper] WARN: physics not found — no disturbance applied")
                self._warned = True
            return False
        try:
            robot_dofs, free_qposadr = _joint_layout(phys)
            if self._ep_type == "push":
                if len(robot_dofs) == 0:
                    raise RuntimeError("no robot dofs detected")
                d = self._rng.standard_normal(len(robot_dofs))
                d *= self._ep_mag / (np.linalg.norm(d) + 1e-9)
                qvel = np.asarray(phys.data.qvel)
                qvel[robot_dofs] += d
                phys.data.qvel[:] = qvel
            elif self._ep_type == "teleport":
                if len(free_qposadr) == 0:
                    raise RuntimeError("no object free joint detected")
                adr = int(self._rng.choice(free_qposadr))
                theta = self._rng.uniform(0.0, 2.0 * np.pi)
                phys.data.qpos[adr] += self._ep_mag * np.cos(theta)
                phys.data.qpos[adr + 1] += self._ep_mag * np.sin(theta)
            _forward(phys)
            return True
        except Exception as e:  # noqa: BLE001 — a failed injection must not kill the eval
            if not self._warned:
                print(f"[DROWrapper] WARN: disturbance failed ({e}) — none applied")
                self._warned = True
            return False

    def _noisy_obs(self, obs):
        if isinstance(obs, dict) and "agent_pos" in obs:
            noise = self._rng.normal(0.0, self._ep_mag, size=np.asarray(obs["agent_pos"]).shape)
            obs = dict(obs)
            obs["agent_pos"] = obs["agent_pos"] + noise.astype(obs["agent_pos"].dtype)
        elif not self._warned:
            print("[DROWrapper] WARN: obs has no 'agent_pos' — obsnoise not applied")
            self._warned = True
        return obs

    # ---------------------------------------------------------------- logging
    def _log_episode(self, ok: bool):
        if not self.log_dir:
            return
        try:
            path = Path(self.log_dir)
            path.mkdir(parents=True, exist_ok=True)
            rec = {
                "episode": self._episode - 1,
                "reset_seed": self._reset_seed,
                "type": self._ep_type,
                "level": self.level,
                "mag": self._ep_mag,
                "t_dist": self._t_dist,
                "applied": ok,
            }
            with open(path / f"dro_log_{os.getpid()}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:  # noqa: BLE001
            pass


def _factory(base_task):
    def make(**kwargs):
        env = gym.make(f"gym_aloha/{base_task}", **kwargs)
        # Strip the inner gym_aloha TimeLimit so ONLY the caller's (lerobot's)
        # max_episode_steps governs episode length — see v9/perturb_env.py for why
        # double-wrapping silently changes episode length and depresses SR.
        from gymnasium.wrappers import TimeLimit

        while isinstance(env, TimeLimit):
            env = env.env
        return DROWrapper(env)

    return make


for _bt in ("AlohaTransferCube-v0", "AlohaInsertion-v0"):
    _pid = f"gym_aloha/{_bt.replace('-v0', '')}DRO-v0"
    if _pid not in gym.registry:
        gym.register(id=_pid, entry_point=_factory(_bt))
