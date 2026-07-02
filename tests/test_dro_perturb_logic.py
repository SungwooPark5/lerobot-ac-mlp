"""Logic tests for lerobot/envs/dro_perturb.py that run WITHOUT gym_aloha/mujoco.

gym_aloha is stubbed and a fake dm_control-style Physics mimics the TransferCube
joint layout (16 robot hinge joints + 1 object free joint). Verifies: env
registration, TimeLimit stripping, per-type injection (push/teleport/obsnoise),
robot-vs-object dof auto-detection, per-episode seed-derived rng, one-shot
application, and jsonl logging.

Run:  python tests/test_dro_perturb_logic.py   (also pytest-compatible)
"""

import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

REPO = Path(__file__).resolve().parents[1]
N_ROBOT = 16  # hinge joints (8 per arm incl. grippers, as in gym_aloha transfer_cube)


# --------------------------------------------------------------- fake physics
class FakeModel:
    def __init__(self, n_robot=N_ROBOT, n_free=1):
        self.jnt_type = np.array([3] * n_robot + [0] * n_free)  # 3=hinge, 0=free
        self.jnt_qposadr = np.array(list(range(n_robot)) + [n_robot + 7 * i for i in range(n_free)])
        self.jnt_dofadr = np.array(list(range(n_robot)) + [n_robot + 6 * i for i in range(n_free)])
        self.nq = n_robot + 7 * n_free
        self.nv = n_robot + 6 * n_free


class FakeData:
    def __init__(self, model):
        self.qpos = np.zeros(model.nq)
        self.qvel = np.zeros(model.nv)


class FakePhysics:
    def __init__(self, n_free=1):
        self.model = FakeModel(n_free=n_free)
        self.data = FakeData(self.model)
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class FakeInnerEnv:
    """Mimics gym_aloha's unwrapped env exposing ._env.physics."""

    def __init__(self, n_free):
        self.physics = FakePhysics(n_free=n_free)


class FakeAlohaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, obs_type="pixels_agent_pos", render_mode="rgb_array", n_free=1, **kwargs):
        self._env = types.SimpleNamespace(physics=FakePhysics(n_free=n_free))
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(-10, 10, shape=(14,), dtype=np.float64),
                "pixels": spaces.Dict({"top": spaces.Box(0, 255, shape=(4, 4, 3), dtype=np.uint8)}),
            }
        )
        self.action_space = spaces.Box(-1, 1, shape=(14,), dtype=np.float32)
        self.render_mode = render_mode

    def _obs(self):
        return {
            "agent_pos": np.zeros(14, dtype=np.float64),
            "pixels": {"top": np.zeros((4, 4, 3), dtype=np.uint8)},
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


# ------------------------------------------------------------------- imports
def _import_dro_perturb():
    sys.modules.setdefault("gym_aloha", types.ModuleType("gym_aloha"))
    for task, n_free in (("AlohaTransferCube-v0", 1), ("AlohaInsertion-v0", 2)):
        if f"gym_aloha/{task}" not in gym.registry:
            gym.register(
                id=f"gym_aloha/{task}",
                entry_point=lambda n_free=n_free, **kw: FakeAlohaEnv(n_free=n_free, **kw),
                max_episode_steps=300,  # mimic gym_aloha's inner TimeLimit (must get stripped)
            )
    spec = importlib.util.spec_from_file_location(
        "dro_perturb", REPO / "src" / "lerobot" / "envs" / "dro_perturb.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DRO = _import_dro_perturb()


def _make(task="AlohaTransferCube", **env_vars):
    for k in list(os.environ):
        if k.startswith("DRO_"):
            del os.environ[k]
    os.environ.update({k: str(v) for k, v in env_vars.items()})
    return gym.make(
        f"gym_aloha/{task}DRO-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=400,
    )


def _wrapper(env):
    e = env
    while not isinstance(e, DRO.DROWrapper):
        e = e.env
    return e


def _physics(env):
    return env.unwrapped._env.physics


def _run(env, n_steps, seed=0):
    env.reset(seed=seed)
    a = np.zeros(14, dtype=np.float32)
    infos = []
    for _ in range(n_steps):
        _, _, _, _, info = env.step(a)
        infos.append(info)
    return infos


# --------------------------------------------------------------------- tests
def test_registration_and_timelimit():
    env = _make(DRO_TYPE="none")
    # exactly one TimeLimit (the outer 400), the inner 300 must have been stripped
    tls = []
    e = env
    while hasattr(e, "env"):
        if isinstance(e, gym.wrappers.TimeLimit):
            tls.append(e._max_episode_steps)
        e = e.env
    assert tls == [400], f"TimeLimit chain wrong: {tls}"
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.zeros(14, dtype=np.float32))
    assert info["dro_type"] == "none" and info["dro_step"] == -1


def test_push_hits_robot_dofs_only():
    env = _make(DRO_TYPE="push", DRO_LEVEL=2, DRO_STEP=5)
    phys = _physics(env)
    infos = _run(env, 10, seed=123)
    qvel = phys.data.qvel
    robot, free_dofs = qvel[:N_ROBOT], qvel[N_ROBOT:]
    assert np.isclose(np.linalg.norm(robot), DRO.DRO_LEVELS["push"][2], atol=1e-6)
    assert np.all(free_dofs == 0), "object dofs must be untouched by push"
    assert phys.forward_calls == 1, "physics.forward must run exactly once"
    assert [i["dro_active"] for i in infos] == [False] * 5 + [True] * 5


def test_teleport_moves_object_xy_only():
    env = _make(DRO_TYPE="teleport", DRO_LEVEL=3, DRO_STEP=3)
    phys = _physics(env)
    _run(env, 6, seed=7)
    qpos = phys.data.qpos
    assert np.all(qpos[:N_ROBOT] == 0), "robot qpos must be untouched by teleport"
    xy = qpos[N_ROBOT : N_ROBOT + 2]
    assert np.isclose(np.linalg.norm(xy), DRO.DRO_LEVELS["teleport"][3], atol=1e-9)
    assert qpos[N_ROBOT + 2] == 0, "z must be unchanged"


def test_teleport_insertion_picks_one_of_two_objects():
    env = _make(task="AlohaInsertion", DRO_TYPE="teleport", DRO_LEVEL=2, DRO_STEP=2)
    phys = _physics(env)
    _run(env, 4, seed=42)
    moved = [
        np.linalg.norm(phys.data.qpos[adr : adr + 2]) > 1e-12 for adr in (N_ROBOT, N_ROBOT + 7)
    ]
    assert sum(moved) == 1, f"exactly one object must move, got {moved}"


def test_obsnoise_window():
    env = _make(DRO_TYPE="obsnoise", DRO_LEVEL=3, DRO_STEP=4, DRO_OBS_STEPS=3)
    env.reset(seed=0)
    a = np.zeros(14, dtype=np.float32)
    norms = []
    for _ in range(10):
        obs, _, _, _, _ = env.step(a)
        norms.append(float(np.linalg.norm(obs["agent_pos"])))
    noisy = [n > 0 for n in norms]
    assert noisy == [False] * 4 + [True] * 3 + [False] * 3, f"noise window wrong: {norms}"
    phys = _physics(env)
    assert np.all(phys.data.qpos == 0) and np.all(phys.data.qvel == 0), "obsnoise must not touch physics"


def test_seed_reproducibility_and_diversity():
    def draw(seed):
        env = _make(DRO_TYPE="push", DRO_LEVEL=1)  # random step
        phys = _physics(env)
        _run(env, 300, seed=seed)
        return _wrapper(env)._t_dist, phys.data.qvel[:N_ROBOT].copy()

    t1, v1 = draw(11)
    t2, v2 = draw(11)
    t3, v3 = draw(22)
    assert t1 == t2 and np.allclose(v1, v2), "same reset seed must reproduce the disturbance"
    assert t3 != t1 or not np.allclose(v3, v1), "different reset seeds must differ"
    assert 80 <= t1 <= 240, f"random step {t1} outside default window"


def test_mag_override_and_one_shot():
    env = _make(DRO_TYPE="push", DRO_MAG=9.99, DRO_STEP=2)
    phys = _physics(env)
    _run(env, 8, seed=5)
    assert np.isclose(np.linalg.norm(phys.data.qvel[:N_ROBOT]), 9.99, atol=1e-6)
    assert phys.forward_calls == 1, "disturbance must be one-shot"


def test_log_jsonl():
    with tempfile.TemporaryDirectory() as td:
        env = _make(DRO_TYPE="teleport", DRO_LEVEL=1, DRO_STEP=3, DRO_LOG_DIR=td)
        _run(env, 5, seed=99)
        _run(env, 5, seed=100)
        files = list(Path(td).glob("dro_log_*.jsonl"))
        assert len(files) == 1
        recs = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(recs) == 2
        assert recs[0]["reset_seed"] == 99 and recs[1]["reset_seed"] == 100
        assert all(r["applied"] and r["type"] == "teleport" and r["t_dist"] == 3 for r in recs)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nall {len(fns)} tests passed")
