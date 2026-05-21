"""v3 실험 공통 모듈.
모든 v3 노트북이 이 파일을 import하여 사용한다.
Usage (노트북 첫 셀):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath('.')))  # v3/ 폴더를 path에 추가
    from common import *
"""
import os, sys, subprocess, json, glob, shutil, re
from pathlib import Path

# ===== Environment Setup =====
HOME = os.path.expanduser('~')
PROJECT = os.path.join(HOME, 'lerobot_project', 'lerobot-ac-mlp')
PYTHON = os.path.join(HOME, 'lerobot_project', 'lerobot_env', 'bin', 'python')
SITE = os.path.join(HOME, 'lerobot_project', 'lerobot_env', 'lib', 'python3.12', 'site-packages')
if SITE not in sys.path:
    sys.path.insert(0, SITE)

os.environ['MPLBACKEND'] = 'Agg'
conda_lib = os.path.join(HOME, 'miniconda3', 'lib')
lib_dirs = {conda_lib}
try:
    r = subprocess.run(
        ['bash', '-c', f'find {HOME}/miniconda3 -name "libOSMesa*" -o -name "libEGL*" 2>/dev/null'],
        capture_output=True, text=True, timeout=15,
    )
    for line in r.stdout.strip().split('\n'):
        if line.strip():
            lib_dirs.add(os.path.dirname(line.strip()))
except Exception:
    pass
LD = ':'.join(lib_dirs) + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = LD
os.environ['MUJOCO_GL'] = 'osmesa'
ENVS = f'MUJOCO_GL=osmesa LD_LIBRARY_PATH={LD}'

# ===== Constants =====
ENV = 'aloha'
TASK = 'AlohaTransferCube-v0'
DATASET = 'lerobot/aloha_sim_transfer_cube_human'
BS = 8

# V3 defaults
EVAL_FREQ = 10000
SAVE_FREQ = 10000
LOG_FREQ = 50
EVAL_EP = 50
EVAL_EPISODES_FINAL = 100  # 본 평가

# Model configs
MODEL_CONFIGS = {
    'act':       {'type': 'act',       'k_list': [50, 100, 150, 200, 300, 400], 'lr_list': ['1e-4', '1e-5', '1e-6']},
    'acm':       {'type': 'acm',       'k_list': [50, 100, 150, 200, 300, 400], 'lr_list': ['1e-4', '1e-5', '1e-6']},
    'acm2':      {'type': 'acm2',      'k_list': [50, 100, 150, 200, 300, 400], 'lr_list': ['1e-4', '1e-5', '1e-6']},
    'acm3':      {'type': 'acm3',      'k_list': [50, 100, 150, 200, 300, 400], 'lr_list': ['1e-4', '1e-5', '1e-6']},
    'diffusion': {'type': 'diffusion', 'k_list': [50, 100, 150, 200, 300, 400], 'lr_list': ['1e-4', '1e-5', '1e-6']},
}
SEEDS = [0, 1, 2]


def detect_gpus():
    r = subprocess.run('nvidia-smi -L 2>/dev/null | wc -l', shell=True, capture_output=True, text=True)
    return max(1, int(r.stdout.strip() or '1'))


def base_dir(seed=0, sub=''):
    """outputs/v3_seed{seed}/{sub}"""
    p = os.path.join(PROJECT, 'outputs', f'v3_seed{seed}', sub)
    os.makedirs(p, exist_ok=True)
    return p


# ===== Training =====
def train_cmd(tag, policy_type, lr, seed, gpu, out_dir, chunk=100, steps=100000,
              eval_freq=EVAL_FREQ, save_freq=SAVE_FREQ, log_freq=LOG_FREQ,
              eval_ep=EVAL_EP, extra_args=''):
    log_dir = os.path.dirname(out_dir)
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f'{tag}.log')
    c = (f'CUDA_VISIBLE_DEVICES={gpu} TQDM_DISABLE=1 PYTHONWARNINGS=ignore {ENVS} '
         f'{PYTHON} -m lerobot.scripts.lerobot_train '
         f'--policy.type={policy_type} --policy.chunk_size={chunk} '
         f'--policy.n_action_steps={chunk} --policy.push_to_hub=false '
         f'--policy.optimizer_lr={lr} '
         f'--env.type={ENV} --env.task={TASK} '
         f'--dataset.repo_id={DATASET} --dataset.video_backend=pyav '
         f'--steps={steps} --eval_freq={eval_freq} --save_freq={save_freq} '
         f'--log_freq={log_freq} --batch_size={BS} '
         f'--eval.n_episodes={eval_ep} --eval.batch_size=1 '
         f'--output_dir={out_dir} --wandb.enable=false --seed={seed} {extra_args}')
    c += f' > {log} 2>&1 && echo "[{tag}] 완료" || echo "[{tag}] 실패"'
    return c


def eval_cmd(tag, checkpoint, m, gpu, out_dir, n_episodes=EVAL_EPISODES_FINAL):
    log_dir = os.path.dirname(out_dir)
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, f'{tag}.log')
    c = (f'CUDA_VISIBLE_DEVICES={gpu} TQDM_DISABLE=1 PYTHONWARNINGS=ignore {ENVS} '
         f'{PYTHON} -m lerobot.scripts.lerobot_eval '
         f'--policy.path={checkpoint} --policy.n_action_steps={m} '
         f'--env.type={ENV} --env.task={TASK} '
         f'--eval.n_episodes={n_episodes} --eval.batch_size=1 '
         f'--output_dir={out_dir} '
         f'> {log} 2>&1 && echo "[{tag}] 완료" || echo "[{tag}] 실패"')
    return c


def run_parallel(jobs, n_gpus):
    q = {i: [] for i in range(n_gpus)}
    for i, (tag, cmd) in enumerate(jobs):
        q[i % n_gpus].append(
            f'echo "[GPU {i % n_gpus}] 시작: {tag}" && {cmd} && echo "[GPU {i % n_gpus}] 완료: {tag}"'
        )
    parts = [f'({" && ".join(c)})' for c in q.values() if c]
    return ' & '.join(parts) + ' & wait'


# ===== Checkpoint Helpers =====
def is_done(out_dir, target_steps):
    jl = os.path.join(out_dir, 'training_log.jsonl')
    if not os.path.exists(jl):
        return False
    try:
        with open(jl) as f:
            lines = f.readlines()
        if not lines:
            return False
        return json.loads(lines[-1]).get('step', 0) >= target_steps
    except Exception:
        return False


def find_best_checkpoint(model_dir):
    """학습 중 eval 결과에서 best step의 체크포인트 경로 반환. 없으면 last fallback."""
    eval_files = sorted(glob.glob(os.path.join(model_dir, 'eval', 'eval_step_*.json')))
    best = None
    for ef in eval_files:
        try:
            with open(ef) as f:
                ev = json.load(f)
            sr = ev['overall']['pc_success']
            step = ev['step']
            if best is None or sr > best[1]:
                best = (step, sr)
        except Exception:
            pass

    ckpt_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.isdir(ckpt_dir):
        return None, None

    if best is not None:
        for c in [f'{best[0]:06d}', str(best[0])]:
            p = os.path.join(ckpt_dir, c, 'pretrained_model')
            if os.path.isdir(p):
                return p, best
        subs = sorted([d for d in os.listdir(ckpt_dir) if d.isdigit()])
        if subs:
            closest = min(subs, key=lambda d: abs(int(d) - best[0]))
            return os.path.join(ckpt_dir, closest, 'pretrained_model'), (int(closest), best[1])

    last = os.path.join(ckpt_dir, 'last', 'pretrained_model')
    if os.path.isdir(last):
        return last, ('last', None)
    subs = sorted([d for d in os.listdir(ckpt_dir) if d.isdigit()])
    if subs:
        return os.path.join(ckpt_dir, subs[-1], 'pretrained_model'), (int(subs[-1]), None)
    return None, None


# ===== Eval Helpers =====
def load_sr(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f).get('overall', {}).get('pc_success', float('nan'))
        except Exception:
            pass
    return float('nan')


def get_m_list(k):
    base = [1, 10, 20, 30, 40, 50]
    extra = [75, 100, 150, 200, 300]
    ms = base + [m for m in extra if m < k] + [k]
    return sorted(set(ms))


def compute_jitter(eval_dir):
    """action_logs/*.pt에서 에피소드별 jitter(프레임간 action 변화량 평균) 계산. 없으면 nan."""
    import torch as _torch
    # action_logs가 eval_dir 바로 아래 또는 videos/ 아래 있을 수 있음
    for candidate in [os.path.join(eval_dir, 'action_logs'),
                      os.path.join(eval_dir, 'videos', 'action_logs')]:
        if os.path.isdir(candidate):
            logs_dir = candidate
            break
    else:
        return float('nan')
    # 기존 로직 (logs_dir 결정 후 동일)
    pts = sorted(glob.glob(os.path.join(logs_dir, 'episode_*.pt')))
    if not pts:
        return float('nan')
    jitters = []
    for p in pts:
        try:
            d = _torch.load(p, map_location='cpu', weights_only=True)
            actions = d['actions']  # (T, action_dim)
            if actions.shape[0] < 2:
                continue
            diffs = actions[1:] - actions[:-1]
            jitters.append(diffs.norm(dim=-1).mean().item())
        except Exception:
            pass
    return float('nan') if not jitters else sum(jitters) / len(jitters)


def load_jitter(eval_dir):
    """eval_dir에서 jitter 값 반환. compute_jitter의 alias."""
    return compute_jitter(eval_dir)


def collect_eval_results(prefix, eval_dir):
    """prefix='act' etc → [(k, m, sr), ...] 내림차순"""
    results = []
    pat = re.compile(rf'^{prefix}_k(\d+)_m(\d+)$')
    if not os.path.isdir(eval_dir):
        return results
    for d in sorted(os.listdir(eval_dir)):
        match = pat.match(d)
        if not match:
            continue
        k, m = int(match.group(1)), int(match.group(2))
        sr = load_sr(os.path.join(eval_dir, d, 'eval_info.json'))
        if not __import__('math').isnan(sr):
            results.append((k, m, sr))
    results.sort(key=lambda x: -x[2])
    return results
