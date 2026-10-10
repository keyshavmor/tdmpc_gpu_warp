"""End-to-end crash / resume test of src/train.py with checkpoints (and optionally wandb).

Trains a short run with frequent checkpoints and kills it twice with SIGKILL:
  crash A: ~1 s after a checkpoint (its metrics are still in wandb's asynchronous upload queue),
  crash B: two episodes after the next checkpoint (metrics not yet sent; the episodes are lost and re-run),
then resumes and finishes. Checks that train.log and eval.log contain every env step exactly once and that the
steps after crash B carry the values of the resumed run; with --wandb_entity it also checks that the wandb run
history equals the local logs step for step (one row per step, identical values) and deletes the test run.

Environment: tdmpc-warp (a CUDA GPU); wandb login for the optional wandb check.
Usage:
	python bench/test_resume.py [--task walker-walk] [--wandb_entity ENTITY --wandb_project tdmpc-gpu-warp-resume-test]
Writes results/checkpoint/resume_test.json. Results: docs/USER_GUIDE.md (section 4.3).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
p = argparse.ArgumentParser()
p.add_argument('--task', default='walker-walk')
p.add_argument('--seed', type=int, default=8)
p.add_argument('--wandb_entity', default=None)
p.add_argument('--wandb_project', default='tdmpc-gpu-warp-resume-test')
p.add_argument('--keep_wandb_run', action='store_true')
a = p.parse_args()
log_root = tempfile.mkdtemp(prefix='resume_test_')
wd = os.path.join(log_root, a.task, 'state', 'resume_test', str(a.seed))
args = [sys.executable, '-u', 'src/train.py', f'task={a.task}', f'seed={a.seed}', 'graph_agent=true', 'pace_host=true',
		'checkpoint_minutes=1', 'max_env_steps=22000', 'eval_freq=4000', 'eval_episodes=1', f'log_root={log_root}',
		'exp_name=resume_test']
if a.wandb_entity:
	args += ['use_wandb=true', f'wandb_project={a.wandb_project}', f'wandb_entity={a.wandb_entity}']


def last_ckpt(out):
	"""Env step of the last 'Checkpoint at env step N' line in the run's output file, or None if there is none yet."""
	m = re.findall(r'^Checkpoint at env step (\d+)', open(out).read(), re.M)
	return int(m[-1]) if m else None


def last_step():
	"""Env step of the last row of the run's train.log (second CSV column); -1 if the file is missing or has no data row yet."""
	try:
		return int(open(os.path.join(wd, 'train.log')).read().strip().splitlines()[-1].split(',')[1])
	except (OSError, ValueError, IndexError):
		return -1


def start(i, resume):
	"""Launch training run i (with resume=true if `resume`); returns (Popen, path of its output file)."""
	out = os.path.join(log_root, f'run{i}.out')
	return subprocess.Popen(args + (['resume=true'] if resume else []), cwd=ROOT, stdout=open(out, 'w'), stderr=subprocess.STDOUT), out


res = {'task': a.task}
# crash A: ~1 s after a checkpoint past the seeding phase
pr, out = start(1, False)
while (last_ckpt(out) or 0) < 12000:
	assert pr.poll() is None, open(out).read()[-2000:]
	time.sleep(0.5)
time.sleep(1); pr.kill(); pr.wait()
res['crash_A'] = {'checkpoint': last_ckpt(out), 'train_log_at': last_step()}
# crash B: two episodes after the next checkpoint, before another checkpoint
pr, out = start(2, True)
while last_ckpt(out) is None:
	assert pr.poll() is None, open(out).read()[-2000:]
	time.sleep(0.5)
ck = last_ckpt(out); L = 1000  # env steps per episode for all DMC tasks
while last_step() < ck + 2 * L:
	time.sleep(0.5)
pr.kill(); pr.wait()
res['crash_B'] = {'checkpoint': ck, 'train_log_at': last_step(), 'clean': last_ckpt(out) == ck}
killed = {int(l.split(',')[1]): float(l.split(',')[2]) for l in open(os.path.join(wd, 'train.log')).read().splitlines()[1:]}
# final run
pr, out = start(3, True); rc = pr.wait()
res['final_exit_code'] = rc
tl = [l.split(',') for l in open(os.path.join(wd, 'train.log')).read().splitlines()[1:]]
steps = [int(r[1]) for r in tl]; rew = {int(r[1]): float(r[2]) for r in tl}
el = [l.split(',') for l in open(os.path.join(wd, 'eval.log')).read().splitlines()[1:]]
res['train_log_steps_unique_continuous'] = steps == list(range(0, steps[-1] + 1, L))
res['eval_log_steps_unique'] = len({float(r[0]) for r in el}) == len(el)
seg = [s for s in killed if s > ck]
res['killed_segment_steps'] = seg
if a.wandb_entity:
	import wandb
	rid = open(os.path.join(wd, 'wandb_run_id.txt')).read().strip()
	time.sleep(20)
	run = wandb.Api().run(f'{a.wandb_entity}/{a.wandb_project}/{rid}')
	rows = [r for r in run.scan_history(keys=['train/env_step', 'train/episode_reward'])]
	ws = [int(r['train/env_step']) for r in rows]
	res['wandb_steps_equal_local'] = ws == steps
	res['wandb_values_equal_local'] = all(abs(r['train/episode_reward'] - rew[int(r['train/env_step'])]) < 1e-4 for r in rows)
	ev = [r for r in run.scan_history(keys=['eval/env_step', 'eval/episode_reward'])]
	res['wandb_eval_equal_local'] = [(int(r['eval/env_step']), round(r['eval/episode_reward'], 3)) for r in ev] == \
		[(int(float(r[0])), round(float(r[1]), 3)) for r in el]
	if not a.keep_wandb_run:
		run.delete()
checks = [v for k, v in res.items() if isinstance(v, bool)] + [res['crash_B']['clean'], res['final_exit_code'] == 0]
res['pass'] = all(checks)
print(json.dumps(res, indent=1)); print('PASS' if res['pass'] else 'FAIL')
d = os.path.join(ROOT, 'results', 'checkpoint'); os.makedirs(d, exist_ok=True)
json.dump(res, open(os.path.join(d, 'resume_test.json'), 'w'), indent=1)
shutil.rmtree(log_root, ignore_errors=True)
