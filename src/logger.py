"""Logging of the original TD-MPC: console output, eval.log (CSV) and optional wandb.

Used by src/train.py (train_vec.py writes its per-seed eval.log directly, in the same format).

Original code: TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (ICML 2022),
https://github.com/nicklashansen/tdmpc (MIT License, Copyright (c) 2021 Nicklas Hansen).
Modifications for the GPU port (tdmpc_gpu_warp) by Keyshav Mor: resumable wandb runs for checkpoint / resume
(src/checkpoint.py), everything else unchanged.
  * A new run gets a fixed wandb run id, stored in <log_dir>/wandb_run_id.txt.
  * `wandb_sync=deferred` (default when checkpointing is on): metrics are kept in memory and sent to wandb only
    after a checkpoint has been written (flush_wandb()). They are first appended to a durable local copy,
    <log_dir>/wandb_history.jsonl (and stored in the checkpoint), because wandb.log() is asynchronous and data not
    yet uploaded when a process is killed is lost. wandb therefore never contains data from after the last
    checkpoint. A resumed run re-attaches to the same run (resume='allow'), re-sends every locally recorded step
    up to the checkpoint (wandb keeps one row per step; re-sent steps carry identical values) and continues
    logging - the wandb curves hold exactly one value per step, as in an uninterrupted run (verified with two
    SIGKILL crashes: during a wandb upload and with unsent metrics; bench/test_resume.py). Cost: wandb lags by up to one checkpoint interval (eval.log / console are immediate).
  * `wandb_sync=live`: metrics are sent immediately; a resumed run is re-attached with wandb's rewind
    (`resume_from='<id>?_step=E'`, truncates history after the checkpoint step E). Rewind is a wandb preview
    feature that must be enabled for the account; without it, use the deferred mode.
"""
import sys
import os
import json
import secrets
import datetime
import re
import numpy as np
import torch
import pandas as pd
from termcolor import colored
from omegaconf import OmegaConf


CONSOLE_FORMAT = [('episode', 'E', 'int'), ('env_step', 'S', 'int'), ('episode_reward', 'R', 'float'), ('total_time', 'T', 'time')]
AGENT_METRICS = ['consistency_loss', 'reward_loss', 'value_loss', 'total_loss', 'weighted_loss', 'pi_loss', 'grad_norm']


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg, reward=None):
	"""Pretty-printing of run information. Call at start of training."""
	prefix, color, attrs = '  ', 'green', ['bold']
	def limstr(s, maxlen=32):
		"""str(s) truncated to maxlen characters (with an ellipsis)."""
		return str(s[:maxlen]) + '...' if len(str(s)) > maxlen else s
	def pprint(k, v):
		"""Print one "key: value" line of the run summary."""
		print(prefix + colored(f'{k.capitalize()+":":<16}', color, attrs=attrs), limstr(v))
	kvs = [('task', cfg.task_title),
		   ('train steps', f'{int(cfg.train_steps*cfg.action_repeat):,}'),
		   ('observations', 'x'.join([str(s) for s in cfg.obs_shape])),
		   ('actions', cfg.action_dim),
		   ('experiment', cfg.exp_name)]
	if reward is not None:
		kvs.append(('episode reward', colored(str(int(reward)), 'white', attrs=['bold'])))
	w = np.max([len(limstr(str(kv[1]))) for kv in kvs]) + 21
	div = '-'*w
	print(div)
	for k,v in kvs:
		pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""Return a wandb-safe group name for logging. Optionally returns group name as list."""
	lst = [cfg.task, cfg.modality, re.sub('[^0-9a-zA-Z]+', '-', cfg.exp_name)]
	return lst if return_list else '-'.join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""
	def __init__(self, root_dir, wandb, render_size=384, fps=15):
		self.save_dir = (root_dir / 'eval_video') if root_dir else None
		self._wandb = wandb
		self.render_size = render_size
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		"""Start a new video (recording only if a save dir and wandb are available)."""
		self.frames = []
		self.enabled = self.save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		"""Render and store one frame."""
		if self.enabled:
			frame = env.render(mode='rgb_array', height=self.render_size, width=self.render_size, camera_id=0)
			self.frames.append(frame)

	def save(self, step):
		"""Upload the recorded frames to wandb as an mp4."""
		if self.enabled:
			frames = np.stack(self.frames).transpose(0, 3, 1, 2)
			self._wandb.log({'eval_video': self._wandb.Video(frames, fps=self.fps, format='mp4')}, step=step)


class Logger(object):
	"""Primary logger object. Logs either locally or using wandb."""
	def __init__(self, log_dir, cfg, resume_step=None):
		"""resume_step: env step of the checkpoint a resumed run continues from (None = new run)."""
		self._log_dir = make_dir(log_dir)
		self._model_dir = make_dir(self._log_dir / 'models')
		self._save_model = cfg.save_model
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._cfg = cfg
		self._eval = []
		self._pending = []  # deferred wandb metrics: [(step, {key: value}), ...] since the last checkpoint
		self._defer = str(cfg.get('wandb_sync', 'deferred')) == 'deferred' and float(cfg.get('checkpoint_minutes', 0) or 0) > 0
		print_run(cfg)
		project, entity = cfg.get('wandb_project', 'none'), cfg.get('wandb_entity', 'none')
		run_offline = not cfg.get('use_wandb', False) or project == 'none' or entity == 'none'
		if run_offline:
			print(colored('Logs will be saved locally.', 'yellow', attrs=['bold']))
			self._wandb = None
		else:
			try:
				os.environ["WANDB_SILENT"] = "true"
				import wandb
				init_kwargs = dict(project=project,
						entity=entity,
						name=str(cfg.seed),
						group=self._group,
						tags=cfg_to_group(cfg, return_list=True) + [f'seed:{cfg.seed}'],
						dir=self._log_dir,
						config=OmegaConf.to_container(cfg, resolve=True))
				id_file = self._log_dir / 'wandb_run_id.txt'
				if resume_step is not None and id_file.exists():
					run_id = id_file.read_text().strip()
					if self._defer:  # wandb only holds data up to the checkpoint: simply continue the run
						wandb.init(id=run_id, resume='allow', **init_kwargs)
						print(colored(f'Resumed wandb run {run_id} after env step {int(resume_step)}.', 'blue', attrs=['bold']))
					else:
						try:  # live mode: rewind (drop history after the checkpoint step) and continue the same run
							wandb.init(resume_from=f'{run_id}?_step={int(resume_step)}', **init_kwargs)
							print(colored(f'Resumed wandb run {run_id} from step {int(resume_step)} (rewind).', 'blue', attrs=['bold']))
						except Exception as e:  # rewind not enabled: continue the run, wandb drops non-increasing steps
							print(colored(f'wandb rewind failed ({e}); resuming run {run_id} without rewind - steps logged after '
										  f'the checkpoint before the crash stay in wandb. Use wandb_sync=deferred.', 'yellow', attrs=['bold']))
							wandb.init(id=run_id, resume='allow', **init_kwargs)
				else:
					run_id = ''.join(secrets.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(8))  # not `random`: keeps the training RNG untouched
					id_file.write_text(run_id)
					wandb.init(id=run_id, **init_kwargs)
				print(colored('Logs will be synced with wandb.', 'blue', attrs=['bold']))
				self._wandb = wandb
			except:
				print(colored('Warning: failed to init wandb. Logs will be saved locally.', 'yellow'), attrs=['bold'])
				self._wandb = None
		self._video = VideoRecorder(log_dir, self._wandb) if self._wandb and cfg.save_video else None

	@property
	def video(self):
		"""The VideoRecorder (None unless W&B and save_video are enabled; video is off by default)."""
		return self._video

	def flush_wandb(self):
		"""Send the deferred metrics to wandb (called after a checkpoint has been written, and at the end).

		The metrics are appended (and fsync'ed) to <log_dir>/wandb_history.jsonl first, so that data lost in
		wandb's asynchronous upload queue when the process is killed can be re-sent on resume."""
		if self._wandb is not None and self._pending:
			with open(self._log_dir / 'wandb_history.jsonl', 'a') as f:
				for step, d in self._pending:
					f.write(json.dumps({'step': int(step), 'data': d}) + '\n')
				f.flush(); os.fsync(f.fileno())
			for step, d in self._pending:
				self._wandb.log(d, step=step)
		self._pending = []

	def resume_sync(self, ckpt_pending, last_env_step):
		"""After resuming: re-send every recorded step that wandb did not receive, up to the checkpoint.

		History = wandb_history.jsonl + the metrics stored in the checkpoint, restricted to steps <= last_env_step
		(the file is rewritten accordingly), merged per step. All of it is sent again (wandb 0.30 reports
		run.step = 0 after re-attaching; the server keeps one row per step, and re-sent values are identical)."""
		if self._wandb is None:
			return 0
		path = self._log_dir / 'wandb_history.jsonl'
		entries = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
		entries += [{'step': int(st), 'data': d} for st, d in ckpt_pending]
		merged = {}
		for e in entries:
			if e['step'] <= last_env_step:
				merged.setdefault(e['step'], {}).update(e['data'])
		with open(path, 'w') as f:
			for st in sorted(merged):
				f.write(json.dumps({'step': st, 'data': merged[st]}) + '\n')
			f.flush(); os.fsync(f.fileno())
		first = int(self._wandb.run.step)
		resent = [st for st in sorted(merged) if st >= first]
		for st in resent:
			self._wandb.log(merged[st], step=st)
		print(colored(f'wandb: re-sent {len(resent)} recorded step(s) up to env step {int(last_env_step)}.', 'blue'))
		return len(resent)

	def finish(self, agent):
		"""Optionally save the model (and upload it to wandb), close wandb and print the run summary."""
		self.flush_wandb()
		if self._save_model:
			fp = self._model_dir / f'model.pt'
			torch.save(agent.state_dict(), fp)
			if self._wandb:
				artifact = self._wandb.Artifact(self._group+'-'+str(self._seed), type='model')
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)
		if self._wandb:
			self._wandb.finish()
		print_run(self._cfg, self._eval[-1][-1])

	def _format(self, key, value, ty):
		"""Format one console metric as 'key: value' for type 'int' (thousands separators), 'float' (1 decimal) or 'time' (H:MM:SS)."""
		if ty == 'int':
			return f'{colored(key+":", "grey")} {int(value):,}'
		elif ty == 'float':
			return f'{colored(key+":", "grey")} {value:.01f}'
		elif ty == 'time':
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "grey")} {value}'
		else:
			raise f'invalid log format type: {ty}'

	def _print(self, d, category):
		"""Print one console line for a train/eval metrics dict, using the keys and formats of CONSOLE_FORMAT."""
		category = colored(category, 'blue' if category == 'train' else 'green')
		pieces = [f' {category:<14}']
		for k, disp_k, ty in CONSOLE_FORMAT:
			pieces.append(f'{self._format(disp_k, d.get(k, 0), ty):<26}')
		print('   '.join(pieces))

	def log(self, d, category='train'):
		"""Log a metrics dict to wandb (if enabled) and the console; eval metrics are also appended to eval.log."""
		assert category in {'train', 'eval'}
		if self._wandb is not None:
			if self._defer:  # sent by flush_wandb() once the next checkpoint is on disk
				self._pending.append((int(d['env_step']), {category + '/' + k: float(v) for k, v in d.items()}))
			else:
				for k,v in d.items():
					self._wandb.log({category + '/' + k: v}, step=d['env_step'])
		if category == 'eval':
			keys = ['env_step', 'episode_reward']
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(self._log_dir / 'eval.log', header=keys, index=None)
		self._print(d, category)
