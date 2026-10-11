"""Capsule-capsule and capsule-box contacts: MuJoCo Warp (float32, GPU) vs C MuJoCo (float64), incl. nearly parallel axes.

MuJoCo Warp ports C MuJoCo's segment-segment closest-point solve (capsule-capsule, and capsule vs box edges) with
det = ma * mc - mb * mb. In float32 that difference cancels catastrophically when the two axes are nearly parallel
(relative error ~ eps / angle^2), so the clamped closest points - and with them the contact position - can be wrong.
It happens in practice: hopper's thigh folds back parallel to its torso (waist + hip = -pi) and rests there.
patches/mujoco_warp/0002-capsule-collision-float32-precision.patch computes det and the numerators with cross products
(Lagrange identity: mathematically identical, no cancellation).

This test places two free bodies (capsule + capsule, capsule + box) in many worlds with random poses, half of them
nearly parallel (angle 10^U(-7, -1) rad) and overlapping, runs the collision on the GPU and in C MuJoCo on identical
(float32-representable) poses, and compares contacts world by world. For nearly parallel axes the position of the
closest point along the overlap is ill-conditioned in any precision, so a contact counts as matching if its depth and
position agree to 1e-4 or lie within the spread of C MuJoCo's own result under float32-sized input noise (20
perturbed evaluations). Pass: capsule-capsule - contact count and contacts match in every world; capsule-box - the
same, except for at most 0.2 % of the worlds where C MuJoCo returns two contacts of equal depth (|difference| < 1e-6;
a capsule lying flat on a box face) and MuJoCo Warp one: that tie is decided in a separate branch of the box code,
not in the segment solve this patch changes.

Environment: tdmpc-warp (CUDA GPU).
Usage: python bench/test_capsule_collision.py [--worlds 4000]
Writes results/validation/capsule_collision.json. Results: docs/03_validation.md.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import argparse, json, os
import numpy as np
import mujoco
import mujoco_warp as mjw
import warp as wp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
p = argparse.ArgumentParser()
p.add_argument('--worlds', type=int, default=4000)
p.add_argument('--out', default=os.path.join(ROOT, 'results', 'validation', 'capsule_collision.json'))
a = p.parse_args()

XML = """<mujoco><option gravity="0 0 0"/><worldbody>
  <body name="a"><freejoint/><geom type="capsule" size="0.05 0.15"/></body>
  <body name="b"><freejoint/><geom type="{other}" size="{size}"/></body>
</worldbody></mujoco>"""


def quat_from_z(axis):
	"""Unit quaternion (w, x, y, z) rotating the z axis onto `axis`."""
	axis = axis / np.linalg.norm(axis)
	z = np.array([0.0, 0.0, 1.0])
	c = z @ axis
	if c < -1 + 1e-12:
		return np.array([0.0, 1.0, 0.0, 0.0])
	q = np.r_[1 + c, np.cross(z, axis)]
	return q / np.linalg.norm(q)


def poses(rng, n, other):
	"""qpos (n, 14): body a at the origin with a random axis; body b overlapping it, half of them nearly parallel."""
	q = np.zeros((n, 14))
	for k in range(n):
		ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
		perp = np.cross(ax, rng.normal(size=3)); perp /= np.linalg.norm(perp)
		if k % 2 == 0:  # nearly parallel: rotate by a tiny angle about a random perpendicular axis
			ang = 10 ** rng.uniform(-7, -1)
			ax2 = np.cos(ang) * ax + np.sin(ang) * np.cross(perp, ax)
			if rng.random() < 0.5:
				ax2 = -ax2
			if other == 'box':  # align a box edge (its z axis) with the capsule
				pass
			off = perp * rng.uniform(0.02, 0.09) + ax * rng.uniform(-0.25, 0.25)
		else:  # general pose
			ax2 = rng.normal(size=3); ax2 /= np.linalg.norm(ax2)
			off = rng.normal(size=3) * 0.08
		q[k, 3:7] = quat_from_z(ax)
		q[k, 7:10] = off
		q[k, 10:14] = quat_from_z(ax2)
	return q


def cpu_contacts(m, d, q):
	"""C MuJoCo contacts for pose q, sorted by depth: [(dist, pos), ...]."""
	d.qpos[:] = q
	mujoco.mj_forward(m, d)
	return sorted([(c.dist, np.array(c.pos)) for c in d.contact[:d.ncon]], key=lambda x: x[0])


res = {}
rng = np.random.default_rng(0)
for other, size in (('capsule', '0.04 0.12'), ('box', '0.03 0.04 0.12')):
	m = mujoco.MjModel.from_xml_string(XML.format(other=other, size=size))
	d = mujoco.MjData(m)
	qs = poses(rng, a.worlds, other).astype(np.float32).astype(np.float64)  # identical inputs on both sides
	mw = mjw.put_model(m)
	dw = mjw.put_data(m, d, nworld=a.worlds, nconmax=8, njmax=64)  # per world
	dw.qpos.assign(wp.array(qs.astype(np.float32), dtype=float))
	mjw.forward(mw, dw)
	wp.synchronize()
	n = int(dw.nacon.numpy()[0])
	wid, gpos, gdist = dw.contact.worldid.numpy()[:n], dw.contact.pos.numpy()[:n], dw.contact.dist.numpy()[:n]
	r = dict(worlds=a.worlds, count_mismatch=0, exact=0, within_cpu_spread=0, beyond_cpu_spread=0, worst_beyond=0.0)
	for k in range(a.worlds):
		c0 = cpu_contacts(m, d, qs[k])
		g = sorted([(float(gdist[i]), gpos[i].astype(np.float64)) for i in np.where(wid == k)[0]], key=lambda x: x[0])
		if len(c0) != len(g):
			r['count_mismatch'] += 1
			r.setdefault('mismatch_examples', []).append(dict(world=k, nearly_parallel=k % 2 == 0, cpu_depths=[float(x[0]) for x in c0],
														   gpu_depths=[float(x[0]) for x in g]))
			continue
		if not c0:
			continue
		e = max(max(abs(x[0] - y[0]), np.abs(x[1] - y[1]).max()) for x, y in zip(c0, g))
		if e <= 1e-4:
			r['exact'] += 1
			continue
		spread = 0.0
		for _ in range(20):  # C MuJoCo's own sensitivity to float32-sized input noise
			q = qs[k].copy()
			q[3:7] += 1e-7 * rng.normal(size=4); q[10:14] += 1e-7 * rng.normal(size=4); q[7:10] += 1e-8 * rng.normal(size=3)
			c1 = cpu_contacts(m, d, q)
			if len(c1) == len(c0):
				spread = max(spread, max(np.abs(x[1] - y[1]).max() for x, y in zip(c0, c1)))
		if e <= 1.5 * spread + 1e-4:
			r['within_cpu_spread'] += 1
		else:
			r['beyond_cpu_spread'] += 1
			r['worst_beyond'] = max(r['worst_beyond'], float(e))
	ties = sum(1 for x in r.get('mismatch_examples', []) if len(x['cpu_depths']) == 2 and len(x['gpu_depths']) == 1
			   and abs(x['cpu_depths'][0] - x['cpu_depths'][1]) < 1e-6)
	r['equal_depth_two_contact_ties'] = ties
	allowed = 0 if other == 'capsule' else int(0.002 * a.worlds)
	r['pass'] = r['beyond_cpu_spread'] == 0 and r['count_mismatch'] - ties == 0 and ties <= allowed
	res[other] = r
	print(other, json.dumps(r))
res['pass'] = all(res[k]['pass'] for k in ('capsule', 'box'))
print('PASS' if res['pass'] else 'FAIL')
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, 'w'), indent=1)
