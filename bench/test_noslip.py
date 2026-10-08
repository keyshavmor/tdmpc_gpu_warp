"""noslip solver validation (dog): GPU noslip port vs CPU MuJoCo.

noslip validation (dog): GPU (mujoco_warp + new noslip) vs CPU MuJoCo qacc on 200 random-action states,
with and without noslip. A dog-run model is pushed with large random controls (10 CPU physics steps between
samples) to produce sliding contacts; every sampled state with at least one contact is copied into
mujoco_warp and both simulators run `forward` with noslip (model default iterations) and without
(noslip_iterations=0). Reported: the effect of noslip on qacc on CPU and on GPU, and GPU-vs-CPU qacc
error with and without noslip (relative to max |qacc|). The GPU noslip is correct if its effect matches
the CPU's on the same states.

Environment: tdmpc-warp (mujoco_warp with the noslip port + dm_control; a CUDA GPU).
Usage: python bench/test_noslip.py
Output: stdout only. Results: docs/03_validation.md, section 3.3 (also docs/01_design_and_faithfulness.md).

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import warnings; warnings.filterwarnings('ignore')
import mujoco, mujoco_warp as mjw, warp as wp, numpy as np
from dm_control import suite
wp.config.quiet=True
env=suite.load('dog','run',task_kwargs={'random':0}); env.reset()
p=env.physics; mjm=p.model.ptr; mjd=p.data.ptr
rng=np.random.RandomState(0)
m=mjw.put_model(mjm)  # GPU model with noslip (dog's default noslip_iterations)
it=mjm.opt.noslip_iterations; mjm.opt.noslip_iterations=0; m0=mjw.put_model(mjm); mjm.opt.noslip_iterations=it
def cpu(nsl):
    """CPU mj_forward on a copy of the current state with noslip_iterations=nsl (model value restored afterwards)."""
    mjm.opt.noslip_iterations=nsl
    d2=mujoco.MjData(mjm); d2.qpos[:]=mjd.qpos; d2.qvel[:]=mjd.qvel; d2.ctrl[:]=mjd.ctrl; d2.act[:]=mjd.act; d2.qacc_warmstart[:]=mjd.qacc_warmstart
    mujoco.mj_forward(mjm,d2); mjm.opt.noslip_iterations=it; return d2
rows=[]
for t in range(200):
    # push dog with large random torques to generate sliding contacts
    for _ in range(10):
        mjd.ctrl[:]=rng.uniform(-1,1,mjm.nu); mujoco.mj_step(mjm,mjd)
    a=cpu(it); b=cpu(0)
    if a.ncon==0: continue
    fa=a.efc_force; fb=b.efc_force
    def g(mm):
        """GPU forward of the current CPU state with model mm (m: noslip, m0: no noslip); returns qacc."""
        d=mjw.put_data(mjm,mjd,nworld=1,nconmax=256,njmax=1024); mjw.forward(mm,d); wp.synchronize(); return d.qacc.numpy()[0]
    gq=g(m); gq0=g(m0)
    sc=np.abs(a.qacc).max()
    # columns: ncon, CPU efc force change by noslip, CPU noslip effect, GPU vs CPU (noslip), GPU vs CPU (plain), GPU noslip effect
    rows.append((a.ncon, np.abs(fa-fb).max(), np.abs(a.qacc-b.qacc).max()/sc, np.abs(gq-a.qacc).max()/sc, np.abs(gq0-b.qacc).max()/sc, np.abs(gq-gq0).max()/sc))
r=np.array(rows)
print("n states with contact:",len(r))
print("max |f_noslip - f_plain| CPU (efc force):  median %.3e max %.3e"%(np.median(r[:,1]),r[:,1].max()))
print("CPU noslip effect on qacc (rel):            median %.3e max %.3e"%(np.median(r[:,2]),r[:,2].max()))
print("GPU noslip effect on qacc (rel):            median %.3e max %.3e"%(np.median(r[:,5]),r[:,5].max()))
print("GPU(noslip) vs CPU(noslip) qacc rel err:    median %.3e p90 %.3e max %.3e"%(np.median(r[:,3]),np.percentile(r[:,3],90),r[:,3].max()))
print("GPU(plain)  vs CPU(plain)  qacc rel err:    median %.3e p90 %.3e max %.3e"%(np.median(r[:,4]),np.percentile(r[:,4],90),r[:,4].max()))
