"""Print a one-line summary per task of the environment-validation results.

Reads results/validation/<task>.json written by bench/validate_env.py (checks A-E: obs/reward function
error, reset-distribution z-score and std ratio, open-loop trajectory error vs CPU MuJoCo and the
CPU-vs-CPU chaos control, random-policy returns, contact / constraint-row capacity, env step timing)
and prints them as compact text lines. The other JSON files in that directory (divergence study,
batched-env test) are skipped. Pure post-processing: no GPU needed, any env with Python 3 works
(normally tdmpc-warp).

Usage: python bench/summarize_validation.py
Output: stdout only. The numbers are reported in docs/03_validation.md, section 3.1.

Part of tdmpc_gpu_warp - TD-MPC on GPU MuJoCo (MuJoCo Warp).
Author of the GPU port: Keyshav Mor.
Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su, "Temporal Difference Learning for Model
Predictive Control" (ICML 2022); original code: https://github.com/nicklashansen/tdmpc
(MIT License, Copyright (c) 2021 Nicklas Hansen).
"""
import sys, json, glob, os
ROOT = os.path.join(os.path.dirname(__file__), '..')
# one line per task: A obs/reward err | B reset z, std ratio | C trajectory err @1/10/50 + chaos control | returns | D capacity | E timing (ms)
for f in sorted(glob.glob(os.path.join(ROOT, 'results', 'validation', '*.json'))):
    if any(k in f for k in ('divergence', 'batched')): continue
    d = json.load(open(f))
    print(f"{d['task']:24s} A obs={d['A_obs_max_abs_err']:.1e} rew={d['A_reward_max_abs_err']:.1e} | B z={d['B_reset_mean_z_max']:.2f} sd=[{d['B_reset_std_ratio_minmax'][0]:.2f},{d['B_reset_std_ratio_minmax'][1]:.2f}] | C init={d['C_initial_obs_max_abs_err']:.1e} e1={d['C_rel_obs_err_after_steps']['1']:.1e} e10={d['C_rel_obs_err_after_steps']['10']:.1e} e50={d['C_rel_obs_err_after_steps']['50']:.1e} chaos10={d.get('C_cpu_chaos_rel_err_after_steps',{}).get('10',float('nan')):.1e} chaos50={d.get('C_cpu_chaos_rel_err_after_steps',{}).get('50',float('nan')):.1e} | Rg={[round(x) for x in d['C_random_policy_return_gpu']]} Rc={[round(x) for x in d['C_random_policy_return_cpu']]} | D con {d['D_max_contacts']}/{d['D_nconmax']} efc {d['D_max_efc_rows']}/{d['D_njmax']} | E gpu {d['E_gpu_env_step_ms']:.2f} cpu {d['E_cpu_env_step_ms']:.2f} reset {d['E_gpu_reset_ms']:.1f}")
