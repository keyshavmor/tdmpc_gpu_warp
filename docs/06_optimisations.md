# 6. Optimisations: what was done, what it gains, and the evidence that it changes nothing

*tdmpc_gpu_warp: GPU port of TD-MPC by Keyshav Mor. Based on TD-MPC by Nicklas Hansen, Xiaolong Wang and Hao Su (https://github.com/nicklashansen/tdmpc).*

Every optimisation below is opt-in (the faithful port with the original agent stays the default of `src/train.py`)
and was only accepted after the checks below. "Exact" = same physics, same maths, same algorithm; the only
permitted differences are floating-point summation order (ulp-level) and the random-number *stream*
(identical distributions).

## 6.1 Approval checklist

| # | optimisation | gain (measured) | physics | maths (deterministic update) | 200-update drift | learning curves A/B | verdict |
|---|---|---|---|---|---|---|---|
| 1 | GPU environment (mujoco_warp, all 21 tasks) | removes CPU physics + host<->device traffic; enables everything below | float32 GPU physics validated vs CPU MuJoCo (docs/03) | n/a | n/a | learning matches the paper reference and the original code on this machine (cartpole, walker; docs/results_learning.md) | **approved** |
| 2 | whole agent step in one CUDA graph (env) | env CPU time 5 ms -> 0.3-1 ms per agent step | bitwise identical to eager env | n/a | n/a | covered by 3/6 | approved |
| 3 | CUDA-graph agent (`graph_agent`, plan + update captured; multinomial sampling on GPU - replacement flag fixed, 6.4) | **2.4-4.8x (mean 3.7x)** single-run speed-up vs CPU baseline at H = 5 (with 1, 4) | unchanged | params identical to 2e-6..7e-6 after 12 updates; plan distributions identical | **bitwise identical** to the eager agent over 200 updates | covered by 5 (same code path) | **approved** |
| 4 | host pacing + blocking sync (`pace_host`) | CPU per run 1.0 -> 0.03-0.17 cores; -7 % single-run speed | unchanged | unchanged (only *when* the CPU waits) | n/a | n/a | approved |
| 5 | vectorised seeds (`train_vec.py num_seeds=S`) | one CUDA context + one N-world physics launch for all seeds: 4x more seeds per GPU, ~2.2x total throughput | per-world bitwise identical to separate envs | per-seed params identical to 3e-6..1e-4 (Adam-amplified ulps) | n/a | 5 seeds vs paper, 20k-100k, after the PER fix: cartpole p = 0.14-0.89, walker p = 0.22-0.86 | **approved** |
| 6 | noslip solver on GPU, tile Cholesky | dog physics step 2.9 -> 1.47 ms | reproduces C MuJoCo noslip (same max effect 3.011e-2) | n/a | n/a | covered by 1 | approved |
| 7 | activation checkpointing (`ckpt_unroll`) | H-term of VRAM 4-9x smaller (dog H=30: 4.1 -> 1.9 GiB, walker H=30: 208 vs 34 seeds per GPU); -10..-18 % speed | unchanged | **bitwise identical** parameters | bitwise (deterministic recompute) | identical by construction | approved |
| 8 | batched update (`batched_update`) | 1.03-1.13x at H = 5, 1.11-1.26x at H = 30 | unchanged | params within 1e-6 (walker) / 3e-5 (dog) after 12 updates | loss rel. diff <= 1.3e-4, 8e-6 at update 200; params 1e-3 (rounding amplified by Adam) | 5 vs 5 seeds A/B: cartpole p = 0.24-1.0, walker p = 0.47-0.86 - no detectable difference | **approved** |
| 9 | skip sensors in intermediate physics sub-steps (`skip_substep_sensors`) | physics step 5-12 % faster; agent step ~1.00x | **bitwise identical** trajectories (7 tasks x 100 steps) | n/a | n/a | identical by construction | approved, but negligible gain |

## 6.2 Evaluated and NOT implemented (or rejected)

| idea | why not |
|---|---|
| exact Triton GEMM + bias + ELU fusion | profile (S = 16): 60 % of GPU time is cuBLAS FP32 GEMMs, fusable elementwise kernels ~20-25 % -> at most ~15-20 %; an FP32 Triton GEMM is slower than cuBLAS on the GEMM part - not a "great" gain |
| making a single simulated world faster | ~2000 tiny kernels per agent step inside mujoco_warp (no hotspot after the noslip fix); would require rewriting mujoco_warp's pipeline |
| more processes without MPS | the driver time-slices processes: total throughput stays constant (4.1) |
| blocking sync alone | CPU still spins on a full launch queue -> host pacing needed (4) |
| TF32 / bf16 / fp16 by default | change numerics (not exact) - offered as options, see chapter 7 |
| bf16 replay buffer by default | rounds stored observations - option only |

## 6.3 Learning-mechanism verdict

**After the PER fix (section 6.4)** the vectorised CUDA-graph agent was re-run (`vec_fp32_fixed`, 5 seeds,
cartpole-swingup and walker-walk, 100k env steps, docs/results_learning.md). It is statistically
indistinguishable from the paper's reference curves at every evaluation from 20k to 100k steps (Welch test,
cartpole p = 0.14-0.89, walker p = 0.22-0.86). Examples: walker 933 +- 46 vs 954 +- 6 at 40k and 962 +- 14 vs 963 +- 5 at 100k; cartpole
854 +- 13 vs 855 +- 4 at 80k. At step 0 the untrained agent's cartpole return (124 +- 17) differs from the paper
(94 +- 8), but the original code on this machine gives the same value (127), so this is not a port effect.

**Before the fix** (runs kept and labelled in results_learning.md), all variants were also within the reference
except one point: cartpole at 80k, 832 +- 19 vs 855 +- 4, p = 0.02. With the fix that point is 854 +- 13 (p = 0.83),
so the earlier marginal deviation was plausibly the sampling bug.

The A/B tests of batched update, bf16 and fp16 against the exact agent (run before the fix, with identical sampling
on both sides) show no difference at any step (p = 0.24-1.0). The learning mechanism is unchanged by the port and
by optimisations 1-9.

## 6.4 Correction: replay-sampling replacement flag (found while documenting the code, fixed)

**The bug.** The original `ReplayBuffer.sample` draws a PER batch with
`np.random.choice(total, B, p=probs, replace=not full)`. That is, it samples **with replacement** until the buffer
has wrapped around once. For almost every TD-MPC run that means the whole run, since the buffer holds `train_steps`
transitions. The GPU-side replacement in the CUDA-graph agent (`GraphReplayBuffer`, optimisation 3) and the
vectorised agent (`VecReplayBuffer`, optimisation 5) used `torch.multinomial(probs, B, replacement=full)`, the
negation. As a result they drew each batch **without replacement** (no duplicate indices) before the buffer was full,
and with replacement afterwards.

**What was and was not affected**
- **Unchanged:** the per-draw probabilities `priority ** per_alpha`, the importance weights, and all other maths, which
  is why the deterministic equivalence tests above (fixed batches) did not see it.
- **Changed:** the batch distribution. It differs most when the batch is a large fraction of the valid transitions,
  that is early in training (first update: 512 of ~5k-10k valid start indices).
- **Not affected:** the faithful port, which uses the original buffer.

**The fix.** `replacement=not full`, a one-token change in each file. The new test
`bench/test_per_sampling.py` fills the original buffer and both GPU buffers with the same episodes and priorities and
draws 3000 batches from each:

| regime | distinct indices per batch: original | before fix (graph / vec) | after fix (graph / vec) |
|---|---|---|---|
| buffer not full, 2 episodes (990 valid) | 388.0 | 512 / 512 | 388.2 / 388.2 |
| buffer not full, 20 episodes (9900 valid) | 497.4 | 512 / 512 | 497.3 / 497.1 |
| buffer full (wrapped) | 512 | 443.7 / 443.7 | 512 / 512 |

After the fix the index frequencies match the exact PER probabilities to within the original's own sampling noise
(total-variation distance 0.0097 / 0.0099 vs 0.0093 for the original with 2 episodes), the importance weights match
to 2e-7, and zero-priority slots are never drawn.

**Consequences for the results**
* **Speed, VRAM, capacity and interference measurements:** unaffected in substance. A timing re-check of the 15
  baseline tasks at H = 5 and 30 (`bench/recheck_timing.py`, `results/sweep_h_recheck/per_fix/`) gives step times
  +2.0 % (H = 5) and +1.5 % (H = 30) on average (range -1.4 % to +5.7 %) relative to the
  sweep of docs/02, and identical peak VRAM. The speed-ups reported in docs/02 and the run-time tables therefore
  hold to within a few percent.
* **Horizon limit:** the old `H_max` (the largest horizon at which the first batch can be drawn without replacement)
  does not apply to the original algorithm. Its true limit is H <= L - 1 (L = episode length). The OOM conclusions are
  unchanged: only dog can go OOM within that range ([results_memory.md](results_memory.md)).
* **Learning curves:** the earlier vectorised runs (`vec_fp32`, `vec_fp32_opt`, `vec_bf16`, `vec_fp16`) used the
  inverted flag. They are kept in [results_learning.md](results_learning.md), labelled "before PER fix". The
  vectorised agent was re-run with the fix (`vec_fp32_fixed`, cartpole-swingup and walker-walk, 5 seeds, 100k steps).
  It matches the reference at every step (section 6.3). Full-budget 5-seed runs of the recommended single-seed
  configuration under MPS are planned.

## 6.5 Further candidates: fused Adam and compiled TOLD networks (evaluated; opt-in, not adopted)

Two more candidates for the CUDA-graph agent (`src/algorithm/tdmpc_graph.py`), both off by default:
- **`fused_adam=true`:** torch's fused Adam kernel. Same formula, one kernel per parameter group.
- **`compile_model=true`:** `torch.compile` on every TOLD sub-network (encoder, dynamics, reward, policy, Q-functions;
  online and target model), captured into the same CUDA graphs. It fuses the element-wise operations into fewer
  kernels.

Two implementation details:
- `nn.Module.compile()` silently does nothing in torch 2.11, so the bound `forward` methods are compiled instead.
- All sub-networks share one compiled wrapper, so the Dynamo recompile limit is raised (default 8) so that every
  network and batch-shape combination is compiled rather than falling back to eager.

**Equivalence** (`bench/test_graph_agent.py <task> 5 <options>`): maximum parameter difference to the original agent
after 12 deterministic updates. Plan-distribution statistics are identical in all variants.

| variant | walker-walk | cartpole-swingup |
|---|---|---|
| CUDA-graph agent (approved, optimisation 3) | 7.2e-6 | 5.0e-6 |
| + fused Adam | 1.8e-6 | 1.5e-6 |
| + compiled networks | 3.2e-5 | 5.0e-6 |
| + both | 1.7e-6 | 1.5e-6 |

All are at float-rounding level (metrics <= 1.6e-5 relative), within the range accepted for the vectorised agent.

**Speed:**
- **Single run** (`bench/recheck_timing.py`, 6 tasks x H = 5 / 30, relative to the docs/02 sweep; the post-fix
  baseline itself is +2 %):
  - fused Adam 0.997x (noise);
  - compiled 0.940x (planning ~10 % faster, update ~4 %);
  - both 0.924x.
- **5 runs in parallel under MPS** (`bench/multi_run.py`, total seed-steps/s, `results/multi_compile/`):

| task | baseline | compiled | compiled + fused Adam |
|---|---|---|---|
| walker-walk | 111.6 | 117.7 (+5.5 %) | 119.7 (+7.3 %) |
| cheetah-run | 123.7 | 132.9 (+7.5 %) | 135.9 (+9.9 %) |

**Decision: kept as validated opt-in switches, not part of the recommended baseline.**
- **Gain:** about 7-10 % throughput. For the 15-task x 5-seed schedule that is ~31 h -> ~28.5 h.
- **Why not adopt:** the gain is not large. Adopting it would require a learning-curve A/B, and it gives up the
  bit-closeness of the baseline to the original agent (fused kernels round differently).
- **Startup cost:** compiling adds to the first start of each configuration. Compiled kernels are cached in
  `$TORCHINDUCTOR_CACHE_DIR`.
