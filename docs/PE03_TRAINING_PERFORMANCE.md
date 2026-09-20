# PE03 training performance

The September 20, 2026 optimization applies automatically to newly started PE03
training processes. A running Python process keeps its imported training code.
Existing checkpoints keep the same schema and resume contract. Environment
count, PPO epochs/minibatches, reward coefficients, network sizes, observation
history, action processing, and simulation/control rates are unchanged.

## GPU update optimization

`training.cache_update_inputs=true` is now the default. Each PPO minibatch's
actor/critic input and frozen velocity estimate are prepared once and reused
across the five epochs. The separate supervised phase similarly reuses gathered
history/velocity batches. Caches are local to one update and PPO caches are
released before the supervised phase, so updated encoder weights and new
rollouts always get fresh inputs. Set `training.cache_update_inputs=false` to
reduce the additional VRAM use or compare against recomputation.

`training.matmul_precision=high` is the PE03 training default, enabling Tensor
Core float32 matrix multiplication during network updates. Start normally:

```bash
bash tools/train.sh pe03_gait_fixed
```

The setting is recorded in the training configuration and checkpoints. Model
parameters, Adam state, rollout storage and exported networks remain float32.
The prior PyTorch matmul setting is restored after each update, including on
exceptions; rollout inference and export do not inherit this optimization.
The `high` mode changes matrix multiplication precision and is **not bitwise
equivalent** to `highest`. Keep `highest` when continuing a full-FP32 run with the
same numerical settings. No new training process is started automatically.

Profiling the original update at 4096 environments found about 76% of CUDA
operator time in matrix multiplication, 7% in concatenation and 5% in indexed
gathers. Fused Adam would target less than 1% of this measured time. Whole-run
GPU utilization includes waiting for CPU simulation; that number does not
measure utilization during the network update itself.

CPU and CUDA tests compare cached versus recomputed updates across two rollouts
for standing, walking and fixed gait, including commands that differ from the
last history frame. Parameters and metrics are bitwise identical in full FP32.
A separate 1024-environment CUDA comparison with the saved pre-change update
also matched bitwise. One TF32 update on that same rollout produced an action
RMSE of 0.000690 and maximum absolute action difference of 0.00333 versus FP32;
the maximum critic-value difference was 0.0000723. These are short numerical
checks, not a long-training convergence assessment.

Update-only probes reuse one real rollout, exclude the first of four updates,
and repeat modes in reverse order. They measured approximately 1.13 s without
input caching, 0.98 s with caching and FP32, and 0.64 s with caching and Tensor
Cores. Caching costs additional VRAM; see the full-loop results below for the
actual training throughput and memory peak.

Full-loop comparison on the idle machine (4096 environments, 32 threads, 24
steps, 5 epochs × 4 minibatches; 3 warmup and 8 measured iterations per run).
The following group uses the current failure thresholds, 0.08 m height and 80°
tilt, with identical task configurations, source hashes and asset hashes:

| Update mode | Collection | Network update | Total / iteration | Samples / s | Peak allocated CUDA |
|---|---:|---:|---:|---:|---:|
| Recompute inputs, FP32 | 1.420 s | 1.165 s | 2.585 s | 38,032 | 1.061 GiB |
| Cache inputs, FP32 | 1.487 s | 1.017 s | 2.504 s | 39,260 | 1.693 GiB |
| Cache inputs, Tensor Core (default) | 1.422 s | 0.630 s | 2.052 s | 47,917 | 1.693 GiB |

Caching alone reduces update time by 12.7% and improves whole-loop throughput
by 3.2%. Tensor Core mode plus caching reduces update time by **45.9%**, total
iteration time by **20.6%**, and improves throughput by **26.0%**. Input caching
adds about 0.63 GiB to peak allocated CUDA memory in this comparison. No swap was
used, and every run completed all 220 PPO plus 220 estimator optimizer steps
with finite metrics. Periodic evaluation, saving and export are excluded.

An earlier matched group using 0.15 m / 60° thresholds independently measured
2.589 s without caching, 2.490 s with FP32 caching, and 2.104 s with Tensor Cores.
The task file changed during the initial measurement series. A mixed-configuration
comparison was discarded and the second Tensor Core measurement was rerun from
`frozen_config.yaml`. Configuration/source/asset equality was checked within
each group; `summary.json` reports the groups separately.

GPU profiling, numerical comparisons, source snapshots and timing records:
`logs/benchmarks/pe03_gpu_20260920/`. Reproduce a complete-loop measurement with:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /ssd/conda/envs/aar_unilab/bin/python tools/benchmark_pe03_training.py \
  --experiment gait_fixed --envs 4096 --threads 32 --warmup 3 --iterations 8 \
  --matmul-precision high --output logs/benchmarks/pe03_tensorcore.json
```

Use `--matmul-precision highest` for FP32 and `--no-cache-update-inputs` to
disable the cache. Without a precision override the benchmark follows the
training configuration. It records both switches with the result.

Final validation for this update pass: **426 passed, 2 xfailed** in 70.66 s,
including CPU/CUDA cache equivalence, exact resume, exports and WE11 tests.
The initial full run exposed a hardcoded 0.10 m failure height in the consecutive
failure-tick test after the task threshold changed to 0.08 m. The test now places
the base 0.01 m below its configured failure threshold and still checks recovery
and exactly five consecutive failure ticks; production termination settings
were preserved. The final suite passed with the current task settings.
Changed Python files pass Ruff format/check and `git diff --check` passes.
The existing 12 formatting files, 7 Ruff errors and one `playback.py:35` mypy
error remain unchanged. Raw outputs are in `pytest_full_final.log`,
`ruff_format.log`, `ruff_check.log` and `mypy.log` in the GPU benchmark directory.

## Changes

- Collection allocates each rollout tensor once, copies observations before the
  environment can reuse their storage, and avoids the final clone/stack peak.
- Each next observation is transferred once and its critic value is reused by
  the following step. Ended episodes still use their pre-reset observations for
  bootstrapping; only those environments require an extra critic evaluation.
- Gait observations reuse the step's foot kinematics until a reset changes the
  simulator state. Full terminal history is materialized only when needed.
- Frozen velocity estimators run without an autograd graph during PPO. Their
  separate supervised optimizer remains unchanged.
- Distribution validation no longer forces repeated CUDA synchronization;
  finite-action, loss, and gradient checks remain. Reporting metrics are copied
  to the CPU together, and adaptive KL scheduling reads its scalar once per batch.

## Measured results (2026-09-20)

Hardware: Ryzen 9 7945HX (16 cores / 32 threads), RTX 4060 Laptop GPU. The
existing 4096-environment `gait_fixed` training continued throughout the tests,
so these are measurements under shared CPU/GPU load, not isolated throughput.
Both versions used 4096 environments, 32 MuJoCo threads, 24 steps, 5 PPO epochs
and 4 minibatches. Each version ran in two fresh processes, with 3 warmup and
8 measured iterations per process, ordered before / after / after / before.

| Metric | Before | After |
|---|---:|---:|
| Samples / second | 24,936 | 28,676 |
| Collection / iteration | 2.108 s | 1.945 s |
| PPO + estimator update / iteration | 1.834 s | 1.484 s |
| Total / iteration | 3.942 s | 3.428 s |
| Peak allocated CUDA memory | 1.443 GiB | 1.171 GiB |

Aggregate throughput improved **15.0%** and peak allocated CUDA memory decreased
**18.8%**. No swap was used. Individual runs varied with the background training;
these figures do not predict exact speed on an idle machine or evaluation time.

An additional within-process check alternated the original and optimized
implementations before / after / after / before every iteration, sharing the
runner, optimizer, allocator and evolving training state. After 4 warmup
iterations, 8 measured iterations per implementation yielded 26,604 → 28,804
samples/s (**8.3%**), with collection time 2.150 → 1.892 s and update time
1.545 → 1.521 s. This check reduces startup and background-load bias; the measured
overall improvement is therefore best described as **8–15% under shared load**.

Raw records, source hashes, the pre-change source snapshot and its benchmark
launcher, and the implementation diff are in
`logs/benchmarks/pe03_speed_20260920/`. `summary.json` aggregates the four runs;
`interleaved.json` contains the within-process check.

### PE03 / WE11 comparison without background training

After the original PE03 run completed all 500 iterations, both current training
stacks were measured sequentially on the same machine: 4096 environments,
32 MuJoCo threads, 24 steps, 3 warmup and 8 measured iterations, CUDA, no evaluation
or checkpoint/export I/O. WE11 uses `dr002_joystick_flat_we11/mujoco`.

| Current implementation | Collection / iteration | Update / iteration | Total / iteration |
|---|---:|---:|---:|
| PE03 fixed gait | 1.466 s | 1.155 s | **2.622 s** |
| WE11 flat | 1.817 s | 0.168 s | **1.985 s** |

The principal difference is network training. PE03 has 1140 history inputs,
512/256/128 actor and critic hidden layers, and a separate 20-minibatch estimator
update after the 20 PPO minibatches. WE11 uses 145 history inputs, a compressed
9-dimensional action-head input, 128/64/32 actor hidden layers, 256/128/64 critic
hidden layers, and includes adaptation loss in the PPO update. Their configured
networks have approximately 1.832 million versus 0.117 million parameters.

The completed original PE03 run's final 30 iterations averaged 1.649 s collection
and 1.059 s update, totaling 2.709 s. That run used pre-optimization code and a trained
policy, while these short benchmarks start fresh policies; this comparison
alone cannot establish an optimization speedup. The controlled before/after
measurements above serve that purpose. All figures exclude periodic evaluation
and saving. Raw current-stack comparison: `logs/benchmarks/pe03_we11_idle_20260920/`.

## Reproduce a measurement

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  /ssd/conda/envs/aar_unilab/bin/python tools/benchmark_pe03_training.py \
  --experiment gait_fixed --envs 4096 --threads 32 \
  --steps 24 --warmup 3 --iterations 8 \
  --output logs/benchmarks/pe03_gait_speed.json
```

Use a new output path for each run. The tool also supports `standing`, `walking`,
and `--device cpu`. It synchronizes CUDA at timing boundaries, reports rollout
and update times separately, records source/model hashes and memory peaks, and
counts samples using the configured rollout length. Evaluation, checkpointing,
and export I/O are excluded from throughput.

For an isolated speed measurement, run without another training process sharing
the GPU. Short throughput checks do not measure gait convergence or final policy
quality.

## Validation

- Full non-slow suite: **417 passed, 2 xfailed**, 79.85 s. This includes the PE03
  checkpoint/resume, gait, export and newly added rollout ownership/bootstrap
  tests for standing, walking and fixed gait with zero, sparse and full resets.
- Both WE11 flat and rough tasks composed, initialized and stepped three times;
  observation dictionaries retained their expected shapes and finite values.
- Direct comparison with the saved pre-change gait environment: fixed and
  variable gait each ran 20 steps through command resampling and episode resets.
  Observations, terminal observations, rewards, termination flags and diagnostics
  were bitwise identical.
- Direct CPU comparison of a complete PPO + estimator update on the same
  rollout: standing, walking and fixed gait produced identical parameters and
  reported metrics. Sparse terminal critic evaluations can still differ by
  floating-point rounding when batch sizes change; tests compare their numerical
  values against full-batch evaluation.
- CPU benchmark smoke passed with a nondefault 2-step rollout; all GPU benchmark
  update metrics were finite.
- Changed Python files pass Ruff format/check; `git diff --check` passes.
  Full-repository static checks retain the previously documented 12 formatting
  files, 7 Ruff errors and the single `playback.py:35` mypy error. See the existing
  [validation record](PE03_WTW_GAIT_VALIDATION.md#全仓已有静态检查问题).

## Continue an existing run

Start normally with `bash tools/train.sh pe03_gait_fixed`, or resume a saved
checkpoint with the same task and algorithm settings:

```bash
bash tools/train.sh pe03_gait_fixed \
  training.resume=/absolute/path/model_100.pt algo.max_iterations=400
```

`max_iterations` is the number of **additional** iterations: this example ends
at iteration 500. Stop the old process before starting its replacement to avoid
resource competition. Unsaved iterations after the selected checkpoint are not
recoverable. This optimization does not stop or restart an existing run.
