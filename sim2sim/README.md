# C++ Sim2Sim

日常命令、参数和两套 C++ 入口的选择见
[日常使用指南](../docs/DAILY_USAGE.md#cpp)。

This directory is the maintained, robot-neutral C++ runtime boundary. It reads
only `aar-unilab.actor.v1` releases and fails closed on unknown schemas. Tensor
names, shapes, history, hidden state, joints and control rates come from the
release manifest; they are not compile-time WE11 constants. `aar_sim2sim`
loads the ONNX actor and MuJoCo scene directly from a release, checks golden
outputs, runs the selected observation-builder adapter and writes the same
CSV/JSONL telemetry contract as Python Play.

The active interactive host provides rendering, follow camera, mouse
rotate/pan/zoom, right-drag external force, GLFW gamepad commands, a live
MuJoCo plot and diagnostic telemetry. The original-style WE11 implementation
is active under `sim2sim/we11_play/`; its `IMPORT.md` retains provenance.
Training-only reward, critic, curriculum and PPO code do not enter the generic target.

PE02 formal releases use the independent `pe02_v2` observation and position-PD
runtime in `include/aar/pe02_runtime.hpp`. The release's `robot/pe02_runtime.json`
owns home angles, gains, delay, gait and normalization; old `pe02_v1` releases
retain their torque-control contract. See the [PE02 migration audit](../docs/PE02_TRAINING_MIGRATION.md).

PE01 formal releases similarly use their own `pe01_v2` runtime in
`include/aar/pe01_runtime.hpp` and `robot/pe01_runtime.json`. The original
`pe01_v1` runtime remains compatible with existing example releases.

Build the dependency-free contract layer with:

```bash
cmake -S sim2sim -B sim2sim/build
cmake --build sim2sim/build --parallel
ctest --test-dir sim2sim/build --output-on-failure

# Headless release playback
sim2sim/build/aar_sim2sim releases/examples/pe01_flat --steps 100

# Interactive playback
sim2sim/build/aar_sim2sim releases/examples/pe01_flat \
  --steps 1000000 --interactive

# Build and validate the preserved WE11 Getup release without duplicating its
# large robot assets in Git.
python tools/build_we11_sim2sim_release.py \
  --destination /ssd/conda/cache/aar_unilab-native/we11-getup-release
sim2sim/build/aar_sim2sim \
  /ssd/conda/cache/aar_unilab-native/we11-getup-release --steps 100
python tools/compare_we11_sim2sim.py --binary sim2sim/build/aar_sim2sim \
  --release /ssd/conda/cache/aar_unilab-native/we11-getup-release
```

The generic `aar_sim2sim` target uses MuJoCo 3.8.0 and ONNX Runtime 1.22.0,
installed below `/ssd/conda/cache/aar_unilab-native`. The original-style
WE11 viewer under `we11_play/` temporarily uses MuJoCo 3.2.7; see its
[runtime notes](we11_play/README.md#运行库). These are separate runtime paths.
