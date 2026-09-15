# C++ Sim2Sim

This directory is the maintained, robot-neutral C++ runtime boundary. It reads
only `aar-unilab.actor.v1` releases and fails closed on unknown schemas. Tensor
names, shapes, history, hidden state, joints and control rates come from the
release manifest; they are not compile-time WE11 constants. `aar_sim2sim`
loads the ONNX actor and MuJoCo scene directly from a release, checks golden
outputs, runs the selected observation-builder adapter and writes the same
CSV/JSONL telemetry contract as Python Play.

The active interactive host provides rendering, follow camera, mouse
rotate/pan/zoom, right-drag external force, GLFW gamepad commands, a live
MuJoCo plot and diagnostic telemetry. The former implementation remains under
`references/legacy_play_source/` for provenance. Training-only reward, critic,
curriculum and PPO code do not enter this target.

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

MuJoCo 3.8.0 and ONNX Runtime 1.22.0 runtime distributions are installed below
`/ssd/conda/cache/aar_unilab-native`; no runtime binaries are committed.
