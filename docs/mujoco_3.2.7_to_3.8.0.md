# C++ MuJoCo 3.2.7 → 3.8.0 migration

The former Play source snapshot identifies MuJoCo 3.2.7. Aar_Unilab pins the
Python and C++ runtime to the same `mujoco-uni` 3.8.0 artifact. CMake links the
smoke/runtime target directly to that package's headers and versioned library,
so a silent system-library fallback cannot occur.

Validated gates:

- load the PE01 scene and execute one C++ `mj_step`;
- require `mj_versionString() == "3.8.0"`;
- load the same models in Python 3.8.0;
- compare release-contract observations/actions and short trajectories before
  a policy is accepted for deployment.

The original viewer source is retained under `references/legacy_play_source`
for UI behavior comparison. Visual rendering and contact trajectories require
manual GPU/display acceptance; the version/link/load gate is automated.
