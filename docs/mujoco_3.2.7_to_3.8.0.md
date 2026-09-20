# C++ MuJoCo 3.2.7 → 3.8.0 migration

The former Play source snapshot identifies MuJoCo 3.2.7. Aar_Unilab pins the
Python and C++ runtime to the same `mujoco-uni` 3.8.0 artifact. CMake links the
smoke/runtime target directly to that package's headers and versioned library,
so a silent system-library fallback cannot occur.

Validated gates:

- load the PE01 scene and execute one C++ `mj_step`;
- require `mj_versionString() == "3.8.0"`;
- load the same models in Python 3.8.0;
- compare PE01 and WE11 release-contract observations/actions and 1/10/100-step
  trajectories before a policy is accepted for deployment. The current WE11
  Getup Python/C++ action, control and state comparison passes with zero error.

The original viewer source is retained under `sim2sim/we11_play`
for provenance. The active GLFW host has rendering, camera, drag force,
gamepad, plotting and telemetry; automated display smoke is complemented by
manual physical-controller acceptance.
