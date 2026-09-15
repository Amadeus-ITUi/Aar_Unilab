# Former C++ Play source snapshot

Imported from the dirty `pheonix-we11/play-bringup` working tree at commit
`d73e5317b69ecc0744497450cec66c3111363c12` on 2026-09-15. The source snapshot
retains the MuJoCo viewer, mouse-force interaction, gamepad, matplotlibcpp,
camera and logging implementations. Bundled shared libraries, build/log data,
replay data, policy binaries and duplicate robot assets were excluded.

Maintained, robot-neutral C++ code lives in `../../sim2sim`; this snapshot must
not become a build dependency.
