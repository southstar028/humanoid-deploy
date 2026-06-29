# External dependencies (referenced, not bundled)

This repository contains only my own deployment code. The components below are public or
third-party and are intentionally not committed here — install or fetch them separately.

## Manufacturer (RobROS, public)

- **IGRIS-C SDK** — [`robrosinc/igris_c_sdk_public`](https://github.com/robrosinc/igris_c_sdk_public).
  Prebuilt Python wheels are published under `dist/`; `setup_nuc.sh` installs the wheel you
  obtain there. Requires **Ubuntu 24.04 / Python 3.12** (the wheel is pinned to glibc 2.38 and
  CPython 3.12).
- **Robot description** (URDF / MJCF / meshes) —
  [`robrosinc/igris_c_description_public`](https://github.com/robrosinc/igris_c_description_public).
  Needed by the `sim_test/` MuJoCo stub and to regenerate the `server/real_params.yaml` limits.
- **Master-arm** (optional, teleoperation) —
  [`robrosinc/igris_c_masterarm_public`](https://github.com/robrosinc/igris_c_masterarm_public).

## Retargeting (third-party, MIT)

- **GMR** (general motion retargeting). `teleop/xrobot_teleop_to_igris.py` imports GMR base
  classes and an IGRIS-C registration (robot XML + parameters). Install GMR in its own
  environment, separate from the server venv; the publisher and server communicate only through
  Redis.

## Runtime (pip)

`numpy`, `onnxruntime`, `redis`, `pyyaml`, `mujoco` — installed by `scripts/setup_nuc.sh`. The
server and feeder deliberately avoid IsaacGym, PyTorch, and ROS 2.

## You must also provide

- A **29-DoF policy** (`.onnx`) matching `docs/POLICY_IO.md` — place it under `policy/`, or
  point `IGRIS_POLICY` at it.
- A **motion `.pkl`** for offline replay / `sim_test`, or use live teleoperation.
