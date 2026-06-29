# Policy I/O contract

The trained policy weights are not distributed in this repository (commissioned research
output). This document specifies the interface the server expects, so the deployment stack is
fully legible without the weights and a compatible policy can be supplied.

## Model

- **Format** — ONNX, executed on CPU via `onnxruntime`.
- **Observation** — a `1432`-dimensional vector (proprioception plus the tracking-reference
  features).
- **Action** — `29`-dimensional, one target offset per controlled joint.
- **Control rate** — 50 Hz (`--policy_frequency`).
- **Action scaling** — `pd_target = clip(action, -10, 10) * action_scale + default_dof_pos`.
  `action_scale` is a policy-specific training constant; inject it with `IGRIS_ACTION_SCALE`
  so deployment matches the value the policy was trained with.

## Reference (tracking) channel

The per-step reference is delivered over **Redis** as a 35-dimensional vector, published by
either the live teleop publisher or the offline feeder. The server reads it each control step
and falls back to a standing hold when no fresh reference is present.

## 29 → 31 joint mapping

The policy controls 29 DoF while the robot exposes 31 motors:

- indices `0..28` — identity mapping, in URDF body-joint order.
- indices `29, 30` — neck (pitch/yaw) — held at `q = 0` with a small hold gain, not driven by
  the policy.

State is read in the SDK's joint-state (PJS) space; commands are sent as `q + kp + kd` per
motor, and the **firmware closes the high-rate PD loop** (the server does not compute torque).
PD gains are carried in the server to match those used during training.

## IMU

The server uses the SDK IMU's `rpy()` directly and cross-checks the quaternion order on the
first boot frame (`--imu_quat_order`).
