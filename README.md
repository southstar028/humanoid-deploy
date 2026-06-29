# Real-Robot RL Deployment — Validated Without the Robot

**The entire control path — policy inference, DDS comms, and the safety bring-up — is
exercised and validated against a *simulated* robot over the real SDK's DDS interface, before
any hardware is ever powered on.** That robot-free "sim2sim-over-DDS" gate is the core of this
project.

A sim-to-real deployment stack that runs a **29-DoF whole-body RL tracking policy** on a
**humanoid robot's onboard computer** — a CPU-only NUC (Ubuntu 24.04 / Python 3.12, no GPU).
It comprises a low-level control server that runs the policy on **CPU** over the robot's
**CycloneDDS** SDK, an offline motion feeder, a live VR-teleoperation publisher, and — the
part I'm most proud of — a **robot-free "sim2sim-over-DDS" harness** that exercises the
*entire* control path before the robot is ever powered on.

> **Scope of this repository.** This is a portfolio extract of my own deployment and
> verification engineering. The trained policy weights and the manufacturer's binary SDK are
> not distributed here — see [What is not included](#what-is-not-included). Everything present
> is code I wrote: the sim-to-real server, the feeder, the teleop publisher, and the DDS test
> harness.

---

## What this demonstrates

- **Sim-to-real boundary engineering.** The real-robot server mirrors a MuJoCo simulation
  server one-to-one and replaces only the physical interface: simulator `qpos/qvel` becomes
  `rt/lowstate` (joint state + IMU), and computed torque becomes a `q/kp/kd` command that the
  **robot firmware closes as a high-rate PD loop**.
- **Safety-first bring-up.** A staged, human-confirmed power-on sequence (state check → BMS/
  motor init → torque on → low-level mode → ramp-to-home → policy loop), backed by joint-limit
  clamping, velocity rate-limiting, a lowstate watchdog, and damping shutdown on exit.
- **Verification without hardware.** A MuJoCo "fake robot" stub speaks the SDK's own DDS
  topics, so the *unmodified* production server can be validated end-to-end — serialization,
  namespaces, kinematic modes, and closed-loop tracking — with no robot in the loop.
- **A lean runtime.** Inference is CPU `onnxruntime`; the server and feeder need neither
  IsaacGym, PyTorch, nor a ROS 2 build on the robot's onboard computer.

## Architecture

```
  PICO VR ──┐ (live teleop)
            │
  teleop/xrobot_teleop_to_igris.py ──┐
                                      ├─► Redis (reference) ─► server/server_low_level_igris_real.py ─► robot
  feeder/feed_igris_motion_standalone.py ──┘ (offline replay)        │  rt/lowcmd  ▲ rt/lowstate
                                                                     └── igris_c_sdk (CycloneDDS)
```

The reference (a per-step tracking target) is delivered over Redis by **either** the live
PICO teleop publisher **or** the offline feeder — the two are interchangeable and differ only
upstream of Redis. The server reads the reference and the robot state, runs ONNX inference,
and publishes PD targets; the firmware closes the PD loop.

## Repository layout

| path | description |
|---|---|
| `server/server_low_level_igris_real.py` | real-robot low-level policy server (SDK DDS, staged bring-up, safety guards) |
| `server/real_params.yaml` | per-joint limits, derived from the public URDF |
| `feeder/feed_igris_motion_standalone.py` | offline motion replay — pure NumPy, no IsaacGym/torch |
| `feeder/convert_pico_for_feeder.py` | converts a captured motion `.pkl` to a NumPy-agnostic format |
| `teleop/xrobot_teleop_to_igris.py` | live PICO → IGRIS-C teleoperation publisher (uses GMR retargeting) |
| `sim_test/dds_robot_stub.py` | MuJoCo "fake robot" that speaks the SDK's `rt/lowcmd` / `rt/lowstate` topics |
| `sim_test/Dockerfile.realtest` | onboard-parity test image (Ubuntu 24.04 / Python 3.12) |
| `scripts/` | `setup_nuc.sh` · `run_sim_test.sh` · `run_real.sh` |
| `docs/POLICY_IO.md` | the policy's observation/action contract — the code is fully legible without the weights |
| `DEPENDENCIES.md` | where the external (public) components come from |

## Project status & validation

Everything that can be verified without the physical robot has been verified. On the adopted
policy and this exact package:

- **SDK conformance** — full symbol/API check against the public SDK, including kinematic-mode
  persistence — pass.
- **DDS loopback** (production server ↔ MuJoCo stub, no robot) — stable standing, no fall — pass.
- **sim2sim parity gate** — 60 s stand with no fall and sub-degree jitter, upright tracking
  through a turning gait, and a dynamic-transition clip — pass.
- **Package self-test** — offline VR-captured motion tracked end-to-end, in-container — pass.

Two items remain that *require the physical robot* and were intentionally left for on-site
commissioning: confirming the live DDS namespace / network interface, and the IMU sign/frame
cross-check on the first boot frame. The staged bring-up and `--dry_run_no_torque` mode exist
precisely to close these safely.

## Running it

With the external components in place (see `DEPENDENCIES.md`) and a 29-DoF policy provided:

```bash
# 1) one-time environment (robot NUC, or any Ubuntu 24.04 host)
bash scripts/setup_nuc.sh && source ~/igris_deploy_venv/bin/activate

# 2) robot-free pre-flight (dev host + Docker): production server vs. MuJoCo stub, over DDS
bash scripts/run_sim_test.sh                  # standing
bash scripts/run_sim_test.sh <motion.pkl>     # track an offline motion

# 3) real robot — only after the gate passes (read Safety first)
bash scripts/run_real.sh --dry_run_no_torque  # publish lowcmd with motors OFF
bash scripts/run_real.sh                       # staged, human-confirmed bring-up
```

## Safety

The real server never simply "runs." It performs a staged bring-up in which a human confirms
each step (`y/N`): lowstate received with torque off → BMS/motor init → torque on → low-level
joint control → two-second ramp to the default pose → policy loop. Operating procedure: robot
on a hoist, e-stop in hand, and the `sim_test` gate passed first. Active guards: joint-limit
clamp, velocity rate-limit, lowstate watchdog, and a damping shutdown on Ctrl-C or any
exception.

## What is not included

This is commissioned-project work, so proprietary and bulky artifacts are **referenced rather
than bundled** (details in `DEPENDENCIES.md`):

| not included | reason | source |
|---|---|---|
| Trained policy weights | commissioned research output | interface specified in `docs/POLICY_IO.md` |
| `igris_c_sdk` wheel | manufacturer binary | [`robrosinc/igris_c_sdk_public`](https://github.com/robrosinc/igris_c_sdk_public) (`dist/`) |
| Robot URDF / MJCF / meshes | large; manufacturer-published | [`robrosinc/igris_c_description_public`](https://github.com/robrosinc/igris_c_description_public) |
| GMR retargeting package | third-party (MIT) | the GMR project (imported by the teleop publisher) |
| Demo / loopback videos | large binaries | linked separately |

## Tech stack

Python · ONNX Runtime (CPU) · CycloneDDS (via the IGRIS-C SDK) · MuJoCo · Redis · Docker

## License

Released under the MIT License (see `LICENSE`). Inline code comments are in Korean (original
engineering notes); all documentation here is in English.
