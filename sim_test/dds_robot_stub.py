#!/usr/bin/env python3
# Fake IGRIS-C robot over the REAL SDK DDS interface.
# Subscribes rt/lowcmd (MotorCmd[31] q/kp/kd + kinematic_modes from our real server),
# emulates the firmware PD with OUR MuJoCo (igris_c 29-DoF), and publishes rt/lowstate
# (joint_state[31] PJS + imu_state) back. No physical robot -> local sim2sim-over-DDS.
# Runs in the NUC-parity container (ubuntu:24.04 / py3.12) where the SDK wheel imports.
import argparse
import os
import struct
import threading
import time
import zlib

import numpy as np
import mujoco
import igris_c_sdk as igc


def _write_png(path, img):
    """Minimal PNG writer (RGB uint8) — no PIL/imageio dep in the SDK container."""
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))

    def chunk(tag, dat):
        c = tag + dat
        return struct.pack(">I", len(dat)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 1))   # level 1: fast, keeps stub near real-time
           + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(png)

NUM_MOTORS = 31
NUM_ACT = 29
NECK = [29, 30]

DEFAULT_DOF = np.array([
    0.0, 0.0, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
    0.0, 0.4, 0.0, -0.4, 0.0, 0.0, 0.0,
    0.0, -0.4, 0.0, -0.4, 0.0, 0.0, 0.0,
], np.float32)

STIFFNESS = np.array([
    100, 100, 100,
    500, 200, 50, 500, 300, 300,
    500, 200, 50, 500, 300, 300,
    50, 50, 30, 30, 5, 5, 5,
    50, 50, 30, 30, 5, 5, 5,
], np.float32)

DAMPING = np.array([
    0.8, 0.8, 0.8,
    3.0, 0.5, 0.5, 3.0, 1.5, 1.5,
    3.0, 0.5, 0.5, 3.0, 1.5, 1.5,
    0.5, 0.5, 0.15, 0.15, 0.1, 0.1, 0.1,
    0.5, 0.5, 0.15, 0.15, 0.1, 0.1, 0.1,
], np.float32)

TORQUE_LIM = np.array([
    60, 60, 60,
    150, 120, 60, 150, 90, 90,
    150, 120, 60, 150, 90, 90,
    60, 60, 60, 60, 8, 8, 8,
    60, 60, 60, 60, 8, 8, 8,
], np.float32)


def quat_to_rpy(qw, qx, qy, qz):
    sinr = 2.0 * (qw * qx + qy * qz)
    cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (qw * qy - qz * qx)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = np.arcsin(sinp)
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw], np.float32)


class CmdBuf:
    def __init__(self):
        self.lock = threading.Lock()
        self.q = None
        self.kp = None
        self.kd = None
        self.kmodes = None
        self.count = 0

    def cb(self, msg):
        try:
            motors = msg.motors()
            q = np.array([motors[i].q() for i in range(NUM_MOTORS)], np.float32)
            kp = np.array([motors[i].kp() for i in range(NUM_MOTORS)], np.float32)
            kd = np.array([motors[i].kd() for i in range(NUM_MOTORS)], np.float32)
            km = [int(x) for x in msg.kinematic_modes()]
            with self.lock:
                self.q, self.kp, self.kd, self.kmodes = q, kp, kd, km
                self.count += 1
        except Exception as e:  # noqa: BLE001
            print("[stub] lowcmd cb error:", e)

    def get(self):
        with self.lock:
            if self.q is None:
                return None
            return self.q.copy(), self.kp.copy(), self.kd.copy(), list(self.kmodes)


def main():
    ap = argparse.ArgumentParser(description="Fake IGRIS-C robot (MuJoCo) over SDK DDS")
    ap.add_argument("--xml", required=True)
    ap.add_argument("--domain_id", type=int, default=0)
    ap.add_argument("--namespace", default="")
    ap.add_argument("--pub_hz", type=float, default=300.0)
    ap.add_argument("--duration", type=float, default=60.0)
    # Opt-in recording (default off = no behavior change). To avoid perturbing the
    # real-time DDS verification loop, we only LOG qpos cheaply during the run, then
    # render all frames OFFSCREEN AFTERWARDS (EGL) — so coverage is the full motion,
    # not limited by software-render speed. Host ffmpeg assembles the PNGs to mp4.
    ap.add_argument("--record_frames_dir", default="", help="dir to dump PNG frames (opt-in)")
    ap.add_argument("--render_fps", type=float, default=30.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    model.opt.timestep = 0.001
    data = mujoco.MjData(model)
    nu = model.nu
    print("[stub] MJCF nq=%d nv=%d nu=%d" % (model.nq, model.nv, nu))
    assert nu == NUM_ACT, "expected nu=29 (neck-removed MJCF), got %d" % nu

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:7] = np.array([0, 0, 0.97, 1, 0, 0, 0], np.float32)
    data.qpos[7:7 + NUM_ACT] = DEFAULT_DOF
    mujoco.mj_forward(model, data)

    ch = igc.ChannelFactory.Instance()
    ch.Init(args.domain_id, args.namespace)
    assert ch.IsInitialized(), "ChannelFactory init failed"
    cmd = CmdBuf()
    sub = igc.LowCmdSubscriber("rt/lowcmd", igc.QosProfile.SensorData())
    assert sub.init(cmd.cb), "LowCmdSubscriber init failed"
    pub = igc.LowStatePublisher("rt/lowstate", igc.QosProfile.SensorData())
    assert pub.init(), "LowStatePublisher init failed"
    print("[stub] DDS up (domain=%d ns='%s') sub=rt/lowcmd pub=rt/lowstate." % (args.domain_id, args.namespace))
    time.sleep(0.3)

    sim_dt = 0.001
    decim = max(1, int(round(1.0 / (args.pub_hz * sim_dt))))
    steps = int(args.duration / sim_dt)

    # --- opt-in recorder: log qpos cheaply now, render offline after ---
    recording = bool(args.record_frames_dir)
    qpos_log = []
    render_decim = max(1, int(round(1.0 / (args.render_fps * sim_dt)))) if recording else 0
    if recording:
        os.makedirs(args.record_frames_dir, exist_ok=True)
        print("[stub] recording qpos (render_fps=%.0f, every %d steps) -> offline render at end"
              % (args.render_fps, render_decim))

    pd_q = DEFAULT_DOF.copy()
    pd_kp = STIFFNESS.copy()
    pd_kd = DAMPING.copy()
    got_cmd = False
    fall_z = 0.45
    for i in range(steps):
        t0 = time.time()

        c = cmd.get()
        if c is not None:
            qc, kpc, kdc, km = c
            pd_q = qc[0:NUM_ACT]
            pd_kp = kpc[0:NUM_ACT]
            pd_kd = kdc[0:NUM_ACT]
            if not got_cmd:
                got_cmd = True
                print("[stub] first lowcmd: kmodes=%s (PJS=1) kp[3]=%.0f q[3]=%.3f" % (km, pd_kp[3], pd_q[3]))

        if not got_cmd:
            # Before the policy is in control, HOLD the clean standing pose (don't free-fall).
            # Mirrors the real bring-up: robot is held (hoist / torque-off) at default until
            # the policy loop starts, so the policy takes over from a stable stand, not mid-fall.
            data.qpos[0:7] = np.array([0, 0, 0.97, 1, 0, 0, 0], np.float32)
            data.qpos[7:7 + NUM_ACT] = DEFAULT_DOF
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
        else:
            dof_pos = data.qpos[7:7 + NUM_ACT].copy()
            dof_vel = data.qvel[6:6 + NUM_ACT].copy()
            tau = (pd_q - dof_pos) * pd_kp - dof_vel * pd_kd
            tau = np.clip(tau, -TORQUE_LIM, TORQUE_LIM)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

        if i % decim == 0:
            st = igc.LowState()
            js = st.joint_state()
            ms = st.motor_state()
            qpos = data.qpos[7:7 + NUM_ACT]
            qvel = data.qvel[6:6 + NUM_ACT]
            for k in range(NUM_ACT):
                js[k].q(float(qpos[k]))
                js[k].dq(float(qvel[k]))
                ms[k].q(float(qpos[k]))
                ms[k].dq(float(qvel[k]))
            for k in NECK:
                js[k].q(0.0)
                js[k].dq(0.0)
                ms[k].q(0.0)
                ms[k].dq(0.0)
            quat = data.qpos[3:7]
            gyro = data.qvel[3:6]
            rpy = quat_to_rpy(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
            imu = st.imu_state()
            imu.quaternion([float(x) for x in quat])
            imu.gyroscope([float(x) for x in gyro])
            imu.rpy([float(x) for x in rpy])
            pub.write(st)

        if recording and i % render_decim == 0:
            qpos_log.append(data.qpos.copy())   # cheap snapshot; render later

        if i % 1000 == 0:
            z = float(data.qpos[2])
            rpy = quat_to_rpy(*[float(x) for x in data.qpos[3:7]])
            tag = "FALLEN" if z < fall_z else "up"
            print("[stub] t=%5.1fs z=%.3f roll=%+.2f pitch=%+.2f cmds=%d [%s]" % (i * sim_dt, z, rpy[0], rpy[1], cmd.count, tag))

        el = time.time() - t0
        if el < sim_dt:
            time.sleep(sim_dt - el)

    print("[stub] done. total lowcmds received=%d" % cmd.count)
    sub.stop()
    pub.stop()
    ch.Release()

    # offline render (no real-time pressure -> renders the FULL logged motion)
    if recording and qpos_log:
        print("[stub] offline-rendering %d frames..." % len(qpos_log))
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        cam.distance, cam.elevation, cam.azimuth = 3.0, -15.0, 90.0
        for idx, qp in enumerate(qpos_log):
            data.qpos[:] = qp
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            cam.lookat[:] = data.qpos[0:3]              # follow the robot root
            renderer.update_scene(data, camera=cam)
            _write_png(os.path.join(args.record_frames_dir, "frame_%05d.png" % idx),
                       renderer.render())
        renderer.close()
        print("[stub] wrote %d PNG frames -> %s" % (len(qpos_log), args.record_frames_dir))


if __name__ == "__main__":
    main()
