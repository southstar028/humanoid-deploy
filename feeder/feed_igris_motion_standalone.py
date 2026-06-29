#!/usr/bin/env python3
# feed_igris_motion_standalone.py -- isaacgym-free / pose-free IGRIS motion feeder.
# Replays a 29-DOF IGRIS motion pkl and publishes the 35-dim mimic obs to the Redis
# channel the low-level server reads: action_body_unitree_g1_with_hands.
# Faithful re-impl of MotionLib.calc_motion_frame + build_mimic_obs using ONLY
# numpy + redis (no isaacgym, no torch, no pose pkg) -> runs on NUC / SDK container / any venv.
# Quat convention: xyzw. mimic(35) = [root_vel_local_xy(2), z(1), roll, pitch,
#                                     root_ang_vel_local_yaw(1), dof_pos(29)].
import argparse
import json
import pickle
import time

import numpy as np
import redis

CHANNEL = "action_body_unitree_g1_with_hands"

IGRIS_DEFAULT_MIMIC_OBS = np.concatenate([
    np.zeros(2), [0.97], np.zeros(3),
    np.array([0., 0., 0., -0.2, 0., 0., 0.4, -0.2, 0., -0.2, 0., 0., 0.4, -0.2, 0.,
              0., 0.4, 0., -0.4, 0., 0., 0., 0., -0.4, 0., -0.4, 0., 0., 0.]),
])


def quat_rotate_inverse(q, v):
    q_vec = q[:3]
    q_w = q[3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (q_vec @ v) * 2.0
    return a - b + c


def quat_conjugate(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_to_exp_map(q):
    q = q if q[3] >= 0 else -q
    xyz = q[:3]
    w = np.clip(q[3], -1.0, 1.0)
    sin_half = np.linalg.norm(xyz)
    angle = 2.0 * np.arctan2(sin_half, w)
    if sin_half < 1e-8:
        return np.zeros(3)
    return (xyz / sin_half) * angle


def euler_from_quat(q):
    x, y, z, w = q
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def slerp(q0, q1, t):
    d = q0 @ q1
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(np.clip(d, -1.0, 1.0))
    theta = theta0 * t
    s0 = np.sin(theta0 - theta) / np.sin(theta0)
    s1 = np.sin(theta) / np.sin(theta0)
    return s0 * q0 + s1 * q1


def so3_derivative(rot, dt):
    T = rot.shape[0]
    omega = np.zeros((T, 3))
    if T < 3:
        for i in range(T - 1):
            omega[i] = quat_to_exp_map(quat_mul(rot[i + 1], quat_conjugate(rot[i]))) / dt
        if T >= 2:
            omega[-1] = omega[-2]
        return omega
    for i in range(1, T - 1):
        q_rel = quat_mul(rot[i + 1], quat_conjugate(rot[i - 1]))
        omega[i] = quat_to_exp_map(q_rel) / (2.0 * dt)
    omega[0] = quat_to_exp_map(quat_mul(rot[1], quat_conjugate(rot[0]))) / dt
    omega[-1] = quat_to_exp_map(quat_mul(rot[-1], quat_conjugate(rot[-2]))) / dt
    return omega


class Motion:
    def __init__(self, pkl):
        d = pickle.load(open(pkl, "rb"))
        self.root_pos = np.asarray(d["root_pos"], dtype=np.float64)
        self.root_rot = np.asarray(d["root_rot"], dtype=np.float64)  # xyzw
        self.dof_pos = np.asarray(d["dof_pos"], dtype=np.float64)
        self.fps = float(d["fps"])
        self.dt = 1.0 / self.fps
        self.n = self.root_pos.shape[0]
        self.length = self.dt * (self.n - 1)
        self.root_vel = np.gradient(self.root_pos, self.dt, axis=0)
        self.root_ang_vel = so3_derivative(self.root_rot, self.dt)

    def frame_blend(self, t):
        t = max(0.0, min(t, self.length))
        f = t / self.dt
        i0 = min(int(np.floor(f)), self.n - 1)
        i1 = min(i0 + 1, self.n - 1)
        return i0, i1, f - i0

    def mimic_obs(self, t):
        i0, i1, b = self.frame_blend(t)
        root_pos = (1 - b) * self.root_pos[i0] + b * self.root_pos[i1]
        root_rot = slerp(self.root_rot[i0], self.root_rot[i1], b)
        dof_pos = (1 - b) * self.dof_pos[i0] + b * self.dof_pos[i1]
        root_vel = self.root_vel[i0]
        root_ang_vel = self.root_ang_vel[i0]
        roll, pitch, _ = euler_from_quat(root_rot)
        root_vel_local = quat_rotate_inverse(root_rot, root_vel)
        root_ang_vel_local = quat_rotate_inverse(root_rot, root_ang_vel)
        return np.concatenate([
            root_vel_local[:2],
            [root_pos[2]],
            [roll, pitch],
            [root_ang_vel_local[2]],
            dof_pos,
        ])


def main():
    ap = argparse.ArgumentParser(description="isaacgym-free IGRIS motion feeder")
    ap.add_argument("--motion_file", required=True)
    ap.add_argument("--redis_ip", default="localhost")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--set_exit_on_end", action="store_true")
    args = ap.parse_args()

    r = redis.Redis(host=args.redis_ip, port=6379, db=0)
    r.ping()
    r.set("motion_exit_signal", "0")

    m = Motion(args.motion_file)
    control_dt = 0.02
    num_steps = int(m.length / control_dt)
    print("[feeder] %s len=%.2fs fps=%.1f -> %d steps @ dt=%.3f"
          % (args.motion_file, m.length, m.fps, num_steps, control_dt))

    def publish(t_step):
        obs = m.mimic_obs(t_step * control_dt)
        r.set(CHANNEL, json.dumps(obs.tolist()))
        r.set("action_hand_left_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        r.set("action_hand_right_unitree_g1_with_hands", json.dumps(np.zeros(7).tolist()))
        r.set("action_neck_unitree_g1_with_hands", json.dumps(np.zeros(2).tolist()))
        return obs

    try:
        while True:
            for s in range(num_steps):
                t0 = time.time()
                o = publish(s)
                print("[feeder] step %4d/%d z=%.3f r/p=%+.3f/%+.3f"
                      % (s, num_steps, o[2], o[3], o[4]), end="\r")
                el = time.time() - t0
                if el < control_dt:
                    time.sleep(control_dt - el)
            if not args.loop:
                break
        print("\n[feeder] motion done.")
        if args.set_exit_on_end:
            r.set("motion_exit_signal", "1")
    except KeyboardInterrupt:
        print("\n[feeder] interrupted.")
    finally:
        r.set(CHANNEL, json.dumps(IGRIS_DEFAULT_MIMIC_OBS.tolist()))


if __name__ == "__main__":
    main()
