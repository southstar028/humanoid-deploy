"""
IGRIS-C PICO VR teleoperation publisher (sim2sim).

Thin subclass of the G1 publisher (xrobot_teleop_to_robot_w_hand.py):
  PICO/XRoboToolkit -> GMR online retarget (tgt_robot="igris_c") -> 35-dim mimic obs
  -> Redis channel `action_body_unitree_g1_with_hands` (the contract the sim server reads).

Differences vs the G1 publisher (all isolated here; the tracked G1 file is untouched):
  * GMR target = "igris_c"  (registered in GMR params.py / xrobot_to_igris_c.json).
  * GMR igris_c qpos has 31 body DOFs (neck at tail). IGRIS policy is 29-DOF (neck
    frozen), so we drop the last 2 DOFs -> 29, matching the ONNX / sim server order
    (waist3 -> Lleg6 -> Rleg6 -> Larm7 -> Rarm7).
  * IGRIS has NO dexterous hands -> hand retargeting / pinch path disabled (hand
    channels carry zeros; neck channel carries zeros — neck is frozen).
  * MuJoCo preview viewer + base use the igris_c asset.

Redis channel names keep the G1 suffix on purpose: that is the deploy-server contract;
only the dof_pos content (IGRIS 29-DOF) differs.

conda env: gmr (see teleop_igris.sh). Human-height param preserved (--actual_human_height).

Usage:
  python xrobot_teleop_to_igris.py --actual_human_height 1.6 --redis_ip localhost --target_fps 100
"""
import argparse
import json
import os
import time

import numpy as np

# Reuse the G1 publisher machinery wholesale.
import xrobot_teleop_to_robot_w_hand as g1tele
from xrobot_teleop_to_robot_w_hand import XRobotTeleopToRobot
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from data_utils.rot_utils import euler_from_quaternion_np, quat_rotate_inverse_np


# IGRIS robot key for GMR assets/IK config (registered in params.py).
IGRIS_ROBOT = "igris_c"
# The deploy-server contract still listens on the G1-with-hands channel; for
# DEFAULT_MIMIC_OBS / hand-pose dict lookups inside the base class we keep this key.
COMPAT_KEY = "unitree_g1_with_hands"
# Redis action-channel suffix the sim server reads. Default = my server
# (_unitree_g1_with_hands). Set IGRIS_REDIS_SUFFIX=igris_c_29 to feed the COLLEAGUE
# sim server (server_low_level_igris_c_29_sim.py reads action_body_igris_c_29).
_REDIS_SUFFIX = os.environ.get("IGRIS_REDIS_SUFFIX", "unitree_g1_with_hands")

# --- P0-2 (spec 07 §6): IGRIS idle/default mimic obs ---------------------------------
# The base class publishes DEFAULT_MIMIC_OBS[robot_name] on the *entire* idle state,
# pause fallback, interpolation start and exit ramp. The stock entry is G1 content
# (z=0.8 + G1-ordered crouch pose) which is garbage for the IGRIS policy
# (e.g. interpreted as waist_yaw=-0.2).
# v3 (2026-06-11): AUTHORITATIVE default pose from the colleague's training repo
# (twist2_arc igris_c_29 config default_joint_angles; their deploy server resets to
# z=0.97, identity quat, this dof). The all-zeros "neutral" was OOD (fell in ~1.4s);
# same values as default_dof_pos/stand_dof_pos in server_low_level_igris_sim.py.
IGRIS_STAND_DOF = np.array([
    0.0, 0.0, 0.0,                            # waist y/r/p
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,           # L-leg
    -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,           # R-leg
    0.0, 0.4, 0.0, -0.4, 0.0, 0.0, 0.0,       # L-arm
    0.0, -0.4, 0.0, -0.4, 0.0, 0.0, 0.0,      # R-arm
])
IGRIS_DEFAULT_MIMIC_OBS = np.concatenate([[0., 0., 0.97, 0., 0., 0.], IGRIS_STAND_DOF])
# DEFAULT_MIMIC_OBS is the shared dict object from data_utils.params (the base module
# imported the same reference), so this one runtime entry-patch covers every base-class
# code path without touching the tracked G1 file.
g1tele.DEFAULT_MIMIC_OBS[COMPAT_KEY] = IGRIS_DEFAULT_MIMIC_OBS
# --------------------------------------------------------------------------------------


def extract_mimic_obs_igris(qpos, last_qpos, dt=1.0 / 30.0):
    """35-dim IGRIS mimic obs from the GMR igris_c qpos.

    GMR igris_c qpos = [root_pos(3), root_quat(4), 31 body dofs (neck last 2)].
    IGRIS policy is 29-DOF (neck frozen) -> take the first 29 of qpos[7:].
    Layout matches build_mimic_obs / the sim server:
      [vx,vy (local), z, roll, pitch, yaw_ang_vel (local), dof_pos(29)]
    """
    root_pos, last_root_pos = qpos[0:3], last_qpos[0:3]
    root_quat, last_root_quat = qpos[3:7], last_qpos[3:7]
    robot_joints = qpos[7:].copy()
    # Drop neck (last 2) -> 29-DOF body in IGRIS order.
    robot_joints_29 = robot_joints[:29]

    base_vel = (root_pos - last_root_pos) / dt
    base_ang_vel = g1tele.quat_diff_np(last_root_quat, root_quat, scalar_first=True) / dt
    roll, pitch, yaw = euler_from_quaternion_np(root_quat.reshape(1, -1), scalar_first=True)
    base_vel_local = quat_rotate_inverse_np(root_quat, base_vel, scalar_first=True)
    base_ang_vel_local = quat_rotate_inverse_np(root_quat, base_ang_vel, scalar_first=True)

    mimic_obs = np.concatenate([
        base_vel_local[:2],        # xy velocity (2)
        root_pos[2:3],             # z (1)
        roll, pitch,               # roll, pitch (2)
        base_ang_vel_local[2:3],   # yaw angular velocity (1)
        robot_joints_29,           # 29 dof
    ])
    return mimic_obs


class XRobotTeleopToIgris(XRobotTeleopToRobot):
    def __init__(self, args):
        super().__init__(args)
        # Point the local preview viewer + base at the IGRIS asset.
        self.xml_file = ROBOT_XML_DICT[IGRIS_ROBOT]
        self.robot_base = ROBOT_BASE_DICT[IGRIS_ROBOT]
        # IGRIS has no dex hands; force pinch off regardless of CLI.
        self.args.pinch_mode = False
        self.state_machine.use_pinch = False

    def setup_retargeting_system(self):
        """Online retargeting straight to igris_c."""
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot=IGRIS_ROBOT,
            actual_human_height=self.args.actual_human_height,
        )
        print(f"Retargeting system initialized (tgt_robot={IGRIS_ROBOT})")

    def process_retargeting(self, smplx_data):
        """Same as base, but build the IGRIS 29-DOF mimic obs (drops neck)."""
        if smplx_data is None or self.retarget is None:
            return None, None
        current_time = time.time()
        self.measured_dt = current_time - self.last_time
        self.last_time = current_time
        qpos = self.retarget.retarget(smplx_data, offset_to_ground=True)
        if self.last_qpos is not None:
            current_retarget_obs = extract_mimic_obs_igris(qpos, self.last_qpos, dt=self.measured_dt)
        else:
            current_retarget_obs = g1tele.DEFAULT_MIMIC_OBS[COMPAT_KEY]
        self.last_qpos = qpos.copy()
        return qpos, current_retarget_obs

    def send_to_redis(self, mimic_obs, neck_data=None):
        """Publish 35-dim IGRIS mimic obs on the server's channel. Neck = G1-parity:
        publish the base-computed neck_data (human_head_to_robot_neck) like the G1
        publisher does; the IGRIS sim server simply ignores the neck channel (29-DOF,
        neck welded). Hands = zeros (no dex hands)."""
        sfx = _REDIS_SUFFIX
        if self.redis_client is not None and mimic_obs is not None:
            assert len(mimic_obs) == 35, f"Expected 35 mimic obs dims, got {len(mimic_obs)}"
            self.redis_pipeline.set(f"action_body_{sfx}",
                                    json.dumps(np.asarray(mimic_obs).tolist()))
        if self.redis_client is not None:
            # IGRIS: no dex hands -> zeros (server ignores hand channels for body policy).
            self.redis_pipeline.set(f"action_hand_left_{sfx}", json.dumps(np.zeros(7).tolist()))
            self.redis_pipeline.set(f"action_hand_right_{sfx}", json.dumps(np.zeros(7).tolist()))
        # neck: G1 방식 — base가 계산한 neck_data 발행(없으면 [0,0]).
        if neck_data is not None:
            self.redis_pipeline.set(f"action_neck_{sfx}", json.dumps(neck_data))
        else:
            self.redis_pipeline.set(f"action_neck_{sfx}", json.dumps([0.0, 0.0]))
        self.redis_pipeline.set("t_action", int(time.time() * 1000))
        self.redis_pipeline.execute()


def parse_arguments():
    parser = argparse.ArgumentParser(description="IGRIS-C PICO VR teleop publisher (sim2sim)")
    parser.add_argument("--redis_ip", type=str, default="localhost")
    parser.add_argument("--actual_human_height", type=float, default=1.6,
                        help="실제 사용자 키(미터). PICO 추정 오차로 실제보다 약간 작게 잡는 게 안정적.")
    parser.add_argument("--target_fps", type=int, default=100)
    parser.add_argument("--measure_fps", type=int, default=0)
    parser.add_argument("--smooth", action="store_true",
                        help="teleop mimic obs sliding-window smoothing")
    parser.add_argument("--smooth_window_size", type=int, default=5)
    parser.add_argument("--record_video", action="store_true")
    # neck frozen on IGRIS; kept for base-class compatibility (unused for publishing).
    parser.add_argument("--neck_retarget_scale", type=float, default=0.0)
    # IGRIS has no dex hands; flag accepted but forced off.
    parser.add_argument("--pinch_mode", action="store_true", default=False)
    args = parser.parse_args()
    # Base class indexes DEFAULT_MIMIC_OBS / DEFAULT_HAND_POSE by args.robot.
    args.robot = COMPAT_KEY
    return args


if __name__ == "__main__":
    args = parse_arguments()
    teleop = XRobotTeleopToIgris(args)
    teleop.run()
