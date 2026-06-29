#!/usr/bin/env python3
# =============================================================================
# server_low_level_igris_real.py   ⚠️ DRAFT — 별도 파일(레포 무수정)
#
#   IGRIS-C 29-DoF WBC 트래킹 정책의 **실로봇 sim2real 끝단 래퍼**.
#   server_low_level_igris_sim.py 를 1:1 미러링하되, MuJoCo 루프를
#   igris_c_sdk(rt/lowcmd · rt/lowstate, CycloneDDS) 로 교체한 것.
#
#   ★★ 하드웨어는 항상 게이트. 이 파일은 "코드"일 뿐, 사람이 e-stop을 쥐고
#      단계별로 직접 가동한다. sim2sim 패리티 게이트를 통과하기 전엔 실기 금지. ★★
#
#   sim 서버와의 유일한 본질적 차이(= sim2real 경계면):
#     - 상태: MuJoCo qpos/qvel  →  rt/lowstate(joint_state[PJS].q/dq + imu_state.rpy)
#     - 명령: 우리가 torque 계산(ctrl[:]=τ)  →  q+kp+kd 를 rt/lowcmd 로 보내고
#             **펌웨어가 고속 PD를 닫음**(tau=0). stiffness/damping = 펌웨어 PD 게인.
#     - 29(policy) → 31(모터) 매핑: index 0..28 항등(순서 동일), 29/30=Neck 고정(q=0).
#     - 추가 안전: 관절 한계 클램프(params.yaml 31), 속도 레이트리밋, 워치독,
#       램프-투-HOME, e-stop, try/finally 댐핑 — 업스트림 TWIST2엔 없음(우리가 추가).
#
#   SDK 2.0.2 대조 완료 (igris_c_sdk_public: python/client.py + cyclonedds_low_level_control.cpp):
#     [OK API] ChannelFactory / IgrisC_Client / Low{State,Cmd}{Subscriber,Publisher} /
#              LowCmd.motors()[i].{id,q,dq,tau,kp,kd} — 전부 1:1 일치(추측 아님).
#     [OK C1]  IMUState.rpy() 직접 사용(quaternion 순서 우회). quaternion()은 부팅 cross-check만.
#     [OK C2]  LowCmd.kinematic_modes(uint8[5]) = PJS(=1). cpp: kinematic_modes().fill(PJS).
#     [OK MS->PJS] 상태는 joint_state()(PJS=Pitch/Roll) 사용. motor_state()(MS=발목 Out/In) 아님.
#   SDK 예제/소스(sdk_inspect/igris_c_sdk_public-main)로 추가 해소 (2026-06-24):
#     [DOF순서 OK] cyclonedds_low_level_control.cpp kp 주석 = Waist3→L다리6→R다리6→L팔7→R팔7→Neck2
#              (=URDF 0_Waist_Yaw..30_Neck_Pitch, 우리 매핑과 일치). idx0..28 항등 + 29/30 neck 확정.
#     [C4 OK] PD 게인 단위 = leg/arm 값이 SDK 예제와 글자그대로 동일 → 단위 동일 확정.
#              waist만 우리=100/100/100(학습값) vs SDK데모=50/25/25 (값 선택 차이, 100 유지).
#     [한계 OK] 실 관절한계 = igris_c_29.urdf(body)+igris_c.urdf(neck) → gen_real_params.py → real_params.yaml.
#   실기에서만 확정(런북, 첫 접속/1프레임):
#     [C3] DDS namespace 패턴=igris_c_<robot_id>(README), 기본 인자 "igris_c_<robot_id>" — 실기 robot_id 확인.
#          NIC / CYCLONEDDS_URI 환경값.
#     [C1] IMU rpy() 부호/프레임의 sim(quatToEuler) 일치 — 내장 부팅대조로 첫 lowstate 1프레임에 확인.
#   SDK가 줄 수 없는 것: action_scale(정책 학습값) — IGRIS_ACTION_SCALE로 주입. 정책 선택은 게이트 결과.
# =============================================================================

from __future__ import annotations
import argparse, json, os, sys, threading, time
from collections import deque
import numpy as np

# --- sim 서버의 obs 유틸 재사용 (deploy_real 를 path 에 추가) -------------------
# IGRIS_TWIST2_DEPLOY 로 위치 override (온보드/도커마다 경로 다름).
TWIST2_DEPLOY = os.environ.get(
    "IGRIS_TWIST2_DEPLOY",
    os.path.expanduser("~/twist2/deploy_real"))
if TWIST2_DEPLOY not in sys.path:
    sys.path.insert(0, TWIST2_DEPLOY)
try:
    from data_utils.rot_utils import quatToEuler  # noqa: E402  (sim 과 동일 함수)
except Exception:  # torch 없는 실기/도커: quatToEuler는 부팅 cross-check 1줄에만 쓰임 -> numpy fallback
    def quatToEuler(quat):
        qw, qx, qy, qz = [float(x) for x in quat]
        sinr = 2.0 * (qw * qx + qy * qz)
        cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        sinp = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
        siny = 2.0 * (qw * qz + qx * qy)
        cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        return np.array([np.arctan2(sinr, cosr), np.arcsin(sinp), np.arctan2(siny, cosy)], np.float32)

import redis  # noqa: E402

try:
    import onnxruntime as ort
except ImportError:
    ort = None

import igris_c_sdk as igc_sdk  # noqa: E402

LOWSTATE_TOPIC = "rt/lowstate"
LOWCMD_TOPIC = "rt/lowcmd"
NUM_MOTORS = 31
NUM_ACTIONS = 29                  # policy / body DoF (목 제외, G1-패리티)
NECK_IDX = [29, 30]               # 31-모터 시스템순서의 Neck_Yaw, Neck_Pitch (고정)

# 29-DoF policy 순서 == 31-모터 시스템순서의 0..28 (검증: igris_c_v2_original.xml /
# params.yaml / igris_c_29.xml actuator 순서 동일). 따라서 매핑은 항등.
POLICY_TO_MOTOR = list(range(NUM_ACTIONS))   # motor[i] = policy[i], i in 0..28

# LowCmd.msg: KINEMATIC_MODE_MS=0, KINEMATIC_MODE_PJS=1 ; field = uint8[5] kinematic_modes.
N_KINEMATIC_MODES = 5
# Neck (motors 29,30) hold gains so the head stays upright at q=0 (kp=0 -> floppy head).
# Values from SDK cpp low-level example (Neck kp 2.0/5.0, kd 0.05/0.1).
NECK_KP = [2.0, 5.0]
NECK_KD = [0.05, 0.1]


def _set_kinematic_modes_pjs(cmd):
    """Set all 5 kinematic_modes to PJS.

    VERIFIED against wheel 2.0.2 (sdk_smoke2.py): the field default is MS(0), and the ONLY
    form that persists into cmd is the setter overload taking KinematicMode ENUM objects:
        kinematic_modes(Sequence[KinematicMode], FixedSize(5)) -> None
    Index-assign on the returned list does NOT persist (returns a copy); plain ints raise
    TypeError; attribute-assign is read-only. So mis-setting silently leaves MS = ankle
    Out/In space = wrong for our PJS-trained policy. Use the enum setter, no fallbacks."""
    cmd.kinematic_modes([igc_sdk.KinematicMode.PJS] * N_KINEMATIC_MODES)


# ----------------------------------------------------------------------------- ONNX
class OnnxPolicyWrapper:
    def __init__(self, session, input_name):
        self.session = session
        self.input_name = input_name

    def __call__(self, obs_np: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input_name: obs_np.astype(np.float32)})
        return np.asarray(out[0], dtype=np.float32)


def load_onnx_policy(policy_path: str) -> OnnxPolicyWrapper:
    if ort is None:
        raise ImportError("onnxruntime 필요")
    sess = ort.InferenceSession(policy_path, providers=["CPUExecutionProvider"])
    print(f"[real] ONNX loaded: {policy_path}  providers={sess.get_providers()}")
    return OnnxPolicyWrapper(sess, sess.get_inputs()[0].name)


# ----------------------------------------------------------------------------- limits
def load_joint_limits(params_yaml: str):
    """params.yaml(ROS2 param) 에서 31-엔트리 position_min/max, velocity_max 를 회수."""
    import yaml
    with open(params_yaml) as f:
        doc = yaml.safe_load(f)

    found = {}
    def walk(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in ("position_max", "position_min", "velocity_max") and isinstance(v, list):
                    found[k] = v
                walk(v)
        elif isinstance(d, list):
            for v in d:
                walk(v)
    walk(doc)
    for k in ("position_max", "position_min", "velocity_max"):
        if k not in found or len(found[k]) != NUM_MOTORS:
            raise ValueError(f"params.yaml 에서 31-엔트리 {k} 회수 실패 (got {found.get(k)})")
    return (np.array(found["position_min"], np.float32),
            np.array(found["position_max"], np.float32),
            np.array(found["velocity_max"], np.float32))


# ----------------------------------------------------------------------------- LowState 콜백
class StateBuffer:
    """rt/lowstate 콜백이 채우는 최신 상태(스레드 세이프) + freshness 타임스탬프."""
    def __init__(self, imu_quat_order="wxyz"):
        self.lock = threading.Lock()
        self.q = np.zeros(NUM_MOTORS, np.float32)
        self.dq = np.zeros(NUM_MOTORS, np.float32)
        self.rpy = np.zeros(3, np.float32)
        self.ang_vel = np.zeros(3, np.float32)
        self.stamp = 0.0
        self.count = 0
        self.imu_quat_order = imu_quat_order
        self._diag_done = False

    def callback(self, msg):
        try:
            imu = msg.imu_state()
            # [C1 RESOLVED] IMUState ships rpy() directly; cpp low-level example uses it as the
            # canonical roll/pitch -> use rpy() as primary, no quaternion-order ambiguity.
            rpy_sdk = np.asarray(imu.rpy(), np.float32)
            rpy = rpy_sdk.copy()
            gyro = np.asarray(imu.gyroscope(), np.float32)
            # [MS->PJS] policy trained in PJS joint space. LowState carries motor_state[31](MS=
            # ankle Out/In) and joint_state[31](PJS=Pitch/Roll); read joint_state (cpp parity).
            js = msg.joint_state()
            q = np.array([js[i].q() for i in range(NUM_MOTORS)], np.float32)
            dq = np.array([js[i].dq() for i in range(NUM_MOTORS)], np.float32)
            with self.lock:
                self.q, self.dq, self.rpy, self.ang_vel = q, dq, rpy, gyro
                self.stamp = time.time()
                self.count += 1
            if not self._diag_done:
                self._diag_done = True
                quat = np.asarray(imu.quaternion(), np.float32)
                if self.imu_quat_order == "xyzw":
                    quat = quat[[3, 0, 1, 2]]
                print(f"[real][C1] IMU 부팅대조: imu.rpy()(사용중)={rpy_sdk[:2]}  "
                      f"quatToEuler(quat)={quatToEuler(quat)[:2]}  (둘이 비슷해야 정상)")
        except Exception as e:  # noqa: BLE001
            print(f"[real] lowstate cb 예외: {e}")

    def snapshot(self):
        with self.lock:
            return self.q.copy(), self.dq.copy(), self.rpy.copy(), self.ang_vel.copy(), self.stamp


# ----------------------------------------------------------------------------- 컨트롤러
class RealController:
    def __init__(self, args):
        self.args = args
        self.device_dt = 1.0 / args.policy_frequency      # 50Hz -> 0.02
        self.policy = load_onnx_policy(args.policy)

        # --- IGRIS 29-DoF config (server_low_level_igris_sim.py 와 1:1 동일) ---
        self.default_dof_pos = np.array([
            0.0, 0.0, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            -0.2, 0.0, 0.0, 0.4, -0.2, 0.0,
            0.0, 0.4, 0.0, -0.4, 0.0, 0.0, 0.0,
            0.0, -0.4, 0.0, -0.4, 0.0, 0.0, 0.0,
        ], np.float32)
        self.stand_dof_pos = self.default_dof_pos.copy()
        self.stand_root_z = 0.97
        self.stand_root_rp = (0.0, 0.0)
        # PD stiffness — twist2_arc 학습값(= 우리 정책이 학습된 게인). leg/arm은 SDK 예제
        # cyclonedds_low_level_control.cpp kp와 **글자 그대로 동일**(500,200,50,500,300,300 /
        # 50,50,30,30,5,5,5) → 단위(N·m/rad) 동일 확정. **waist만 차이**: 우리=100/100/100(학습값),
        # SDK 데모=50/25/25. 정책이 100으로 학습됐으므로 100 유지(예제값으로 바꾸면 학습과 불일치).
        self.stiffness = np.array([
            100, 100, 100,
            500, 200, 50, 500, 300, 300,
            500, 200, 50, 500, 300, 300,
            50, 50, 30, 30, 5, 5, 5,
            50, 50, 30, 30, 5, 5, 5,
        ], np.float32)
        self.damping = np.array([
            0.8, 0.8, 0.8,
            3.0, 0.5, 0.5, 3.0, 1.5, 1.5,
            3.0, 0.5, 0.5, 3.0, 1.5, 1.5,
            0.5, 0.5, 0.15, 0.15, 0.1, 0.1, 0.1,
            0.5, 0.5, 0.15, 0.15, 0.1, 0.1, 0.1,
        ], np.float32)
        # action_scale: MUST match the deployed policy's training config (sim parity).
        # wbc02/순정 1단계=0.25, g1damp류=0.5. Opt-in env override (sim 서버와 동일 규약):
        #   IGRIS_ACTION_SCALE=0.5 python server_low_level_igris_real.py ...
        _ascale = float(os.environ.get("IGRIS_ACTION_SCALE", "0.25"))
        self.action_scale = np.full(NUM_ACTIONS, _ascale, np.float32)
        print(f"[real] action_scale={_ascale} (정책 학습값과 반드시 일치)")
        self.ankle_idx = [7, 8, 13, 14]
        self.neck_kp = np.array(NECK_KP, np.float32)   # motors 29,30 hold (head upright)
        self.neck_kd = np.array(NECK_KD, np.float32)

        gm = float(os.environ.get("IGRIS_GAIN_MULT", "1.0"))
        if gm != 1.0:
            print(f"[real] IGRIS_GAIN_MULT={gm}")
            self.stiffness *= gm
            self.damping *= gm ** 0.5

        # obs 차원 (sim 과 동일: 1432)
        self.n_mimic_obs = 35
        self.n_proprio = 3 + 2 + 3 * NUM_ACTIONS
        self.n_obs_single = self.n_mimic_obs + self.n_proprio        # 127
        self.history_len = 10
        self.total_obs_size = self.n_obs_single * (self.history_len + 1) + self.n_mimic_obs  # 1432
        self.proprio_history_buf = deque(maxlen=self.history_len)
        self.history_primed = False
        self.last_action = np.zeros(NUM_ACTIONS, np.float32)

        # --- 안전 한계 (params.yaml 31-엔트리) ---
        self.pos_min, self.pos_max, self.vel_max = load_joint_limits(args.params_yaml)
        print(f"[real] joint limits loaded ({args.params_yaml}): 31 entries, vel_max[:3]={self.vel_max[:3]}")
        self.prev_cmd_q = None   # 레이트리밋용 직전 명령(31)

        # --- Redis (레퍼런스 mimic) ---
        self.r = redis.Redis(host=args.redis_ip, port=6379, db=0)
        self.r.set("motion_exit_signal", "0")

        # --- DDS / SDK ---
        ch = igc_sdk.ChannelFactory.Instance()
        ch.Init(args.domain_id, args.namespace)
        if not ch.IsInitialized():
            raise RuntimeError("ChannelFactory init 실패 (domain/namespace/NIC 확인)")
        self.channel = ch
        self.client = igc_sdk.IgrisC_Client()
        self.client.Init()
        self.client.SetTimeout(5.0)

        self.state = StateBuffer(imu_quat_order=args.imu_quat_order)
        self.sub = igc_sdk.LowStateSubscriber(LOWSTATE_TOPIC, igc_sdk.QosProfile.SensorData())
        if not self.sub.init(self.state.callback):
            raise RuntimeError("LowStateSubscriber init 실패")
        self.pub = igc_sdk.LowCmdPublisher(LOWCMD_TOPIC, igc_sdk.QosProfile.SensorData())
        if not self.pub.init():
            raise RuntimeError("LowCmdPublisher init 실패")
        time.sleep(0.3)

    # ---------------- LowCmd 송신 헬퍼 ----------------
    def _write_lowcmd(self, q31, kp31, kd31):
        cmd = igc_sdk.LowCmd()
        _set_kinematic_modes_pjs(cmd)          # [C2] PJS joint-space targets
        motors = cmd.motors()
        for i in range(NUM_MOTORS):
            m = motors[i]
            m.id(i); m.q(float(q31[i])); m.dq(0.0); m.tau(0.0)
            m.kp(float(kp31[i])); m.kd(float(kd31[i]))
        self.pub.write(cmd)

    def _full_gains(self):
        """31-vector PD gains: idx 0..28 = policy stiffness/damping, 29/30 = neck hold."""
        kp31 = np.zeros(NUM_MOTORS, np.float32)
        kd31 = np.zeros(NUM_MOTORS, np.float32)
        kp31[:NUM_ACTIONS] = self.stiffness
        kd31[:NUM_ACTIONS] = self.damping
        kp31[NECK_IDX] = self.neck_kp
        kd31[NECK_IDX] = self.neck_kd
        return kp31, kd31

    def publish_damping(self):
        """모든 모터 kp=0, kd=소량, tau=0 → 안전 댐핑(추락 완충)."""
        self._write_lowcmd(np.zeros(NUM_MOTORS), np.zeros(NUM_MOTORS),
                           np.full(NUM_MOTORS, 2.0))

    def _clamp_and_rate_limit(self, q31):
        q = np.clip(q31, self.pos_min, self.pos_max)
        if self.prev_cmd_q is not None:
            dmax = self.vel_max * self.device_dt * self.args.vel_limit_frac
            q = np.clip(q, self.prev_cmd_q - dmax, self.prev_cmd_q + dmax)
        self.prev_cmd_q = q.copy()
        return q

    # ---------------- 게이트된 브링업 ----------------
    def _confirm(self, prompt):
        if self.args.assume_yes:
            print(f"[gate] (assume_yes) {prompt} -> YES")
            return True
        return input(f"[gate] {prompt}  진행? [y/N] ").strip().lower() == "y"

    def _svc(self, label, fn):
        res = fn()
        ok = res.success()
        print(f"[gate] {label}: {'OK' if ok else 'FAILED ' + res.message()}")
        if not ok:
            raise RuntimeError(f"{label} 실패")

    def staged_bringup(self):
        print("\n==== GATED BRING-UP (사람 입회 · e-stop 손에) ====")
        if not self._confirm("토크 OFF 상태에서 rt/lowstate 수신 확인됨?"):
            raise SystemExit("브링업 중단")
        # lowstate 살아있는지
        _, _, _, _, st = self.state.snapshot()
        if self.state.count == 0:
            raise RuntimeError("rt/lowstate 미수신 — 연결/namespace 확인")

        if self._confirm("InitBms(BMS_AND_MOTOR_INIT)?"):
            self._svc("InitBms", lambda: self.client.InitBms(
                igc_sdk.BmsInitType.BMS_AND_MOTOR_INIT, 5000))
        if self._confirm("SetTorque(TORQUE_ON)?  (이후 로봇에 힘이 들어감)"):
            self._svc("SetTorque ON", lambda: self.client.SetTorque(
                igc_sdk.TorqueType.TORQUE_ON, 5000))
        if self._confirm("ControlMode = LOW_LEVEL_JOINT_CONTROL?"):
            self._svc("LOW_LEVEL_JOINT_CONTROL", lambda: self.client.SendControlModeCommand(
                igc_sdk.ControlModeCommandType.CONTROL_MODE_CMD_LOW_LEVEL_JOINT_CONTROL, "", False, 5000))

        # 현재자세 → default 로 ~2s 램프 (q+kp/kd, 펌웨어 PD)
        if self._confirm("현재자세→default(HOME류)로 2초 램프?"):
            self.ramp_to_default(duration=self.args.ramp_seconds)
        print("==== 브링업 완료. 정책 루프 시작 직전 ====")
        if not self._confirm("정책(텔레옵) 루프 시작?"):
            raise SystemExit("정책 시작 보류")

    def ramp_to_default(self, duration=2.0):
        q0, _, _, _, _ = self.state.snapshot()
        steps = max(1, int(duration / self.device_dt))
        kp31, kd31 = self._full_gains()
        target = q0.copy()
        target[:NUM_ACTIONS] = self.default_dof_pos
        target[NECK_IDX] = 0.0
        for s in range(steps + 1):
            a = s / steps
            q = (1 - a) * q0 + a * target
            q = self._clamp_and_rate_limit(q)
            self._write_lowcmd(q, kp31, kd31)
            time.sleep(self.device_dt)

    # ---------------- obs (sim 과 동일 구성) ----------------
    def build_obs(self, q, dq, rpy, ang_vel):
        dof_pos = q[:NUM_ACTIONS]
        dof_vel = dq[:NUM_ACTIONS].copy()
        obs_dof_vel = dof_vel.copy()
        obs_dof_vel[self.ankle_idx] = 0.0
        obs_proprio = np.concatenate([
            ang_vel * 0.25,
            rpy[:2],
            (dof_pos - self.default_dof_pos),
            obs_dof_vel * 0.05,
            self.last_action,
        ])
        # 레퍼런스 mimic (Redis) — 없으면 sim 과 동일 standing fallback
        mb = self.r.get("action_body_unitree_g1_with_hands")
        ref_age_ok = True
        if mb is None:
            fb_root = np.array([0., 0., self.stand_root_z,
                                self.stand_root_rp[0], self.stand_root_rp[1], 0.])
            action_mimic = np.concatenate([fb_root, self.stand_dof_pos]).tolist()
            ref_age_ok = False
        else:
            action_mimic = json.loads(mb)
        obs_full = np.concatenate([action_mimic, obs_proprio])     # 127
        if not self.history_primed:
            for _ in range(self.history_len):
                self.proprio_history_buf.append(obs_full.copy())
            self.history_primed = True
        obs_hist = np.array(self.proprio_history_buf).flatten()
        self.proprio_history_buf.append(obs_full)
        future_obs = np.array(action_mimic).copy()
        obs_buf = np.concatenate([obs_full, obs_hist, future_obs]) # 1432
        assert obs_buf.shape[0] == self.total_obs_size, \
            f"obs {obs_buf.shape[0]} != {self.total_obs_size}"
        return obs_buf.astype(np.float32), ref_age_ok

    # ---------------- 50Hz 정책 루프 ----------------
    def run_policy_loop(self):
        kp31, kd31 = self._full_gains()
        pd_target = self.default_dof_pos.copy()
        print("[real] 정책 루프 시작 (Ctrl-C / 'e-stop' → 댐핑 종료)")
        # dry_run/loopback에선 staged_bringup을 건너뛰므로 첫 lowstate 수신을 기다린다
        # (빈 StateBuffer면 워치독이 즉시 트립). 실기 경로는 bringup이 이미 보장.
        t_wait = time.time()
        while self.state.count == 0 and (time.time() - t_wait) < 5.0:
            time.sleep(0.02)
        if self.state.count == 0:
            print("[real] WARN: 5s 동안 lowstate 미수신 — 그래도 진입(워치독이 처리)")
        else:
            print(f"[real] lowstate OK (count={self.state.count}) — 루프 진입")
        next_t = time.time()
        step = 0
        while True:
            t0 = time.time()
            q, dq, rpy, ang_vel, st = self.state.snapshot()

            # --- 워치독: lowstate freshness ---
            if st == 0 or (t0 - st) > self.args.lowstate_timeout:
                print(f"[real][WATCHDOG] lowstate stale ({t0 - st:.3f}s) → 댐핑")
                self.publish_damping()
                raise RuntimeError("lowstate watchdog")

            obs, ref_ok = self.build_obs(q, dq, rpy, ang_vel)
            raw = self.policy(obs[None, :]).squeeze()
            self.last_action = raw
            pd_target = np.clip(raw, -10., 10.) * self.action_scale + self.default_dof_pos

            # 진단 텔레메트리(루프백/dry_run 검증용): 2s마다 mimic 추종 상태 출력
            if step % 100 == 0:
                mimic = obs[:self.n_mimic_obs]
                dev = float(np.mean(np.abs(q[:NUM_ACTIONS] - self.default_dof_pos)))
                print("[real][trk] step=%5d ref_ok=%s mimic[z/r/p]=%.3f/%+.3f/%+.3f "
                      "|q-def|mean=%.4f act|max|=%.3f"
                      % (step, ref_ok, mimic[2], mimic[3], mimic[4], dev,
                         float(np.max(np.abs(raw)))))

            # 31 벡터로 확장(목 고정) + 클램프 + 레이트리밋
            q_cmd = q.copy()
            q_cmd[:NUM_ACTIONS] = pd_target
            q_cmd[NECK_IDX] = 0.0
            q_cmd = self._clamp_and_rate_limit(q_cmd)
            self._write_lowcmd(q_cmd, kp31, kd31)

            # 상태 되먹임(Redis) — 로깅/하이레벨용(선택)
            state_body = np.concatenate([ang_vel, rpy[:2], q[:NUM_ACTIONS]])
            self.r.set("state_body_unitree_g1_with_hands", json.dumps(state_body.tolist()))

            next_t += self.device_dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()   # 오버런 리셋
                print(f"[real] loop overrun {-sleep*1000:.1f}ms")
            step += 1

    # ---------------- 안전 종료 ----------------
    def safe_shutdown(self):
        print("[real] 안전 종료: 댐핑 → MOTION_STOP → TORQUE_OFF")
        try:
            for _ in range(10):
                self.publish_damping(); time.sleep(0.02)
        except Exception as e:  # noqa: BLE001
            print(f"  damping 송신 예외: {e}")
        try:
            self.client.SendControlModeCommand(
                igc_sdk.ControlModeCommandType.CONTROL_MODE_CMD_MOTION_STOP, "", False, 5000)
        except Exception as e:  # noqa: BLE001
            print(f"  MOTION_STOP 예외: {e}")
        try:
            self.client.SetTorque(igc_sdk.TorqueType.TORQUE_OFF, 5000)
        except Exception as e:  # noqa: BLE001
            print(f"  TORQUE_OFF 예외: {e}")
        try:
            self.sub.stop(); self.pub.stop(); self.channel.Release()
        except Exception:  # noqa: BLE001
            pass


def main():
    p = argparse.ArgumentParser(description="IGRIS-C 실로봇 저수준 정책 서버 (DRAFT, GATED)")
    p.add_argument("--policy", required=True, help="29-DoF ONNX (obs1432/act29)")
    p.add_argument("--params_yaml",
                   default=os.path.join(os.path.dirname(__file__), "real_params.yaml"),
                   help="joint-limit yaml (default: bundled real_params.yaml)")
    p.add_argument("--redis_ip", default="localhost")
    p.add_argument("--domain_id", type=int, default=0)
    p.add_argument("--namespace", default="igris_c_ROBOT_ID",
                   help="[C3] live DDS namespace; set to your robot's igris_c_<robot_id>")
    p.add_argument("--policy_frequency", type=int, default=50)
    p.add_argument("--imu_quat_order", choices=["wxyz", "xyzw"], default="wxyz", help="[C1]")
    p.add_argument("--lowstate_timeout", type=float, default=0.1, help="워치독: lowstate 최대 age(s)")
    p.add_argument("--vel_limit_frac", type=float, default=1.0, help="velocity_max 대비 레이트리밋 비율")
    p.add_argument("--ramp_seconds", type=float, default=2.0)
    p.add_argument("--assume_yes", action="store_true", help="⚠️ 게이트 프롬프트 생략(자동화 전용, 실기 비권장)")
    p.add_argument("--dry_run_no_torque", action="store_true",
                   help="브링업/토크ON 생략하고 루프만(토크OFF lowcmd 송신 점검용)")
    args = p.parse_args()

    if not os.path.exists(args.policy):
        sys.exit(f"정책 파일 없음: {args.policy}")

    print("=" * 70)
    print(" IGRIS-C REAL low-level server (DRAFT)  —  ⚠️ 하드웨어 게이트")
    print(" sim2sim 패리티 게이트 통과 전 실기 금지. e-stop 손에. 사람 입회.")
    print(f"  policy={args.policy}")
    print(f"  DDS domain={args.domain_id} ns={args.namespace}  (NIC/CYCLONEDDS_URI=env)")
    print("=" * 70)

    ctrl = RealController(args)
    try:
        if not args.dry_run_no_torque:
            ctrl.staged_bringup()
        else:
            print("[real] dry_run_no_torque: 브링업/토크ON 생략")
        ctrl.run_policy_loop()
    except KeyboardInterrupt:
        print("\n[real] KeyboardInterrupt")
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        print(f"[real] 예외: {e}")
    finally:
        ctrl.safe_shutdown()


if __name__ == "__main__":
    main()
