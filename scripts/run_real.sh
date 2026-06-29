#!/bin/bash
# run_real.sh -- launch the IGRIS-C low-level policy server on the REAL robot.
#  *** HARDWARE GATE: read docs/README.md section 5 first. Robot on hoist, e-stop
#      in hand, sim_test gate passed. The server runs a staged, human-confirmed
#      bring-up (BMS -> Torque ON -> LOW_LEVEL -> ramp) before the policy loop. ***
# Reference comes from Redis (action_body_unitree_g1_with_hands), published by either
# the live teleop (teleop/xrobot_teleop_to_igris.py) or the offline feeder.
set -e
HERE=$(dirname "$(realpath "$0")")
ROOT=$(dirname "$HERE")
source "$HOME/igris_deploy_venv/bin/activate"

POLICY=${IGRIS_POLICY:-$ROOT/policy/A3_arc_student_wh2_10k.onnx}
PARAMS=${IGRIS_PARAMS:-$ROOT/server/real_params.yaml}
NS=${IGRIS_NS:-igris_c_ROBOT_ID}   # set to your robot's namespace: igris_c_<robot_id>
DOMAIN=${IGRIS_DOMAIN:-0}

echo "policy=$POLICY"
echo "params=$PARAMS (limits)"
echo "DDS domain=$DOMAIN namespace=$NS"
echo "redis: $(redis-cli ping 2>/dev/null || echo DOWN)"

exec python3 "$ROOT/server/server_low_level_igris_real.py" \
    --policy "$POLICY" \
    --params_yaml "$PARAMS" \
    --domain_id "$DOMAIN" \
    --namespace "$NS" \
    "$@"
# add --dry_run_no_torque to test lowcmd publishing WITHOUT enabling motors.
