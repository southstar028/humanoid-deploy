#!/bin/bash
# run_sim_test.sh -- robot-free pre-flight: sim2sim-over-DDS loopback.
# Runs EVERYTHING in an ubuntu:24.04 container (NUC-parity glibc/py3.12):
#   feeder (standalone, optional) + dds_robot_stub (fake robot in MuJoCo) +
#   real low-level server, over the SDK rt/lowcmd / rt/lowstate DDS. No hardware.
# Needs: docker + host redis.
#   scripts/run_sim_test.sh                 # standing (no reference)
#   scripts/run_sim_test.sh <motion.pkl>    # track an offline motion (e.g. PICO)
set -e
HERE=$(dirname "$(realpath "$0")")
ROOT=$(dirname "$HERE")
IMG=igris_real_test:latest
MOTION="$1"

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "==> building $IMG ..."
  docker build -f "$ROOT/sim_test/Dockerfile.realtest" -t "$IMG" "$ROOT/sim_test"
fi

redis-cli ping >/dev/null 2>&1 || { echo "redis not up (sudo service redis-server start)"; exit 1; }
redis-cli del action_body_unitree_g1_with_hands >/dev/null 2>&1

POLICY=$ROOT/policy/A3_arc_student_wh2_10k.onnx
MNT_MOTION=""
FEEDER_CMD="echo '[c] no motion -> standing fallback'"
if [[ -n "$MOTION" ]]; then
  MNT_MOTION="-v $(realpath "$MOTION"):/motion.pkl:ro"
  FEEDER_CMD="python3 -u /feeder/feed_igris_motion_standalone.py --motion_file /motion.pkl --loop > /logs/feeder.log 2>&1 & sleep 2; echo '[c] feeder up'"
fi

mkdir -p "$ROOT/sim_test/logs"
docker run --rm --network host \
  -e IGRIS_TWIST2_DEPLOY=/nonexistent \
  -v "$ROOT/sim_test":/work:ro \
  -v "$ROOT/sim_test/assets":/assets:ro \
  -v "$ROOT/feeder":/feeder:ro \
  -v "$ROOT/server":/server:ro \
  -v "$ROOT/sim_test/logs":/logs:rw \
  -v "$POLICY":/policy.onnx:ro \
  $MNT_MOTION \
  "$IMG" bash -lc "
    set -e
    $FEEDER_CMD
    echo '[c] stub (fake robot) up...'
    python3 -u /work/dds_robot_stub.py --xml /assets/igris_c/igris_c.xml --domain_id 0 --namespace '' --pub_hz 300 --duration 22 > /logs/stub.log 2>&1 &
    STUB=\$!
    sleep 3
    echo '[c] real server (dry_run, no torque) 18s...'
    timeout 18 python3 -u /server/server_low_level_igris_real.py --policy /policy.onnx --params_yaml /server/real_params.yaml --domain_id 0 --namespace '' --dry_run_no_torque > /logs/server.log 2>&1 || true
    kill \$STUB 2>/dev/null || true; sleep 1
    echo '===== STUB TAIL ====='; tail -24 /logs/stub.log
    echo '===== SERVER TAIL ====='; tail -8 /logs/server.log
  "
echo "logs in $ROOT/sim_test/logs/"
