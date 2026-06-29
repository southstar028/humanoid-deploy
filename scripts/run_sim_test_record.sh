#!/bin/bash
# run_sim_test_record.sh -- like run_sim_test.sh but RECORDS an mp4 of the fake robot.
# Container (EGL offscreen) dumps PNG frames; host ffmpeg assembles the mp4.
#   scripts/run_sim_test_record.sh [<motion.pkl>] [<out.mp4>]
set -e
HERE=$(dirname "$(realpath "$0")")
ROOT=$(dirname "$HERE")
IMG=igris_real_test:latest
MOTION="$1"
OUT="${2:-$ROOT/sim_test/loopback.mp4}"
FPS=30

docker image inspect "$IMG" >/dev/null 2>&1 || \
  docker build -f "$ROOT/sim_test/Dockerfile.realtest" -t "$IMG" "$ROOT/sim_test"
redis-cli ping >/dev/null 2>&1 || { echo "redis not up (sudo service redis-server start)"; exit 1; }
redis-cli del action_body_unitree_g1_with_hands >/dev/null 2>&1

POLICY=$ROOT/policy/A3_arc_student_wh2_10k.onnx
FRAMES="$ROOT/sim_test/logs/frames"
rm -rf "$FRAMES"; mkdir -p "$FRAMES" "$ROOT/sim_test/logs"

MNT_MOTION=""
FEEDER_CMD="echo '[c] no motion -> standing fallback'"
if [[ -n "$MOTION" ]]; then
  MNT_MOTION="-v $(realpath "$MOTION"):/motion.pkl:ro"
  FEEDER_CMD="python3 -u /feeder/feed_igris_motion_standalone.py --motion_file /motion.pkl --loop > /logs/feeder.log 2>&1 & sleep 2; echo '[c] feeder up'"
fi

docker run --rm --network host \
  -e IGRIS_TWIST2_DEPLOY=/nonexistent -e MUJOCO_GL=egl \
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
    echo '[c] stub (fake robot, RECORDING) up...'
    python3 -u /work/dds_robot_stub.py --xml /assets/igris_c/igris_c.xml --domain_id 0 --namespace '' \
        --pub_hz 300 --duration 20 --record_frames_dir /logs/frames --render_fps $FPS > /logs/stub.log 2>&1 &
    STUB=\$!
    sleep 3
    echo '[c] real server (dry_run) 18s...'
    timeout 18 python3 -u /server/server_low_level_igris_real.py --policy /policy.onnx \
        --params_yaml /server/real_params.yaml --domain_id 0 --namespace '' --dry_run_no_torque > /logs/server.log 2>&1 || true
    echo '[c] waiting for stub to finish loop + offline render...'
    wait \$STUB 2>/dev/null || true
    echo '===== STUB TAIL ====='; tail -6 /logs/stub.log
  "

N=$(ls "$FRAMES"/frame_*.png 2>/dev/null | wc -l)
echo "[host] $N frames -> assembling mp4 with ffmpeg..."
[ "$N" -gt 0 ] || { echo "no frames rendered"; exit 1; }
ffmpeg -nostdin -y -framerate $FPS -i "$FRAMES/frame_%05d.png" -c:v libx264 -pix_fmt yuv420p "$OUT" >/dev/null 2>&1
echo "[host] wrote $OUT ($(du -h "$OUT" | cut -f1), $N frames @ ${FPS}fps)"
