#!/bin/bash
# setup_nuc.sh -- one-time environment setup for IGRIS-C low-level deploy.
# Target: robot NUC (Ubuntu 24.04 / Python 3.12) OR any Ubuntu 24.04 host.
# Installs ONLY what the low-level SERVER + offline FEEDER need (no isaacgym,
# no torch, no ROS2 build). Live PICO teleop (GMR) has its own env -- see README.
set -e
HERE=$(dirname "$(realpath "$0")")
ROOT=$(dirname "$HERE")

echo "==> python check (need 3.12 for the SDK wheel)"
python3 --version

echo "==> create venv  ~/igris_deploy_venv"
python3 -m venv "$HOME/igris_deploy_venv"
source "$HOME/igris_deploy_venv/bin/activate"
pip install -U pip -q

echo "==> install SDK wheel + runtime deps"
pip install -q "$ROOT"/sdk/igris_c_sdk-2.0.2*.whl
pip install -q numpy onnxruntime redis pyyaml mujoco

echo "==> verify SDK imports"
python3 -c "import igris_c_sdk; print('  igris_c_sdk OK'); import onnxruntime,redis,numpy,yaml,mujoco; print('  deps OK')"

echo "==> redis-server present?"
command -v redis-server >/dev/null && echo "  redis-server OK" || echo "  WARNING: need: sudo apt install redis-server"

echo ""
echo "DONE. Activate with:  source ~/igris_deploy_venv/bin/activate"
echo "  robot-free pre-flight (dev PC + docker): scripts/run_sim_test.sh"
echo "  real robot:                              scripts/run_real.sh  (read README first)"
