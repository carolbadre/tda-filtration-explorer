#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# TDA Filtration Explorer — build & run script
#
# Usage:
#     bash run.sh
#
# What it does:
#   1. Verifies Python 3.10+ is available.
#   2. (Optional) creates a local venv if one doesn't exist; comment out the
#      venv block if you want to install into your current pyenv environment.
#   3. Installs Flask, NumPy, and GUDHI from requirements.txt.
#   4. Starts the Flask dev server on http://127.0.0.1:5000
#
# Notes on GUDHI:
#   - On Ubuntu/WSL `pip install gudhi` ships pre-built wheels for recent
#     Pythons; CGAL deps come bundled.
#   - If the install fails with a CGAL error, you may need:
#         sudo apt install libgmp-dev libmpfr-dev libcgal-dev
#     and then retry.
# ----------------------------------------------------------------------------

set -euo pipefail

cd "$(dirname "$0")"

# --- Python version check ---------------------------------------------------
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH" >&2
    exit 1
fi

PYV=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "Using Python $PYV"

# --- Optional venv ----------------------------------------------------------
# Comment this block out if you'd rather install into your active environment.
if [ ! -d ".venv" ]; then
    echo "Creating local venv at ./.venv ..."
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# --- Install dependencies ---------------------------------------------------
echo "Installing Python dependencies ..."
pip install --upgrade pip >/dev/null
pip install -r requirements.txt

# --- Run --------------------------------------------------------------------
echo ""
echo "----------------------------------------------------------------"
echo "  TDA Filtration Explorer running at http://127.0.0.1:5000"
echo "  Open that URL in your browser."
echo "  Ctrl-C here to stop."
echo "----------------------------------------------------------------"
echo ""
python3 server.py
