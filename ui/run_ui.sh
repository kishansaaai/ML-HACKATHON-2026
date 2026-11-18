#!/usr/bin/env bash
# PathSnap Evaluator UI Launcher (Laptop 2 - macOS/Linux)
# Usage:
#   ./run_ui.sh
#   ./run_ui.sh http://192.168.1.25:8000

set -euo pipefail
cd "$(dirname "$0")"

if [[ $# -ge 1 ]]; then
  export PATHSNAP_API_URL="$1"
fi

export PATHSNAP_API_URL="${PATHSNAP_API_URL:-http://127.0.0.1:8000}"

echo "========================================================"
echo " PathSnap Evaluator UI (Laptop 2)"
echo " Target API: ${PATHSNAP_API_URL}"
echo "========================================================"
python3 -m streamlit run app.py --server.port 8501
