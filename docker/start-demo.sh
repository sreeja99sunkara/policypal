#!/usr/bin/env bash
# Single-container demo: the FastAPI app on an internal port (127.0.0.1 only)
# and Streamlit on $PORT, the one port the host exposes. If either process
# exits, stop the other and exit too, so the platform restarts the container.
set -euo pipefail

API_PORT="${API_PORT:-8000}"
PORT="${PORT:-7860}"
export POLICYPAL_API_URL="http://127.0.0.1:${API_PORT}"

uvicorn app.main:app --host 127.0.0.1 --port "${API_PORT}" &
api_pid=$!

streamlit run streamlit_app.py \
    --server.address=0.0.0.0 --server.port="${PORT}" &
ui_pid=$!

shutdown() {
    kill -TERM "${api_pid}" "${ui_pid}" 2>/dev/null || true
    wait || true
}
trap 'shutdown; exit 143' TERM INT

set +e
wait -n "${api_pid}" "${ui_pid}"
status=$?
set -e
echo "start-demo: a process exited (status ${status}); stopping the container" >&2
shutdown
exit "${status}"
