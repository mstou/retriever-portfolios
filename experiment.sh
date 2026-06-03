#!/usr/bin/env bash
set -euo pipefail

### ============== CONFIG ==================
# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"
ENV_ACTIVATE="${ENV_ACTIVATE:-${PROJECT_DIR}/.venv/bin/activate}"
LAUNCH_70="${PROJECT_DIR}/setup/launch_llama70b.sh"
LAUNCH_GEMMA="${PROJECT_DIR}/setup/launch_gemma27b.sh"

# vLLM server host/ports (must match your launch_* scripts)
HOST="127.0.0.1"
PORT_70="${PORT_70:-8000}"
PORT_GEMMA="${PORT_GEMMA:-5000}"

# Which servers to start & wait for
START_LLAMA70B="${START_LLAMA70B:-false}"
START_GEMMA27B="${START_GEMMA27B:-false}"

# Logs
SRV_LOG_DIR="${SRV_LOG_DIR:-${PROJECT_DIR}/server_logs}"
EXP_LOG_DIR="${EXP_LOG_DIR:-${PROJECT_DIR}/experiment_logs}"

# Experiment
EXP_LOG_SUFFIX="${EXP_LOG_SUFFIX:-run}"
EXP_LOG="${EXP_LOG_DIR}/experiment_${EXP_LOG_SUFFIX}.out"
EXP_CMD="${EXP_CMD:-python3 -u main.py --help}"

# Readiness wait config
READINESS_TIMEOUT_SEC="${READINESS_TIMEOUT_SEC:-2400}"
READINESS_POLL_SEC="${READINESS_POLL_SEC:-3}"
### ===========================================

mkdir -p "$SRV_LOG_DIR" "$EXP_LOG_DIR"
source "$ENV_ACTIVATE"
cd "$PROJECT_DIR"

# ---------------- helpers ----------------
pids_to_kill=()

cleanup() {
  if ((${#pids_to_kill[@]})); then
    echo "[INFO] Cleaning up server processes… (${pids_to_kill[*]})"
    kill "${pids_to_kill[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

check_ready() {
  local port="$1"
  if curl -sf -m 2 "http://${HOST}:${port}/health" >/dev/null 2>&1; then return 0; fi
  if curl -sf -m 2 "http://${HOST}:${port}/v1/models" >/dev/null 2>&1; then return 0; fi
  if command -v nc >/dev/null 2>&1 && nc -z -w2 "$HOST" "$port"; then return 0; fi
  return 1
}

wait_for_ready() {
  local name="$1" port="$2" start_ts now elapsed
  start_ts=$(date +%s)
  echo -n "[INFO] Waiting for ${name} on ${HOST}:${port} "
  while ! check_ready "$port"; do
    now=$(date +%s); elapsed=$(( now - start_ts ))
    if (( elapsed > READINESS_TIMEOUT_SEC )); then
      echo; echo "[ERROR] Timeout for ${name} on ${port}. See logs: ${SRV_LOG_DIR}/${name}.out"
      exit 1
    fi
    echo -n "."; sleep "$READINESS_POLL_SEC"
  done
  echo "✓ ready"
}

# launch one server if flag=true; do NOT wait here
launch_if_enabled() {
  local name="$1" launch_script="$2" port="$3" start_flag="$4"
  if [[ "$start_flag" == "true" ]]; then
    echo "[INFO] Launching ${name}…"
    nohup bash -lc "$launch_script" >"${SRV_LOG_DIR}/${name}.out" 2>&1 & pid=$!
    pids_to_kill+=("$pid")                # for cleanup
    started_names+=("$name")              # remember what we launched
    started_ports+=("$port")
    echo "[INFO] ${name} PID: $pid"
  else
    echo "[INFO] ${name}: START flag is false — skipping."
  fi
}

# --------------- launch block ---------------
started_names=()
started_ports=()

launch_if_enabled "gemma27b" "$LAUNCH_GEMMA" "$PORT_GEMMA" "$START_GEMMA27B"
launch_if_enabled "llama70b" "$LAUNCH_70"   "$PORT_70"   "$START_LLAMA70B"

# --------------- parallel readiness waits ---------------
if ((${#started_names[@]})); then
  echo "[INFO] Waiting for all requested servers in parallel…"
  wait_pids=()
  for i in "${!started_names[@]}"; do
    name="${started_names[$i]}"; port="${started_ports[$i]}"
    # run readiness check in background
    wait_for_ready "$name" "$port" &
    wait_pids+=("$!")
  done

  # join: fail fast if any readiness fails
  for w in "${wait_pids[@]}"; do
    if ! wait "$w"; then
      echo "[ERROR] A server failed to become ready. See logs in ${SRV_LOG_DIR}/"
      exit 1
    fi
  done
  echo "[INFO] All requested servers are ready."
else
  echo "[INFO] No servers were started (all START_* flags false)."
fi


# --------------- experiment -----------------
echo "[INFO] Starting experiment..."
echo "[INFO]  -> logging to: ${EXP_LOG}"
set -o pipefail
${EXP_CMD} 2>&1 | tee "${EXP_LOG}"
status=${PIPESTATUS[0]}

echo "[INFO] Experiment finished with status ${status}."
exit "${status}"
