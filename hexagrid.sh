#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# hexagrid.sh — HexaGrid uvicorn service manager
#
# Usage:
#   ./hexagrid.sh start
#   ./hexagrid.sh stop
#   ./hexagrid.sh restart
#   ./hexagrid.sh status
#   ./hexagrid.sh logs          # tail -f last 50 lines
#   ./hexagrid.sh logs 200      # tail last N lines
#
# Install (run once):
#   chmod +x hexagrid.sh
#   sudo cp hexagrid.sh /usr/local/bin/hexagrid
#   Then use:  hexagrid start | stop | restart | logs | status
# ─────────────────────────────────────────────────────────────────────────────

# ── Config — adjust if your layout differs ───────────────────────────────────
APP_DIR="$HOME/hexagrid/api"
APP_MODULE="api:app"
HOST="0.0.0.0"
PORT="8000"
WORKERS="1"                          # keep at 1 — shared in-memory state
LOG_FILE="$HOME/hexagrid/uvicorn.log"
PID_FILE="$HOME/hexagrid/uvicorn.pid"
VENV="$HOME/hexagrid/venv"           # set to "" if not using a venv
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'; YEL='\033[1;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'
TS() { date '+%H:%M:%S'; }
OK()   { echo -e "  ${GRN}✓${RST}  $*"; }
ERR()  { echo -e "  ${RED}✗${RST}  $*"; }
INFO() { echo -e "  ${CYN}→${RST}  $*"; }
WARN() { echo -e "  ${YEL}!${RST}  $*"; }

header() {
    echo -e "${BLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
    echo -e "${BLD}  HexaGrid  [$(TS)]  $1${RST}"
    echo -e "${BLD}${CYN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RST}"
}

# Resolve uvicorn binary — venv first, then PATH
uvicorn_bin() {
    if [[ -n "$VENV" && -x "$VENV/bin/uvicorn" ]]; then
        echo "$VENV/bin/uvicorn"
    elif command -v uvicorn &>/dev/null; then
        command -v uvicorn
    else
        echo ""
    fi
}

get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null)
        # Verify process is actually running
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
        else
            rm -f "$PID_FILE"
            echo ""
        fi
    else
        # Fallback: scan processes
        pgrep -f "uvicorn.*${APP_MODULE}" 2>/dev/null | head -1
    fi
}

# ── start ─────────────────────────────────────────────────────────────────────
cmd_start() {
    header "Starting"

    local pid
    pid=$(get_pid)
    if [[ -n "$pid" ]]; then
        WARN "Already running (PID $pid) — use 'restart' to reload"
        return 0
    fi

    local bin
    bin=$(uvicorn_bin)
    if [[ -z "$bin" ]]; then
        ERR "uvicorn not found — activate your venv or: pip install uvicorn"
        return 1
    fi

    # Activate venv if present
    if [[ -n "$VENV" && -f "$VENV/bin/activate" ]]; then
        # shellcheck source=/dev/null
        source "$VENV/bin/activate"
        INFO "venv: $VENV"
    fi

    mkdir -p "$(dirname "$LOG_FILE")"

    INFO "Dir:     $APP_DIR"
    INFO "Module:  $APP_MODULE"
    INFO "Listen:  $HOST:$PORT"
    INFO "Log:     $LOG_FILE"
    echo ""

    # ── Source auth environment variables ─────────────────────────────────────
    local env_auth="$HOME/hexagrid/.env.auth"
    if [[ -f "$env_auth" ]]; then
        # shellcheck source=/dev/null
        source "$env_auth"
        INFO "Auth env: $env_auth"
    else
        WARN ".env.auth not found at $env_auth — JWT auth may not work"
        WARN "Run setup_credentials.py to create it"
    fi

    cd "$APP_DIR" || { ERR "Cannot cd to $APP_DIR"; return 1; }

    nohup "$bin" "$APP_MODULE" \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level info \
        >> "$LOG_FILE" 2>&1 &

    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"

    # Give it a moment to either bind or crash
    sleep 1.5

    if kill -0 "$new_pid" 2>/dev/null; then
        OK "Started — PID $new_pid"
        # Quick health check
        sleep 0.5
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            "http://localhost:${PORT}/api/v1/health" --max-time 3 2>/dev/null || echo "000")
        if [[ "$http_code" == "200" ]]; then
            OK "Health check passed (HTTP 200)"
        else
            WARN "Health check returned HTTP $http_code — check logs if API calls fail"
        fi
        echo ""
        INFO "Dashboard: http://localhost:${PORT}/"
        INFO "API docs:  http://localhost:${PORT}/docs"
    else
        ERR "Process exited immediately — check logs:"
        echo ""
        tail -20 "$LOG_FILE"
        rm -f "$PID_FILE"
        return 1
    fi
}

# ── stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
    header "Stopping"

    local pid
    pid=$(get_pid)
    if [[ -z "$pid" ]]; then
        WARN "Not running"
        return 0
    fi

    INFO "Sending SIGTERM to PID $pid..."
    kill -TERM "$pid" 2>/dev/null

    # Wait up to 8 seconds for clean shutdown
    local waited=0
    while kill -0 "$pid" 2>/dev/null && [[ $waited -lt 8 ]]; do
        sleep 0.5
        ((waited++))
    done

    if kill -0 "$pid" 2>/dev/null; then
        WARN "Did not stop cleanly — sending SIGKILL"
        kill -KILL "$pid" 2>/dev/null
        sleep 0.5
    fi

    rm -f "$PID_FILE"

    if kill -0 "$pid" 2>/dev/null; then
        ERR "Could not stop PID $pid"
        return 1
    else
        OK "Stopped"
    fi
}

# ── restart ───────────────────────────────────────────────────────────────────
cmd_restart() {
    header "Restarting"
    cmd_stop
    echo ""
    sleep 0.5
    cmd_start
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
    header "Status"

    local pid
    pid=$(get_pid)

    if [[ -n "$pid" ]]; then
        OK "Running — PID $pid"

        # Process details
        local cpu mem
        cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | xargs)
        mem=$(ps -p "$pid" -o %mem= 2>/dev/null | xargs)
        local rss
        rss=$(ps -p "$pid" -o rss= 2>/dev/null | xargs)
        rss=$(( ${rss:-0} / 1024 ))
        local started
        started=$(ps -p "$pid" -o lstart= 2>/dev/null | xargs)

        INFO "Started: $started"
        INFO "CPU:     ${cpu}%   Memory: ${mem}% (${rss} MB RSS)"

        # Health check
        local http_code
        http_code=$(curl -s -o /dev/null -w "%{http_code}" \
            "http://localhost:${PORT}/api/v1/health" --max-time 3 2>/dev/null || echo "000")
        if [[ "$http_code" == "200" ]]; then
            OK "API health: HTTP 200"
        else
            WARN "API health: HTTP $http_code (server may still be starting)"
        fi

        # Telemetry DB
        local db="$HOME/hexagrid/data/telemetry.db"
        [[ ! -f "$db" ]] && db="/var/lib/hexagrid/telemetry.db"
        if [[ -f "$db" ]]; then
            local db_size
            db_size=$(du -sh "$db" 2>/dev/null | cut -f1)
            local row_count
            row_count=$(python3 -c \
                "import sqlite3; c=sqlite3.connect('$db'); print(c.execute('SELECT COUNT(*) FROM gpu_telemetry').fetchone()[0])" \
                2>/dev/null || echo "?")
            INFO "Telemetry DB: $db_size  ($row_count rows in gpu_telemetry)"
        else
            WARN "Telemetry DB not found at $db"
        fi

        echo ""
        INFO "Dashboard: http://localhost:${PORT}/"
        INFO "API docs:  http://localhost:${PORT}/docs"
    else
        ERR "Not running"
        if [[ -f "$LOG_FILE" ]]; then
            echo ""
            WARN "Last 10 log lines:"
            tail -10 "$LOG_FILE" | sed 's/^/    /'
        fi
        return 1
    fi
}

# ── logs ──────────────────────────────────────────────────────────────────────
cmd_logs() {
    local lines="${1:-50}"
    header "Logs (last $lines lines, following)"

    if [[ ! -f "$LOG_FILE" ]]; then
        ERR "Log file not found: $LOG_FILE"
        INFO "Start the server first with: hexagrid start"
        return 1
    fi

    INFO "File: $LOG_FILE"
    INFO "Press Ctrl+C to stop following"
    echo ""
    tail -n "$lines" -f "$LOG_FILE"
}

# ── help ──────────────────────────────────────────────────────────────────────
cmd_help() {
    echo ""
    echo -e "${BLD}hexagrid${RST} — HexaGrid uvicorn service manager"
    echo ""
    echo -e "  ${BLD}hexagrid start${RST}          Start uvicorn in the background"
    echo -e "  ${BLD}hexagrid stop${RST}           Stop uvicorn gracefully"
    echo -e "  ${BLD}hexagrid restart${RST}        Stop then start (picks up code changes)"
    echo -e "  ${BLD}hexagrid status${RST}         PID, CPU/mem, API health, telemetry DB info"
    echo -e "  ${BLD}hexagrid logs${RST}           Follow log (last 50 lines)"
    echo -e "  ${BLD}hexagrid logs 200${RST}       Follow log (last N lines)"
    echo ""
    echo -e "  Config at top of script:"
    echo -e "    APP_DIR   = $APP_DIR"
    echo -e "    PORT      = $PORT"
    echo -e "    LOG_FILE  = $LOG_FILE"
    echo -e "    VENV      = ${VENV:-'(none)'}"
    echo ""
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "${1:-help}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_restart ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${2:-50}" ;;
    help|--help|-h) cmd_help ;;
    *)
        ERR "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
