#!/bin/bash
# Usage: bash run_nohup.sh <script.sh>
if [ -z "$1" ]; then
    echo "Usage: bash run_nohup.sh <script.sh>"
    exit 1
fi

SCRIPT="$1"
shift
mkdir -p run_logs/date
LOG="run_logs/date/run_$(date +"%m%d%Y_%H%M").out"
nohup bash "$SCRIPT" "$@" > "$LOG" 2>&1 &
echo "Started: $SCRIPT $*"
echo "Log:     $LOG"
echo "PID:     $!"
