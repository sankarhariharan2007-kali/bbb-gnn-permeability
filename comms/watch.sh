#!/usr/bin/env bash
# Block until system-2's outbox changes, then exit so the task notification
# wakes system-1. ALWAYS RE-ARM after it fires -- see the banner it prints.
cd "$(dirname "$0")/.."
BRANCH=${WATCH_BRANCH:-myfork/brics-generator}
FILE=${WATCH_FILE:-comms/to-system1.md}
SELF=${WATCH_SELF:-system-1}
MAX_MIN=${1:-240}
INTERVAL=60

rearm() {
  echo
  echo "============================================================"
  echo "  RE-ARM THE WATCHER NOW -- it fired once and exited."
  echo "    bash comms/watch.sh $MAX_MIN   (run_in_background)"
  echo "  The channel is dead until you do. Re-arm EVERY time,"
  echo "  including on timeout and including when you send no reply."
  echo "  Reply only if you have something substantive; acknowledging"
  echo "  for its own sake is what turns this into a ping-pong."
  echo "============================================================"
}

before=$(git rev-parse --verify --quiet "$BRANCH:$FILE" || echo ABSENT)
deadline=$(( $(date +%s) + MAX_MIN * 60 ))
echo "[$SELF] watching $BRANCH:$FILE  (baseline $before, timeout ${MAX_MIN}m)"

while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep "$INTERVAL"
  git fetch -q myfork 2>/dev/null
  now=$(git rev-parse --verify --quiet "$BRANCH:$FILE" || echo ABSENT)
  if [ "$now" != "$before" ]; then
    echo "MESSAGE at $(date)   $before -> $now"
    echo "--- inbox ---"
    git show "$BRANCH:$FILE" 2>/dev/null | head -80
    rearm
    exit 0
  fi
done
echo "[$SELF] no message in ${MAX_MIN}m."
rearm
exit 0
