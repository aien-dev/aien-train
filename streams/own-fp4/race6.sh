#!/bin/bash
# race6.sh (L284 oracle): resume after Drake's 35B stop (22:20Z 2026-10-20). The chip belongs to L306 until the 35B serves on 18091;
# wait for that port, then run race4.sh (each run under /tmp/aien-gb10.lock) on the whole remaining queue. Usage: race6.sh "recipe:seed ..."
while ! ss -ltn | grep -q ':18091 '; do sleep 60; done
echo "$(date -u +%H:%MZ) 18091 is up, starting queue"
exec "$(dirname "$0")/race4.sh" "$1"
