#!/bin/bash
# race5.sh (L284 oracle): second queue; starts when race4's process (pid in $1) is gone, then runs race4.sh on "$2".
while kill -0 "$1" 2>/dev/null; do sleep 60; done
exec "$(dirname "$0")/race4.sh" "$2"
