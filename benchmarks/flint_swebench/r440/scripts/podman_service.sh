#!/usr/bin/env bash
# (re)start the rootless podman API socket used by the swebench docker SDK
. $HOME/flint-swebench/env.sh
mkdir -p $XDG_RUNTIME_DIR/podman
if curl -s -m 5 --unix-socket $XDG_RUNTIME_DIR/podman/podman.sock http://d/_ping | grep -q OK; then exit 0; fi
rm -f $XDG_RUNTIME_DIR/podman/podman.sock
nohup setsid nice -n 19 ionice -c3 $PODMAN system service --time=0 \
  unix://$XDG_RUNTIME_DIR/podman/podman.sock > $R/run/podman-service.log 2>&1 < /dev/null &
for i in $(seq 1 20); do
  curl -s -m 2 --unix-socket $XDG_RUNTIME_DIR/podman/podman.sock http://d/_ping | grep -q OK && exit 0
  sleep 0.5
done
echo "podman service failed to start" >&2; exit 1
