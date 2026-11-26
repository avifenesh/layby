#!/usr/bin/env bash
# cg_exec.sh CGROUP_PATH CMD [ARG...]: move this process into the cgroup v2 directory CGROUP_PATH, then exec CMD, so CMD
# and every child it starts (engine core, workers, I/O threads) are charged to that cgroup from their first page.
# run_arms.sh and run_arms_sglang.sh put it in front of the server command through SERVER_WRAP. Writing 0 to
# cgroup.procs moves the writer itself, which also works inside a container whose PID namespace differs from the
# host's (the container needs --privileged --cgroupns=host so /sys/fs/cgroup is the host's and writable).
# An empty CGROUP_PATH runs CMD unconfined (used by tests); a failed move exits 97 without running CMD.
set -u
CG=${1?usage: cg_exec.sh CGROUP_PATH CMD [ARG...]}; shift
if [ -n "$CG" ]; then
  echo 0 > "$CG/cgroup.procs" || { echo "cg_exec: cannot join $CG" >&2; exit 97; }
fi
exec "$@"
