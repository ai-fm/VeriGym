#!/usr/bin/env bash
# Entrypoint of the VeriGym image. Unless a checkout is mounted at /workspace (as docker/run-docker.sh
# does for development), the commands run the code built into the image:
#   docker run --rm IMAGE test [ARGS]                              pytest [ARGS] and the Julia tests
#   docker run --rm -v "$PWD/results:/results" IMAGE repro [--quick | --full]
#                                                                  notebooks (quick by default), results in ./results
#   docker run --rm -it IMAGE [shell]                              a shell
# Any other command is run as given, e.g. docker run --rm IMAGE python -c "import verigym".
set -euo pipefail

SCRIPT="/results/docs/notebooks/run_notebooks.sh"

case "${1:-shell}" in
    test)
        shift
        cd /workspace
        pytest "$@"
        julia tests/julia/test_pomdps_umb.jl ;;
    repro)
        shift
        if ! mountpoint -q /results; then
            echo "Warning: no directory is mounted at /results, so the results are lost when the container exits." >&2
        fi
        if [[ ! -w /results ]]; then
            echo "Error: /results is not writable for UID $(id -u). With rootless Podman, add --userns=keep-id:uid=1000,gid=1000;" \
                "otherwise make the directory writable, e.g. chmod a+rwx results" >&2; exit 1
        fi
        # A fresh run copies the image's repository into /results; otherwise continue the previous run there.
        # Only the contents are copied: /results itself may belong to another user (a mounted host directory),
        # so cp -a could not set its timestamps.
        if [[ -z "$(ls -A /results)" ]]; then
            (shopt -s dotglob; cp -a /workspace/* /results/)
        elif [[ ! -x "$SCRIPT" ]]; then
            echo "Error: /results is neither empty nor a previous run; use an empty directory" >&2; exit 1
        fi
        exec "$SCRIPT" "$@" ;;
    shell)
        exec bash ;;
    *)
        exec "$@" ;;
esac
