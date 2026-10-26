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
            echo "Error: /results is not writable for UID $(id -u); make it writable, e.g. chmod a+rwx results" >&2; exit 1
        fi
        # A fresh run copies the image's repository into /results; otherwise continue the previous run there
        if [[ -z "$(ls -A /results)" ]]; then
            cp -a /workspace/. /results/
        elif [[ ! -x "$SCRIPT" ]]; then
            echo "Error: /results is neither empty nor a previous run; use an empty directory" >&2; exit 1
        fi
        exec "$SCRIPT" "$@" ;;
    shell)
        exec bash ;;
    *)
        exec "$@" ;;
esac
