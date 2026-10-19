#!/usr/bin/env bash
# Build and run the VeriGym container (see Dockerfile).
#
# Usage: ./run.sh build [--prism]   build the image; --prism adds Java and PRISM
#        ./run.sh test [ARGS]       run pytest [ARGS] and the Julia tests in the container
#        ./run.sh repro [ARGS]      run examples/run_workflows.sh [ARGS] in the container
#        ./run.sh shell             open a shell in the container
#
# The repository is mounted at /workspace, so code changes and results stay on the host.
# Set OFFLINE=1 to run without network access, e.g. OFFLINE=1 ./run.sh test

set -euo pipefail
cd "$(dirname "$0")"

IMAGE=verigym
# :z relabels the mount for SELinux (e.g. Fedora); ignored where SELinux is off.
# Coverage data stays in the container: its /workspace paths would break the host's .coverage
run_args=(--rm -v "$PWD:/workspace:z" -e COVERAGE_FILE=/tmp/.coverage)
# Rootless Podman: keep the host user's UID inside the container
if [[ "$(docker --version 2>/dev/null)" == *podman* ]]; then run_args+=(--userns=keep-id); fi
if [[ "${OFFLINE:-0}" == 1 ]]; then run_args+=(--network none); fi
# In a terminal: colours, progress bars, and Ctrl+C reaches pytest; not in CI, pipes or log files
if [[ -t 0 && -t 1 ]]; then run_args+=(-it); fi

case "${1:-}" in
    build)
        if [[ "${OFFLINE:-0}" == 1 ]]; then
            echo "Error: building the image needs network access; run it either without OFFLINE=1 (or, equivalently, set OFFLINE=0)." >&2; exit 2
        fi
        build_args=(--build-arg USER_UID="$(id -u)" --build-arg USER_GID="$(id -g)")
        if [[ "${2:-}" == --prism ]]; then build_args+=(--build-arg INSTALL_PRISM=true); fi
        docker build "${build_args[@]}" -t "$IMAGE" . ;;
    test)
        shift
        docker run "${run_args[@]}" "$IMAGE" \
            bash -c 'pytest "$@" && julia tests/julia/test_pomdps_umb.jl' bash "$@" ;;
    repro)
        shift
        docker run "${run_args[@]}" "$IMAGE" examples/run_workflows.sh "$@" ;;
    shell)
        docker run "${run_args[@]}" "$IMAGE" ;;
    *)
        sed -n '/^# Usage:/,/^#$/p' "$(basename "$0")"; exit 2 ;;
esac
