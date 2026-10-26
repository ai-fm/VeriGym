#!/usr/bin/env bash
# Build and run the VeriGym container (see Dockerfile).
#
# Usage: ./docker/run-docker.sh build [OPTIONS]   build the image:
#                                                   --prism adds Java and PRISM,
#                                                   --install-julia installs and precompiles the Julia packages
#        ./docker/run-docker.sh test [ARGS]       run pytest [ARGS] and the Julia tests in the container
#        ./docker/run-docker.sh repro [ARGS]      run run_workflows.sh [ARGS] in the container
#        ./docker/run-docker.sh shell             open a shell in the container
#        ./docker/run-docker.sh save              export the image to verigym-<arch>.tar.gz
#        ./docker/run-docker.sh load FILE         import an image exported with save
#        ./docker/run-docker.sh pull REF          pull an image, e.g. from ghcr.io, and use it
#
# For development: the repository is mounted at /workspace, so the container runs your local code and
# results stay on the host. To run the code built into an image instead (artifact evaluation), use the
# image's own commands (docker run --rm IMAGE test|repro, see docker/entrypoint.sh and README.md).
# Set OFFLINE=1 to run without network access, e.g. OFFLINE=1 ./docker/run-docker.sh test
# Set IMAGE to use another image name (default: localhost/verigym).

set -euo pipefail
# Work from the repository root: it is the build context and is mounted at /workspace
cd "$(dirname "$0")/.."

# localhost/verigym is the same name in Docker and Podman, so a saved image loads under it in both
IMAGE="${IMAGE:-localhost/verigym}"
# :z relabels the mount for SELinux (e.g. Fedora); ignored where SELinux is off.
# Coverage data stays in the container: its /workspace paths would break the host's .coverage
run_args=(--rm -v "$PWD:/workspace:z" -e COVERAGE_FILE=/tmp/.coverage)
# Rootless Podman: map the host user onto the image user (UID 1000), whatever the host UID
podman=0
if [[ "$(docker --version 2>/dev/null)" == *podman* ]]; then
    podman=1
    run_args+=("--userns=keep-id:uid=1000,gid=1000")
fi
if [[ "${OFFLINE:-0}" == 1 ]]; then run_args+=(--network none); fi
# In a terminal: colours, progress bars, and Ctrl+C reaches pytest; not in CI, pipes or log files
if [[ -t 0 && -t 1 ]]; then run_args+=(-it); fi

case "${1:-}" in
    build)
        if [[ "${OFFLINE:-0}" == 1 ]]; then
            echo "Error: building the image needs network access; run it either without OFFLINE=1 (or, equivalently, set OFFLINE=0)." >&2; exit 2
        fi
        # Record the commit the image is built from (-dirty: with uncommitted changes)
        rev="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        if [[ -n "$(git status --porcelain 2>/dev/null)" ]]; then rev+="-dirty"; fi
        build_args=(-f docker/Dockerfile --label org.opencontainers.image.revision="$rev")
        # Docker: the image user gets the host UID/GID. Podman: keeps UID 1000 and maps it at runtime.
        if (( ! podman )); then build_args+=(--build-arg USER_UID="$(id -u)" --build-arg USER_GID="$(id -g)"); fi
        shift
        for option in "$@"; do
            case "$option" in
                --prism) build_args+=(--build-arg INSTALL_PRISM=true) ;;
                --julia) build_args+=(--build-arg INSTALL_JULIA=true) ;;
                *) echo "Error: unknown build option: $option" >&2; exit 2 ;;
            esac
        done
        docker build "${build_args[@]}" -t "$IMAGE" . ;;
    test)
        shift
        docker run "${run_args[@]}" "$IMAGE" test "$@" ;;
    repro)
        shift
        docker run "${run_args[@]}" "$IMAGE" "docs/notebooks/run_workflows.sh" "$@" ;;
    shell)
        docker run "${run_args[@]}" "$IMAGE" ;;
    save)
        out="verigym-$(docker image inspect -f '{{.Architecture}}' "$IMAGE").tar.gz"
        # Write to .part first, so a failed export never leaves a complete-looking archive
        docker save "$IMAGE" | gzip > "$out.part"
        mv "$out.part" "$out"
        echo "Saved $IMAGE to $out" ;;
    load)
        docker load -i "${2:?usage: ./docker/run-docker.sh load FILE}" ;;
    pull)
        # Pull an image (e.g. the multi-platform one from ghcr.io) and use it under $IMAGE from now on
        docker pull "${2:?usage: ./docker/run-docker.sh pull REF}"
        docker tag "$2" "$IMAGE" ;;
    *)
        sed -n '/^# Usage:/,/^#$/p' "docker/$(basename "$0")"; exit 2 ;;
esac
