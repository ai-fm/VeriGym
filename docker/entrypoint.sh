#!/usr/bin/env bash
# Entrypoint of the VeriGym image. Unless a checkout is mounted at /workspace (as docker/run-docker.sh
# does for development), the commands run the code built into the image:
#   docker run --rm IMAGE test [ARGS]                              pytest [ARGS], plus the Julia tests if the
#                                                                  image has the Julia packages (build --julia)
#   docker run --rm -v "$PWD/results:/results" IMAGE repro [--quick | --full]
#                                                                  notebooks (quick by default); only their outputs go to
#                                                                  ./results (logs/, workflows/, tech_eval/)
#   docker run --rm -it IMAGE [shell]                              a shell
# Any other command is run as given, e.g. docker run --rm IMAGE python -c "import verigym".
set -euo pipefail

OUTPUT=/workspace/docs/notebooks/output  # where the notebooks write everything they produce

case "${1:-shell}" in
    test)
        shift
        cd /workspace
        pytest "$@"
        if [[ "${INSTALL_JULIA:-false}" == true ]]; then
            julia tests/julia/test_pomdps_umb.jl
        else
            echo "Skipping the Julia tests: the image was built without Julia installed." >&2
        fi ;;
    repro)
        shift
        if mountpoint -q /workspace; then
            echo "Error: repro runs the image's own code and would replace docs/notebooks/output in the mounted checkout;" \
                "run docs/notebooks/run_notebooks.sh instead (as docker/run-docker.sh repro does)" >&2; exit 1
        fi
        if ! mountpoint -q /results; then
            echo "Warning: no directory is mounted at /results, so the results are lost when the container exits." >&2
        fi
        if [[ ! -w /results ]]; then
            echo "Error: /results is not writable for UID $(id -u). With rootless Podman, add --userns=keep-id:uid=1000,gid=1000;" \
                "otherwise make the directory writable, e.g. chmod a+rwx results" >&2; exit 1
        fi
        if [[ -n "$(ls -A /results)" ]]; then
            echo "Continuing with the outputs already in /results (use an empty directory for a fresh run)" >&2
        fi
        # Run the notebooks from the image's own copy, with their output folder linked to /results: the mounted
        # directory receives only the outputs (logs, figures, models, data)
        rm -rf "$OUTPUT"
        ln -s /results "$OUTPUT"
        exec /workspace/docs/notebooks/run_notebooks.sh "$@" ;;
    shell)
        exec bash ;;
    *)
        exec "$@" ;;
esac
