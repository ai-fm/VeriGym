# VeriGym image to reproduce the results and to develop in, for linux/amd64 and linux/arm64 (Linux and macOS hosts). 
# Build and run it with docker/run-docker.sh, or directly from the repository root:
#   docker build -f docker/Dockerfile -t localhost/verigym [--build-arg INSTALL_PRISM=true] .
#   docker run --rm -it -v "$PWD:/workspace" localhost/verigym
ARG PYTHON_VERSION=3.14
ARG JULIA_VERSION=1.11.5
ARG UV_VERSION=0.12.19

FROM docker.io/library/julia:${JULIA_VERSION} AS julia
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# trixie for glibc >= 2.38 (z3-solver wheels on arm64)
FROM docker.io/library/python:${PYTHON_VERSION}-slim-trixie

ARG TARGETARCH
ARG INSTALL_PRISM=false
ARG PRISM_VERSION=4.11
ARG USERNAME=verigym
ARG USER_UID=1000
ARG USER_GID=1000

# build-essential: box2d-py is built from source on Python 3.14; curl: PRISM download
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
        $(if [ "$INSTALL_PRISM" = true ]; then echo default-jre-headless; fi) \
    && rm -rf /var/lib/apt/lists/*

# Optional: PRISM binary release (x86 or arm build)
RUN if [ "$INSTALL_PRISM" = true ]; then \
        arch=$(if [ "$TARGETARCH" = arm64 ]; then echo arm; else echo x86; fi) \
        && mkdir /opt/prism \
        && curl -fsSL "https://www.prismmodelchecker.org/dl/prism-${PRISM_VERSION}-linux64-${arch}.tar.gz" \
            | tar -xz -C /opt/prism --strip-components=1 \
        && cd /opt/prism && ./install.sh \
        && ln -s /opt/prism/bin/prism /usr/local/bin/prism; \
    fi

COPY --from=julia /usr/local/julia /usr/local/julia
COPY --from=uv /uv /uvx /usr/local/bin/

# Non-root user (UID 1000 by default). With Docker, run-docker.sh gives it the host user's UID/GID; with Podman,
# it maps the host user onto it. Either way, files in a mounted checkout retain ownership.
RUN groupadd -o -g "$USER_GID" "$USERNAME" \
    && useradd -o -m -s /bin/bash -u "$USER_UID" -g "$USER_GID" "$USERNAME" \
    && install -d -o "$USER_UID" -g "$USER_GID" /opt/venv /workspace

# The uv venv lives outside /workspace, so mounting a checkout there does not hide it
ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # Below enforces a fixed Python version (supplied by the Docker image)
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:/usr/local/julia/bin:$PATH \
    # Make Python output immediately
    PYTHONUNBUFFERED=1

USER $USERNAME
WORKDIR /workspace

# Julia packages for the external-tools workflow, at the versions used for the results.
# Precompiled for the CPU targets of the official Julia builds (targets from JuliaCI/julia-buildkite utilities/build_envs.sh)
RUN case "$TARGETARCH" in \
        arm64) export JULIA_CPU_TARGET="generic;cortex-a57;thunderx2t99;carmel,clone_all;apple-m1,base(3);neoverse-512tvb,base(3)" ;; \
        *) export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)" ;; \
    esac \
    && julia -e 'using Pkg; pkg"add POMDPs@1.0.0 MCTS@0.5.7 POMDPTools@1.1.0 JSON3@1.14.3 CodecZlib@0.7.9 CodecXz@0.7.4"; Pkg.precompile()'

# Dependencies first (layer cached until uv.lock changes), then verigym itself (editable)
COPY --chown=$USER_UID:$USER_GID pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project
COPY --chown=$USER_UID:$USER_GID . .
# Additional dependencies for pytest and papermill which is used by examples/run_workflows.sh to run notebooks (not in uv.lock).
RUN uv sync --frozen --no-cache && uv pip install --no-cache papermill pytest-xdist pytest-timeout

CMD ["bash"]
