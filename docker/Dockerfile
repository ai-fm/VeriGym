# VeriGym image to reproduce results and to develop in, for linux/amd64 and linux/arm64 (Linux and macOS hosts). 
# Build it with docker/run-docker.sh (which runs it on your checkout, e.g., for development), or directly
# from the repository root:
#   docker build -f docker/Dockerfile -t localhost/verigym --build-arg INSTALL_JULIA=true [--build-arg INSTALL_PRISM=true] .
# To run the code the image was shipped with (see docker/entrypoint.sh):
#   docker run --rm localhost/verigym test
#   docker run --rm -v "$PWD/results:/results" localhost/verigym repro
ARG PYTHON_VERSION=3.15
ARG JULIA_VERSION=1.10.12
ARG UV_VERSION=0.13.5
# Set below as --build-arg to install Julia or not (true or false); any other value fails at FROM julia-${INSTALL_JULIA} below ("pull access denied for julia-<value>")
ARG INSTALL_JULIA=false

# Julia is optional (~630 MB): the julia stage is the official Julia image with INSTALL_JULIA=true.
# Otherwise the COPY is executed just on an empty folder, adding nothing (A hack since a Dockerfile cannot have a conditional COPY)
FROM docker.io/library/julia:${JULIA_VERSION} AS julia-true
FROM docker.io/library/python:${PYTHON_VERSION}-slim-trixie AS julia-false
RUN mkdir -p /usr/local/julia
FROM julia-${INSTALL_JULIA} AS julia
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# trixie for glibc >= 2.38 (z3-solver wheels on arm64)
FROM docker.io/library/python:${PYTHON_VERSION}-slim-trixie

ARG TARGETARCH
ARG INSTALL_PRISM=false
# Needed again: the global INSTALL_JULIA above is only visible to FROM lines, not to RUN/ENV in this stage
ARG INSTALL_JULIA
ARG PRISM_VERSION=4.11
ARG USERNAME=verigym
ARG USER_UID=1000
ARG USER_GID=1000

# build-essential because `box2d-py` is built from source on Python >= 3.14; curl for the PRISM binary download
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential ca-certificates curl git \
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

# Non-root user (UID 1000 by default). With Docker, run-docker.sh gives it the host user's UID/GID; 
# with Podman, it maps the host user onto it. Either way, files in a mounted checkout retain ownership.
RUN groupadd -o -g "$USER_GID" "$USERNAME" \
    && useradd -o -m -s /bin/bash -u "$USER_UID" -g "$USER_GID" "$USERNAME" \
    && install -d -o "$USER_UID" -g "$USER_GID" /opt/venv /workspace /results

# The uv venv folder location is outside /workspace, such that mounting a checkout there does not hide it
ENV VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    # Below enforces a fixed Python version (supplied by the Docker image)
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:/usr/local/julia/bin:$PATH \
    # Make Python output immediately
    PYTHONUNBUFFERED=1 \
    # The entrypoint's test command uses this flag to skip the Julia tests in images without the Julia packages
    INSTALL_JULIA=$INSTALL_JULIA

USER $USERNAME
WORKDIR /workspace

# Optional: Julia packages for the external-tools workflow, at the versions used for the results.
# Precompiled for the CPU targets of the official Julia builds (targets from JuliaCI/julia-buildkite utilities/build_envs.sh)
RUN if [ "$INSTALL_JULIA" = true ]; then \
        case "$TARGETARCH" in \
            arm64) export JULIA_CPU_TARGET="generic;cortex-a57;thunderx2t99;carmel,clone_all;apple-m1,base(3);neoverse-512tvb,base(3)" ;; \
            *) export JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)" ;; \
        esac \
        && julia -e 'using Pkg; Pkg.add([PackageSpec(name=name, version=version) for (name, version) in [ \
                "POMDPs" => "1.0.0", "MCTS" => "0.5.7", "POMDPTools" => "1.1.0", \
                "JSON3" => "1.14.3", "CodecZlib" => "0.7.9", "CodecXz" => "0.7.4", \
                "ArgParse" => "1.2.0"]]); Pkg.precompile()'; \
    fi

# Dependencies first (layer cached until uv.lock changes), then verigym itself (editable)
COPY --chown=$USER_UID:$USER_GID pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project
COPY --chown=$USER_UID:$USER_GID . .
# Additional dependencies for pytest and papermill which is used by docs/notebooks/run_notebooks.sh to run notebooks (not in uv.lock).
RUN uv sync --frozen --no-cache && uv pip install --no-cache papermill pytest-xdist pytest-timeout

# Artifact commands (test, repro, shell); other commands run as given, see docker/entrypoint.sh
COPY --chmod=755 docker/entrypoint.sh /usr/local/bin/verigym
ENTRYPOINT ["verigym"]
CMD ["shell"]
