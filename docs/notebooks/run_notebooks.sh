#!/usr/bin/env bash
# Run the experiment scripts listed below and report which ones fail.
#
# Usage: docs/notebooks/run_notebooks.sh [--quick | --full]
#   --quick     quick run, sets QUICK=1 (the default)
#   --full      full run, sets QUICK=0
# The terminal only shows progress and failures; each script's full output goes to notebooks/output/logs.

set -u  # stop if a variable is used before it is set (catches typos)

# Default settings, changed by the flags below
QUICK=1       # 1 = quick run (--quick), 0 = full run (--full)

cd "$(dirname "$0")/.."  # move into docs/, wherever the script is called from

# Scripts to run, in this order (paths relative to docs/)
NOTEBOOKS=(
    notebooks/workflow_external_tools.py
    notebooks/workflow_comparing_abstraction_policies.py
    notebooks/workflow_shielding.py
    notebooks/workflow_comparing_exploration_policies.py
    notebooks/tech_eval_discretization_overhead.py
    notebooks/tech_eval_mdp_memory.py
    notebooks/workflow_compare_rl_model_checking.py
    notebooks/workflow_abstraction_refinement.py
    notebooks/workflow_featurereduction.py
    notebooks/tech_eval_runtimes.py
)

# Everything a run produces goes under notebooks/output (next to the notebooks' own outputs)
LOG_DIR=notebooks/output/logs       # one log per script: everything it prints

# Read the flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --full) QUICK=0; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

export QUICK  # the scripts read it from the environment
export PROGRESS_BAR=0  # no training progress bars: nobody watches them, and their updates flood the logs
export MPLBACKEND=Agg  # no plot windows: plt.show() does nothing, figures are still saved
export PYTHONUNBUFFERED=1  # write output to the log as it happens, so a crash still saves the full log

mkdir -p "$LOG_DIR"

# Use the current Python where it has verigym (e.g. the Docker image), otherwise the project's uv environment.
if python -c "import verigym" 2>/dev/null; then
    python=(python)
else
    python=(uv run --group dev python)
fi

# Minutes and seconds since the given $SECONDS value
elapsed() { local t=$((SECONDS - $1)); printf '%dm%02ds' $((t / 60)) $((t % 60)); }

failed=()  # scripts that exited with an error

echo "Running ${#NOTEBOOKS[@]} notebook scripts (QUICK=$QUICK); logs in docs/$LOG_DIR/"
for i in "${!NOTEBOOKS[@]}"; do
    nb=${NOTEBOOKS[$i]}
    name=$(basename "$nb" .py)
    log="$LOG_DIR/$name.log"
    printf '[%d/%d] %s ... ' $((i + 1)) ${#NOTEBOOKS[@]} "$name"
    start=$SECONDS
    # Run from the script's own folder, so its relative file paths (and imports of workflow_utils etc.) work.
    if (cd "$(dirname "$nb")" && "${python[@]}" "$(basename "$nb")") > "$log" 2>&1; then
        echo "✅ $(elapsed "$start")"
    else
        echo "❌ $(elapsed "$start"), see docs/$log"
        tail -n 15 "$log" | sed 's/^/    /'  # the end of the log holds the error
        failed+=("$nb")
    fi
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All ${#NOTEBOOKS[@]} notebook scripts passed."
else
    echo "${#failed[@]} of ${#NOTEBOOKS[@]} notebook scripts failed:"
    printf '  ❌ %s\n' "${failed[@]}"
    exit 1  # non-zero exit code signals failure (e.g. to CI)
fi
