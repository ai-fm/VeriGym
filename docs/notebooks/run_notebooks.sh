#!/usr/bin/env bash
# Run the example notebooks listed in NOTEBOOKS below and report which ones fail.
#
# Usage: docs/notebooks/run_notebooks.sh [--inplace] [--quick | --full]
#   --inplace   overwrite the notebooks themselves with the executed versions
#               (default: save them to notebooks/output/executed)
#   --quick     quick run, sets QUICK=1 (the default)
#   --full      full run, sets QUICK=0
# The terminal only shows progress and failures; each notebook's full output goes to notebooks/output/logs.

set -u  # stop if a variable is used before it is set (catches typos)

# Default settings, changed by the flags below
INPLACE=0     # 1 = overwrite the notebooks with the executed versions (--inplace), 0 = save them to EXEC_DIR
QUICK=1       # 1 = quick run (--quick), 0 = full run (--full)

cd "$(dirname "$0")/.."  # move into docs/, wherever the script is called from

# Notebooks to run, in this order (paths relative to docs/)
NOTEBOOKS=(
    notebooks/workflow_external_tools.ipynb
    notebooks/workflow_comparing_abstraction_policies.ipynb
    notebooks/workflow_shielding.ipynb
    notebooks/workflow_comparing_exploration_policies.ipynb
    notebooks/tech_eval_discretization_overhead.ipynb
    notebooks/tech_eval_mdp_memory.ipynb
    notebooks/workflow_compare_rl_model_checking.ipynb
    notebooks/workflow_abstraction_refinement.ipynb
    notebooks/workflow_featurereduction.ipynb
    notebooks/tech_eval_runtimes.ipynb
)

# Everything a run produces goes under notebooks/output (next to the notebooks' own outputs)
EXEC_DIR=notebooks/output/executed  # executed notebooks (unless --inplace)
LOG_DIR=notebooks/output/logs       # one log per notebook: everything papermill and the cells print

# Read the flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --inplace) INPLACE=1; shift ;;
        --quick) QUICK=1; shift ;;
        --full) QUICK=0; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

export QUICK  # the notebooks read it from the environment
export PROGRESS_BAR=0  # no training progress bars: nobody watches them, and their updates flood the logs and kernel output

mkdir -p "$EXEC_DIR" "$LOG_DIR"

# papermill is not a project dependency: use it where the current Python has it (e.g. the Docker image),
# otherwise uv adds it just for this run.
if python -c "import papermill" 2>/dev/null; then
    papermill=(python -m papermill)
else
    papermill=(uv run --group dev --with papermill papermill)
fi

# Minutes and seconds since the given $SECONDS value
elapsed() { local t=$((SECONDS - $1)); printf '%dm%02ds' $((t / 60)) $((t % 60)); }

failed=()  # notebooks that raised an error

echo "Running ${#NOTEBOOKS[@]} notebooks (QUICK=$QUICK); logs in docs/$LOG_DIR/"
for i in "${!NOTEBOOKS[@]}"; do
    nb=${NOTEBOOKS[$i]}
    name=$(basename "$nb" .ipynb)
    if [[ $INPLACE -eq 1 ]]; then out=$nb; else out="$EXEC_DIR/$name.ipynb"; fi
    log="$LOG_DIR/$name.log"
    printf '[%d/%d] %s ... ' $((i + 1)) ${#NOTEBOOKS[@]} "$name"
    start=$SECONDS
    # All output goes to the log: each cell's output as it finishes (--log-output), no progress bar.
    # --cwd runs the notebook from its own folder, so its relative file paths work.
    if "${papermill[@]}" "$nb" "$out" --cwd "$(dirname "$nb")" \
        --log-output --no-progress-bar > "$log" 2>&1; then
        echo "✅ $(elapsed "$start")"
    else
        echo "❌ $(elapsed "$start"), see docs/$log"
        tail -n 15 "$log" | sed 's/^/    /'  # the end of the log holds the error
        failed+=("$nb")
    fi
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All ${#NOTEBOOKS[@]} notebooks passed."
else
    echo "${#failed[@]} of ${#NOTEBOOKS[@]} notebooks failed:"
    printf '  ❌ %s\n' "${failed[@]}"
    exit 1  # non-zero exit code signals failure (e.g. to CI)
fi
