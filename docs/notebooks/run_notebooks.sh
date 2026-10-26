#!/usr/bin/env bash
# Run the example notebooks listed in NOTEBOOKS below and report which ones fail.
#
# Usage: docs/notebooks/run_notebooks.sh [--inplace] [--quick | --full]
#   --inplace          save the outputs into the notebooks themselves
#   --quick            quick run, sets QUICK=1 (the default)
#   --full             full run, sets QUICK=0
# Without --inplace, the executed notebooks are saved to a temp folder instead.

set -u  # stop if a variable is used before it is set (catches typos)

# Default settings, changed by the flags below
INPLACE=1     # 1 = overwrite the notebooks with the new outputs, 0 = save copies to a temp folder
QUICK=1       # 1 = quick run (--quick), 0 = full run (--full)

cd "$(dirname "$0")/.."  # move into docs/, wherever the script is called from

# Notebooks to run, in this order (paths relative to docs/)
NOTEBOOKS=(
    notebooks/workflow_external_tools.ipynb
    notebooks/workflow_comparing_abstraction_policies.ipynb
    notebooks/workflow_shielding.ipynb
    notebooks/workflow_abstraction_refinement.ipynb
    notebooks/workflow_compare_rl_model_checking.ipynb
    notebooks/workflow_comparing_exploration_policies.ipynb
    notebooks/workflow_featurereduction.ipynb
    # tech eval
    notebooks/tech_eval_discretization_overhead.ipynb
    notebooks/tech_eval_mdp_memory.ipynb
    notebooks/tech_eval_runtimes.ipynb
)


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

# Where to save the executed notebooks
if [[ $INPLACE -eq 1 ]]; then
    OUT_DIR=.  # this folder, so the originals get overwritten
else
    OUT_DIR="$(mktemp -d)"
    echo "Saving executed notebooks to $OUT_DIR"
fi

# papermill is not a project dependency: use it where the current Python has it (e.g. the Docker image),
# otherwise uv adds it just for this run.
if python -c "import papermill" 2>/dev/null; then
    papermill=(python -m papermill)
else
    papermill=(uv run --group dev --with papermill papermill)
fi

failed=()  # notebooks that raised an error
passed=()  # notebooks that ran through

for nb in "${NOTEBOOKS[@]}"; do
    echo "\n=== $nb \n\n"
    mkdir -p "$OUT_DIR/$(dirname "$nb")"  # create the matching subfolder for the copy
    # Run all cells, printing each cell's output as it finishes (--log-output).
    # --cwd runs the notebook from its own folder, so its relative file paths work.
    if ! "${papermill[@]}" "$nb" "$OUT_DIR/$nb" \
        --cwd "$(dirname "$nb")" --log-output; then
        failed+=("$nb")
    else
        passed+=("$nb")
    fi
done

echo
if [[ ${#passed[@]} -gt 0 ]]; then
    echo "Passed:"
    printf '  ✅ %s\n' "${passed[@]}"
fi
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All notebooks passed."
else
    echo "Failed:"
    printf '  ❌ %s\n' "${failed[@]}"
    exit 1  # non-zero exit code signals failure (e.g. to CI)
fi
