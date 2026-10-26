#!/usr/bin/env bash
# Run the example notebooks listed in NOTEBOOKS below and report which ones fail.
#
# Usage: docs/workflows/run_workflows.sh [--inplace] [--quick] [--timeout SECONDS]
#   --inplace          save the outputs into the notebooks themselves
#   --quick            quick runs, sets QUICK=1 (default QUICK=0)
#   --timeout SECONDS  max seconds per cell (default 1800, -1 = no limit)
# Without --inplace, the executed notebooks are saved to a temp folder instead.

set -u  # stop if a variable is used before it is set (catches typos)

# Default settings, changed by the flags below
INPLACE=1     # 1 = overwrite the notebooks with the new outputs, 0 = save copies to a temp folder
QUICK=0       # 1 = quick run (--quick), 0 = elaborate run

cd "$(dirname "$0")/.."  # move into docs/, wherever the script is called from

# Notebooks to run, in this order (paths relative to docs/)
NOTEBOOKS=(
    workflows/workflow_external_tools.ipynb
    workflows/workflow_comparing_abstraction_policies.ipynb
    workflows/workflow_shielding.ipynb
    workflows/workflow_abstraction_refinement.ipynb
    workflows/workflow_compare_rl_model_checking.ipynb
    workflows/workflow_comparing_exploration_policies.ipynb
    workflows/workflow_featurereduction.ipynb
    # tech eval
    workflows/tech_eval_discretization_overhead.ipynb
    workflows/tech_eval_mdp_memory.ipynb
    workflows/tech_eval_runtimes.ipynb
)


# Read the flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --inplace) INPLACE=1; shift ;;
        --quick) QUICK=1; shift ;;
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

failed=()  # notebooks that raised an error

for nb in "${NOTEBOOKS[@]}"; do
    echo "=== $nb"
    mkdir -p "$OUT_DIR/$(dirname "$nb")"  # create the matching subfolder for the copy
    # Run all cells, printing each cell's output as it finishes (--log-output).
    # --cwd runs the notebook from its own folder, so its relative file paths work.
    # papermill is not a project dependency, so uv adds it just for this.
    if ! uv run --group dev --with papermill papermill "$nb" "$OUT_DIR/$nb" \
        --cwd "$(dirname "$nb")" --log-output; then
        failed+=("$nb")
    fi
done

echo
if [[ ${#failed[@]} -eq 0 ]]; then
    echo "All notebooks passed."
else
    echo "Failed: ${failed[*]}"
    exit 1  # non-zero exit code signals failure (e.g. to CI)
fi
