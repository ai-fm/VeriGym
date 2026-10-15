#!/usr/bin/env bash
# Collect the data for the runtime breakdown per abstraction size: a fixed environment, policy and budget, with
# an increasing number of state bins. Every (env, policy, budget, bins, seed) run writes its own file to ./data
# (shared with collect_scaling.sh); runs whose file already exists are skipped, so an interrupted collection can
# be resumed by simply re-running this script.
#
# Note: do not run this in parallel with collect_scaling.sh, as concurrent runs distort the timings.
#
# Usage: ./collect_runtimes.sh
set -u

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

ENV="MountainCarContinuous-v0"  # 2-dimensional observations, so n_states = bins^2
POLICIES=("ActiveLearning")
BUDGET=100000  # 100 iterations of 1000 steps
# bins per dimension for ~100, 200, 500, 1000, ..., 10^5 total abstract states (rounded sqrt)
BINS_STATES=(10 14 22 32 45 71 100 141 224 316)
BINS_ACTIONS=10
SEEDS=(0 1 2 3 4)
STEPS_PER_ITER=1000
EVAL_EPISODES=1000
OUT_DIR="data"

failed=0
for bins in "${BINS_STATES[@]}"; do
    for policy in "${POLICIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            out="${OUT_DIR}/${ENV}__${policy}__${BUDGET}__s${bins}a${BINS_ACTIONS}__seed${seed}.json"
            if [[ -f "$out" ]]; then
                echo "Skipping existing $out"
                continue
            fi
            if ! "$PYTHON" run_profiling.py \
                    --env "$ENV" --policy "$policy" --budget "$BUDGET" --seed "$seed" \
                    --bins-states "$bins" --bins-actions "$BINS_ACTIONS" \
                    --steps-per-iter "$STEPS_PER_ITER" --eval-episodes "$EVAL_EPISODES" \
                    --out-dir "$OUT_DIR"; then
                echo "FAILED: $ENV $policy $BUDGET bins $bins seed $seed" >&2
                failed=$((failed + 1))
            fi
        done
    done
done

echo "Done, $failed failed run(s)."
[[ $failed -eq 0 ]]
