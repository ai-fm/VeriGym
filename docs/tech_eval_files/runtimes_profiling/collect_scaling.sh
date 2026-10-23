#!/usr/bin/env bash
# Collect runtime and policy-quality data for increasing exploration budgets, for all exploration policies.
# Every (env, policy, budget, seed) run writes its own file to ./data; runs whose file already exists are
# skipped, so an interrupted collection can be resumed by simply re-running this script.
#
# Usage: ./collect_scaling.sh          full collection
#        QUICK=1 ./collect_scaling.sh  small smoke test (written to ./data/quick)
set -u

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

ENVS=("MountainCarContinuous-v0")
POLICIES=("Random" "Q-Learning" "ActiveLearning" "Entropy")
# POLICIES=("Entropy")
BUDGETS=(1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 2000000 5000000)
SEEDS=(0 1 2 3 4)
STEPS_PER_ITER=1000
BINS_STATES=25
BINS_ACTIONS=10
EVAL_EPISODES=1000
OUT_DIR="data"

failed=0
for env in "${ENVS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        for policy in "${POLICIES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                out="${OUT_DIR}/${env}__${policy}__${budget}__s${BINS_STATES}a${BINS_ACTIONS}__seed${seed}.json"
                if [[ -f "$out" ]]; then
                    echo "Skipping existing $out"
                    continue
                fi
                if ! "$PYTHON" run_profiling.py \
                        --env "$env" --policy "$policy" --budget "$budget" --seed "$seed" \
                        --steps-per-iter "$STEPS_PER_ITER" --eval-episodes "$EVAL_EPISODES" \
                        --bins-states "$BINS_STATES" --bins-actions "$BINS_ACTIONS" \
                        --out-dir "$OUT_DIR"; then
                    echo "FAILED: $env $policy $budget seed $seed" >&2
                    failed=$((failed + 1))
                fi
            done
        done
    done
done

echo "Done, $failed failed run(s)."
[[ $failed -eq 0 ]]