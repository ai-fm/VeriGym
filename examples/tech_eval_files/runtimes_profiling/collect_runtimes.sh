#!/usr/bin/env bash
# Collect the data for the runtime breakdown per environment: every exploration policy on every environment
# with a fixed budget. Every (env, policy, budget, seed) run writes its own file to ./data (shared with
# collect_scaling.sh); runs whose file already exists are skipped, so an interrupted collection can be resumed
# by simply re-running this script.
#
# Note: do not run this in parallel with collect_scaling.sh, as concurrent runs distort the timings.
#
# Usage: ./collect_runtimes.sh
set -u

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"

# ENVS=("MountainCarContinuous-v0" "Pendulum-v1" "Acrobot-v1")
ENVS=("Acrobot-v1")
POLICIES=("Entropy")
BUDGETS=(100000)  # 100 iterations of 1000 steps
SEEDS=(0 1 2)
STEPS_PER_ITER=1000
EVAL_EPISODES=20
OUT_DIR="data"

failed=0
for env in "${ENVS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        for policy in "${POLICIES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                out="${OUT_DIR}/${env}__${policy}__${budget}__seed${seed}.json"
                if [[ -f "$out" ]]; then
                    echo "Skipping existing $out"
                    continue
                fi
                if ! "$PYTHON" run_profiling.py \
                        --env "$env" --policy "$policy" --budget "$budget" --seed "$seed" \
                        --steps-per-iter "$STEPS_PER_ITER" --eval-episodes "$EVAL_EPISODES" \
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
