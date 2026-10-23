#!/usr/bin/env bash
# Collect runtime and policy-quality data for increasing exploration budgets, for all exploration policies.
# Every (env, policy, budget, seed) run writes its own file to ./data; runs whose file already exists are
# skipped, so an interrupted collection can be resumed by simply re-running this script.
#
# Usage: ./collect_scaling.sh          full collection (written to ./data)
#        ./collect_scaling.sh --quick  small smoke test (written to ./data/quick, always re-run)
set -u

QUICK=0
case "${1:-}" in
    "" | --full) ;;
    --quick) QUICK=1 ;;
    -h | --help) sed -n '2,/^set -u/{/^set -u/d;s/^# \{0,1\}//;p}' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1 (expected --quick or --full)" >&2; exit 2 ;;
esac

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
OUT_DIR="../output/tech_eval/runtimes"

if [[ $QUICK -eq 1 ]]; then
    POLICIES=("Entropy")
    BUDGETS=(1000 10000)
    SEEDS=(0)
    EVAL_EPISODES=100
    OUT_DIR="data/quick"
fi

failed=0
for env in "${ENVS[@]}"; do
    for budget in "${BUDGETS[@]}"; do
        for policy in "${POLICIES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                out="${OUT_DIR}/${env}__${policy}__${budget}__s${BINS_STATES}a${BINS_ACTIONS}__seed${seed}.json"
                if [[ $QUICK -eq 0 && -f "$out" ]]; then
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