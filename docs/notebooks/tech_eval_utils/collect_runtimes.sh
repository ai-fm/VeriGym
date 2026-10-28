#!/usr/bin/env bash
# Collect the data for the runtime breakdown per abstraction size: a fixed environment, policy and budget, with
# an increasing number of state bins. Every (env, policy, budget, bins, seed) run writes its own file to ../output/tech_eval/runtimes
# (shared with collect_scaling.sh); runs whose file already exists are skipped, so an interrupted collection can
# be resumed by simply re-running this script.
#
# Note: do not run this in parallel with collect_scaling.sh, as concurrent runs distort the timings.
#
# Usage: ./collect_runtimes.sh          full collection (written to ../output/tech_eval/runtimes)
#        ./collect_runtimes.sh --quick  small smoke test (written to ./data/quick, always re-run)
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

ENV="MountainCarContinuous-v0"  # 2-dimensional observations, so n_states = bins^2
POLICIES=("ActiveLearning")
BUDGET=100000  # 100 iterations of 1000 steps
# bins per dimension for ~100, 200, 500, 1000, ..., 10^5 total abstract states (rounded sqrt)
BINS_STATES=(10 14 22 32 45 71 100 141 224 316)
BINS_ACTIONS=10
SEEDS=(0 1 2 3 4)
STEPS_PER_ITER=1000
EVAL_EPISODES=1000
OUT_DIR="../output/tech_eval/runtimes/data"

if [[ $QUICK -eq 1 ]]; then
    BUDGET=10000
    BINS_STATES=(10)
    SEEDS=(0)
    EVAL_EPISODES=100
    OUT_DIR="$OUT_DIR/quick"
fi

failed=0
for bins in "${BINS_STATES[@]}"; do
    for policy in "${POLICIES[@]}"; do
        for seed in "${SEEDS[@]}"; do
            out="${OUT_DIR}/${ENV}__${policy}__${BUDGET}__s${bins}a${BINS_ACTIONS}__seed${seed}.json"
            if [[ $QUICK -eq 0 && -f "$out" ]]; then
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
