#!/bin/zsh

filename="$1"
constants="$2"
task="$3"

python "$(dirname "$0")/memory_eval_runner.py" "$task" "$filename" --constants "$constants"
