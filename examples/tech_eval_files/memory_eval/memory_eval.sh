#!/bin/zsh

filename="$1"
constants="$2"
task="$3"

python tech_eval_files/memory_eval_runner.py "$task" "$filename" --constants "$constants"
