#!/bin/bash

# Arguments (you can parameterize these later if needed)
model="llava"
method="haltrapper"
eval_task="chair"
total_sampled_images=500
fixed="True"
max_new_tokens=2048
expt="expt"  # define if needed, else remove from filename pattern

# Directory for logs
log_dir="results/logs"
mkdir -p "$log_dir"

# Find highest existing iter for same config
# Matching: results/haltrapper-chair-expt-2-1024-*.yaml
pattern="${log_dir}/${model}-${method}-${eval_task}-${expt}-${total_sampled_images}-${max_new_tokens}-*.jsonl"

last_iter=$(ls $pattern 2>/dev/null \
    | sed -E 's/.*-([0-9]+)\.yaml/\1/' \
    | sort -n \
    | tail -1)

# If no previous runs, start at 1
if [[ -z "$last_iter" ]]; then
    iter=1
else
    iter=$((last_iter + 1))
fi

# Final log file
log_file="${log_dir}/${model}-${method}-${eval_task}-${expt}-${total_sampled_images}-${max_new_tokens}-${iter}.jsonl"

echo "Using iter = $iter"
echo "Saving log to: $log_file"

# Run HalTrapper decontext script
# python3 decontext.py \
#     --model "$model" \
#     --method "$method" \
#     --eval "$eval_task" \
#     --total-sampled-images "$total_sampled_images" \
#     --fixed "$fixed" \
#     --max_new_tokens "$max_new_tokens" \
#     --log-file-path "$log_file"

python3 main.py 

python3 playground/report.py 

echo "Report generated for iter $iter"