#!/bin/bash

# Run inference on processed-data-2010-2025_updated.csv.gz in parallel chunks
# Each chunk is 8,000 sentences (defined in predict_json.py FULL_DATASET_PATHS)

# First, determine how many chunks you have in your dataset
# You can adjust the range {0..N} based on your dataset size

# Completed chunks from previous runs under the old 20k split (skip these ranges)
completed_old_indices=(0 3 7 8 9 10 15 16 17)
old_chunk_size=20000
new_chunk_size=8000

# With 8k chunks, the total count is higher than the old 20k split.
# Update the upper bound if needed after checking dataset size.
for i in {0..44}; do
    # Check if output directory already exists (skip if already processed)
    start=$((i * new_chunk_size))
    end=$(((i + 1) * new_chunk_size - 1))
    skip=false

    for old_idx in "${completed_old_indices[@]}"; do
        old_start=$((old_idx * old_chunk_size))
        old_end=$(((old_idx + 1) * old_chunk_size - 1))

        if (( end < old_start || start > old_end )); then
            continue
        fi

        if (( start >= old_start && end <= old_end )); then
            skip=true
        elif (( start < old_start && end >= old_start )); then
            end=$((old_start - 1))
        elif (( start <= old_end && end > old_end )); then
            start=$((old_end + 1))
        fi
        break
    done

    if $skip; then
        echo "Chunk $i fully covered by completed old range, skipping..."
        continue
    fi

    if (( start > end )); then
        echo "Chunk $i has no remaining range after overlap check, skipping..."
        continue
    fi

    if (( start == i * new_chunk_size && end == ((i + 1) * new_chunk_size - 1) )); then
        output_dir="/net/projects2/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/llama31_ft__600s_train-now_and_proquest_sample_"$i"_2010-2025_updated"
        if [ -d "$output_dir" ]; then
            echo "Chunk $i already processed, skipping..."
        else
            echo "Submitting chunk $i"
            sbatch run_predict_2010_2025_updated.sh $i
        fi
    else
        output_dir="/net/projects2/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/llama31_ft__600s_train-now_and_proquest_range_"$start"_"$end"_2010-2025_updated"
        if [ -d "$output_dir" ]; then
            echo "Range $start-$end already processed, skipping..."
        else
            echo "Submitting range $start-$end (chunk $i)"
            sbatch run_predict_2010_2025_updated.sh $i $start $end
        fi
    fi
done

echo "All jobs submitted!"
