#!/bin/bash

# Run inference on processed-data-2010-2025_updated.csv.gz in parallel chunks
# Each chunk is 50,000 sentences (defined in predict_json.py FULL_DATASET_PATHS)

# First, determine how many chunks you have in your dataset
# You can adjust the range {0..N} based on your dataset size

for i in {1..10}; do
    # Check if output directory already exists (skip if already processed)
    output_dir="/net/projects/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/llama31_ft__600s_train-now_and_proquest_sample_"$i"_2010-2025_updated"

    if [ -d "$output_dir" ]; then
        echo "Chunk $i already processed, skipping..."
    else
        echo "Submitting chunk $i"
        sbatch run_predict_2010_2025_updated.sh $i
    fi
done

echo "All jobs submitted!"
