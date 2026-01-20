#!/bin/bash

# Run inference on Federal Reserve data in parallel chunks
# Each chunk is 8,000 sentences (defined in predict_json.py FULL_DATASET_PATHS)
# Total sentences: ~76,122
# Number of chunks: ceil(76122 / 8000) = 10 chunks (0-9)

CHUNK_SIZE=8000
TOTAL_SENTENCES=76122
NUM_CHUNKS=$(( (TOTAL_SENTENCES + CHUNK_SIZE - 1) / CHUNK_SIZE ))

echo "Processing Fed data: $TOTAL_SENTENCES sentences in $NUM_CHUNKS chunks"

for i in $(seq 0 $((NUM_CHUNKS - 1))); do
    output_dir="/net/projects/chai-lab/mourad/narratives-data/model_json_preds/fed/full_fed/llama31_ft__600s_train-now_and_proquest_sample_${i}"

    if [ -d "$output_dir" ]; then
        echo "Chunk $i already processed, skipping..."
    else
        echo "Submitting chunk $i"
        sbatch run_predict_fed.sh $i
    fi
done

echo "All jobs submitted!"
