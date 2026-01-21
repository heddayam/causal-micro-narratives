#!/bin/bash

# Re-run chunks 28, 29, 30, 31 (accidentally deleted)
# Each chunk is 8,000 sentences

for i in 28 29 30 31; do
    echo "Submitting chunk $i"
    sbatch run_predict_2010_2025_updated.sh $i
done

echo "All jobs submitted!"
