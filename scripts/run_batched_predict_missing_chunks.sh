#!/bin/bash

# Run missing 8k chunks for proquest 2010-2025 updated dataset
# These chunks were identified as missing from coverage analysis:
#   - chunk 3:  rows 24,000-31,999
#   - chunk 10: rows 80,000-87,999
#   - chunk 15: rows 120,000-127,999
#   - chunk 16: rows 128,000-135,999

for i in 3 10 15 16; do
    echo "Submitting chunk $i"
    sbatch run_predict_2010_2025_updated.sh $i
done

echo "All missing chunks submitted!"
