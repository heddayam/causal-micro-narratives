for i in {0..24}; do
    if [ $i -eq 28 ]; then
       continue
    fi
    if [ -d "/net/projects/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/llama31_ft__600s_train-now_and_proquest_sample_"$i"_2010-2025" ]; then
        echo "Directory exists"
    else
        sbatch run_predict.sh llama31 $i &
    fi
done