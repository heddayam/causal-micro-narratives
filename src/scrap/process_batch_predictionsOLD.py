from src.utils import utils
from datasets import concatenate_datasets, load_from_disk
import argparse
from glob import glob
from os import path
import json
from datasets import Dataset
import pandas as pd
import swifter
import re


# def extract_narrative_category(pred):
#     cats = []
#     if pred['contains-narrative']:
#         narrs = pred['inflation-narratives']['narratives'][0]
#         if isinstance(narrs, str):
#             narrs = re.sub(r'\s', '', narrs)
#             narrs = eval(narrs)
#         if isinstance(narrs, tuple) or isinstance(narrs, list):
#             cats = [list(narr.values())[0] for narr in narrs]
#         else:
#             cats = [list(narrs.values())[0]]
            
#     cats = [cat for cat in cats if cat in utils.causes+utils.effects]
#     if len(cats) > 0:
#         cats.append('contains_narrative')
#     return list(set(cats))


def main(ds, model, train_ds):
    # GOLD DATA
    # gold_train = utils.load_hf_dataset(path=f"/data/mourad/narratives/sft_data", dataset='proquest')['train'].to_pandas()
    gold_test = utils.load_hf_dataset(path=f"/data/mourad/narratives/sft_data", dataset='proquest')['test'].to_pandas()
    # gold_df = pd.concat([gold_train, gold_test])
    gold_df = gold_test
    gold_df['annotation'] = gold_df.apply(lambda x: json.loads(utils.reconstruct_training_input(x)), axis=1)
    gold_df['narrative'] = gold_df['annotation'].swifter.progress_bar(True).apply(utils.extract_narrative_category) #
    gold_df['contains'] = gold_df['narrative'].apply(lambda x: len(x) > 0).astype(int)
    x = gold_df.narrative.str.join('|').str.get_dummies().add_prefix('mean_')
    gold_df = gold_df[['text']]
    gold_df = pd.concat([gold_df, x], axis=1)
    # breakpoint()
    
    # MODEL PREDICTION DATA
    dir_base = f"/data/mourad/narratives/model_json_preds/{ds.lower()}"
    all_ds = []
    all_dfs = []
    # /data/mourad/narratives/model_json_preds/proquest/llama31_ft__600s_train-now_and_proquest_sample_2
    batchfiles = glob(path.join(dir_base, f"{model}_train-{train_ds}_sample_*"))
    # batchefiles = glob(f"/data/mourad/narratives/model_json_preds/{ds.lower()}/phi2_ft_NOW_filtered_sample_*")
    for sample, batchfile in enumerate(batchfiles):
        dataset = load_from_disk(batchfile)
        # ds = utils.load_hf_dataset(model=f"phi2_ft_{ds}_filtered_sample_{sample}")
        dataset = dataset.map(lambda e: {'prediction': json.loads(e['completion'])})
        data = dataset.to_pandas()
        data['lens'] = data['text'].apply(lambda x: len(x.split()))
        data = data[data['lens'] <= utils.PROQUEST_MAX_LEN]
        dataset = Dataset.from_pandas(data, preserve_index=False)
        all_ds.append(dataset)
        all_dfs.append(data)
        
    dataset = concatenate_datasets(all_ds)
    dataset_df = pd.concat(all_dfs)
    
    
    dataset_df['narrative'] = dataset_df['prediction'].swifter.progress_bar(True).apply(utils.extract_narrative_category) #
    dataset_df['contains'] = dataset_df['narrative'].apply(lambda x: len(x) > 0).astype(int)
    
    # breakpoint()
    # for city_df in dataset_df.groupby('city'):
    #     breakpoint()
    
    # x = dataset_df[['id']].join(dataset_df.narrative.str.join('|').str.get_dummies())
    x = dataset_df.narrative.str.join('|').str.get_dummies().add_prefix('mean_pred_')
    dataset_df = pd.concat([dataset_df, x], axis=1)
    
    dataset_df = dataset_df[~dataset_df.city.isin(['missoula, vermillion'])]
    
    dataset_df = dataset_df.drop(['file_id', 'lens', 'title', 'loc', 'year', 'month', 'state', 'id', 'completion', 'scope', 'prediction', 'narrative', 'contains', 'city'], axis=1)
    
    dataset_df = dataset_df.rename({'year_month': 'cand_id'}, axis=1)
    
    
    
    # dataset_df['post_sample_prob'] = (dataset_df['text'].isin(gold_df.text).astype(int).sum() / len(dataset_df))
    # cand_sample_prob = dataset_df.groupby('cand_id').post_sample_prob.sum() / dataset_df['text'].isin(gold_df.text).astype(int).sum()
    cand_sample_prob = dataset_df.groupby('cand_id').size()/len(dataset_df)
    dataset_df['cand_sample_prob'] = dataset_df['cand_id'].map(cand_sample_prob)
    
    # dataset_df['has_expert'] = dataset_df['text'].isin(gold_df.text).astype(int)
    # cand_sample_prob = dataset_df.groupby('year_month').has_expert.sum()
    # dataset_df['text_sample_prob'] = 1499 / len(dataset_df)
    # dataset_df['cand_sample_prob'] = dataset_df['year_month'].map(cand_sample_prob) / len(dataset_df)
    # breakpoint()
    dataset_df = dataset_df.merge(gold_df, on='text', how='left')

    
    dataset_df = dataset_df.drop(['text', 'region'], axis=1)
    dataset_df = dataset_df.groupby(['cand_id', 'cand_sample_prob']).mean().reset_index()
    # text_sample_prob = dataset_df.groupby('text').has_expert.mean()
    breakpoint()
    dataset_df.to_csv(f"/data/mourad/narratives/model_json_preds/{ds.lower()}/DSL_READY-test_gold_only-{model.replace('__', '_')}_train-{train_ds}_all.csv")#, sep='\t')    
    breakpoint()
    dataset.save_to_disk(f"/data/mourad/narratives/model_json_preds/{ds.lower()}/{model.replace('__', '_')}_train-{train_ds}_all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['NOW', 'PROQUEST'], help="dataset to combine batch preds for", required=True)
    parser.add_argument("--model", choices=['phi2', 'llama31_ft__600s'], default='llama31_ft__600s', help="model to combine batch preds for")
    parser.add_argument('--train_ds', choices=['now', 'proquest', 'now_and_proquest'], default='now_and_proquest', help="train dataset of model used")
    args = parser.parse_args()

    main(args.dataset, args.model, args.train_ds)