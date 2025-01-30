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
from tqdm.auto import tqdm

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
    # breakpoint()
    
    # MODEL PREDICTION DATA
    full_dfs = []
    for dataset_name in ['proquest', 'now']: #'proquest', 
        print(f"Processing {dataset_name} data")
        dir_base = f"/data/mourad/narratives/model_json_preds/full_{dataset_name}"
        all_ds = []
        all_dfs = []
        # /data/mourad/narratives/model_json_preds/proquest/llama31_ft__600s_train-now_and_proquest_sample_2
        batchfiles = glob(path.join(dir_base, f"{model}_train-{train_ds}_sample_*"))
        # breakpoint()
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
        dataset_df.source = dataset_name
        if dataset_name == 'now':
            dataset_df = dataset_df.rename({'id': 'file_id'}, axis=1)
        if dataset_name == 'proquest':
            dataset_df = dataset_df.drop(['year_month', 'title', 'loc'], axis=1)
        dataset_df = dataset_df.drop(['lens'], axis=1)

        # breakpoint()
        full_dfs.append(dataset_df)
            
    dataset_df = pd.concat(full_dfs)
    
    dataset_df['narrative'] = dataset_df['prediction'].apply(utils.extract_narrative_category) #.swifter.progress_bar(True)
    dataset_df = dataset_df[~dataset_df.narrative.isna()]
    dataset_df['contains'] = dataset_df['narrative'].apply(lambda x: len(x) > 0).astype(int)
    dataset_df = dataset_df[~dataset_df.city.isin(['missoula, vermillion'])]
    
    gold_df = gold_df.merge(dataset_df[['text', 'city']], on='text', how='left')
    
    dataset_df = dataset_df.drop('__index_level_0__', axis=1)
    
    # breakpoint()

    dataset_df.to_csv(f"/data/mourad/narratives/regression_data/all_news_data_llama_preds.csv")
    
    dataset_df = dataset_df.explode('narrative')
    dataset_df = dataset_df[dataset_df.narrative != 'contains_narrative']
    dataset_df.narrative = dataset_df.narrative.fillna('none')
    dataset_df = dataset_df.drop(['file_id', 'contains', 'completion', 'prediction'], axis=1)
    dataset_df = dataset_df.drop_duplicates()
    meta_cols = dataset_df.columns.drop('narrative').tolist()
    dataset_df = pd.get_dummies(dataset_df, prefix="", prefix_sep="", columns=['narrative'], dtype=int) 
    # dataset_df = dataset_df.drop('narrative', axis=1)
    # dataset_df = dataset_df.drop_duplicates()
    # result = dataset_df.groupby(meta_cols).agg(
    #     {col: 'sum' for col in meta_cols}  # Sum for all specified columns
    # ).assign(n_=lambda x: x.sum(axis=1))  # Add Group_Size column as the sum of rows
    # result = result.reset_index(inplace=True)
    # breakpoint()
    dataset_df = dataset_df.groupby(meta_cols, sort=False, dropna=False).sum().reset_index()
    breakpoint()
    dataset_df.drop('text', axis=1).to_csv(f"/data/mourad/narratives/regression_data/all_news_data_llama_preds_for_alex.csv")
    breakpoint()
    
    city_dfs = []
    gold_city_dfs = []
    for city_name, city_df in tqdm(dataset_df.groupby('city')):
        # breakpoint()
        city_name = city_name.replace(' ', '_')
    
        x = city_df.narrative.str.join('|').str.get_dummies().add_prefix(f'mean_pred_{city_name}_')
        city_df = pd.concat([city_df, x], axis=1)
        
        city_df = city_df.drop(['file_id', 'lens', 'title', 'loc', 'year', 'month', 'state', 'completion', 'scope', 'prediction', 'narrative', 'contains',  'city'], axis=1)
        if 'id' in city_df.columns:
            city_df = city_df.drop('id', axis=1)

        city_df.to_csv(f"/data/mourad/narratives/regression_data/citylevel/{city_name}.csv")
    
        city_dfs.append(city_df)
        
        
        gold_city_df = gold_df[gold_df.city == city_name]
        x = gold_city_df.narrative.str.join('|').str.get_dummies().add_prefix(f'mean_{city_name}_')   
        gold_city_df = gold_city_df[['text']]
        gold_city_df = pd.concat([gold_city_df, x], axis=1)
        
        gold_city_dfs.append(gold_city_df)
        

     
    breakpoint()
    dataset_df = pd.concat(city_dfs)
    dataset_df = dataset_df.rename({'year_month': 'cand_id'}, axis=1)
    gold_df = pd.concat(gold_city_dfs)
    
    breakpoint() 
    
    
    dataset_df['post_sample_prob'] = (dataset_df['text'].isin(gold_df.text).astype(int).sum() / len(dataset_df))
    # cand_sample_prob = dataset_df.groupby('cand_id').agg(('cand_sample_prob' , lambda x: 1 - (1 - x['post_sample_prob']).prod())).reset_index()
    # cand_sample_prob = dataset_df.groupby('cand_id').apply(lambda x: 1 - (1 - x['post_sample_prob']).prod()).reset_index()
    cand_sample_prob = dataset_df.groupby('cand_id').post_sample_prob.sum() / dataset_df['text'].isin(gold_df.text).astype(int).sum()
    # dataset_df['cand_sample_prob'] = dataset_df.groupby('cand_id').post_sample_prob.sum() / dataset_df['text'].isin(gold_df.text).astype(int).sum()
    dataset_df['cand_sample_prob'] = dataset_df['cand_id'].map(cand_sample_prob)
    # breakpoint()
    # print(cand_sample_prob.head())
    
    # dataset_df['labeled'] = dataset_df['text'].isin(gold_df.text).astype(int)
    # cand_sample_prob = dataset_df.groupby('year_month').labeled.sum()
    # dataset_df['text_sample_prob'] = 1499 / len(dataset_df)
    # dataset_df['cand_sample_prob'] = dataset_df['year_month'].map(cand_sample_prob) / len(dataset_df)
    # breakpoint()
    dataset_df = dataset_df.merge(gold_df, on='text', how='left')

    # dataset_df['labeled'] = dataset_df['text'].isin(gold_df.text).astype(int)
    dataset_df = dataset_df.drop(['text', 'region'], axis=1)
    dataset_df = dataset_df.groupby(['cand_id', 'cand_sample_prob']).mean().reset_index()
    
    # text_sample_prob = dataset_df.groupby('text').labeled.mean()
   
    breakpoint()
    dataset_df.to_csv(f"/data/mourad/narratives/model_json_preds/{ds.lower()}/DSL_READY-test_gold_only-{model.replace('__', '_')}_train-{train_ds}_all_citylevel.csv")#, sep='\t')    
    # breakpoint()
    # dataset.save_to_disk(f"/data/mourad/narratives/model_json_preds/{ds.lower()}/{model.replace('__', '_')}_train-{train_ds}_all")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['NOW', 'PROQUEST', 'full_proquest'], help="dataset to combine batch preds for", required=True)
    parser.add_argument("--model", choices=['phi2', 'llama31_ft__600s'], default='llama31_ft__600s', help="model to combine batch preds for")
    parser.add_argument('--train_ds', choices=['now', 'proquest', 'now_and_proquest'], default='now_and_proquest', help="train dataset of model used")
    args = parser.parse_args()

    main(args.dataset, args.model, args.train_ds)