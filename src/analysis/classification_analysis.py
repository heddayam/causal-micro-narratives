# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
import numpy as np
from src.utils import utils
import argparse
import swifter
from tqdm import tqdm
# import old_prompts
import seaborn as sns
# import pickle
from collections import defaultdict
import json
# from thefuzz import fuzz

time = ['past', 'present', 'future', 'general']

categories = {
    "contains-narrative": [
        True, False
    ],
    "foreign": [
        True, False
    ],
    "inflation-time": time,
    "counter-narrative": [
        True, False
    ],
    "cause": [
        'demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause'
        ],
    "effect": [
        'cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect'
    ]
    # "narrative-time": time,
}

def clean_completion(completion):
    completion = completion.replace("```json\n", "")
    completion = completion.replace("\n```", "")
    return completion

def count_labels(input):
    gen_counts = defaultdict()
    narrative_meta_counts = defaultdict()
    narrative_counts = defaultdict(list)
    for cat, labels in categories.items():
        if cat in ['inflation-time', 'counter-narrative']:
            if input['contains-narrative']:
                narrative_meta_counts[cat] = input['inflation-narratives'][cat]
        elif cat in ['cause', 'effect']:
            if input['contains-narrative']:
                for narr in input['inflation-narratives']['narratives']:
                    if cat in narr:
                        # breakpoint()
                        if narr[cat] not in utils.categories[f"{cat}_category"]: # ignore isntances where the a cause label used for effect or vice versa
                            continue
                        narrative_counts[cat].append(narr[cat])
                        narrative_counts[f"{cat}-time"].append(narr['time'])
        else:
            gen_counts[cat] = input[cat]
    return gen_counts, narrative_meta_counts, narrative_counts

def adjust_value_name(row):
    if row['Property'] == "contains-narrative":
        if row.label == "True":
            return "Narrative"
        else:
            return "No Narrative"
    elif row['Property'] == "foreign":
        if row.label == "True":
            return "Foreign"
        else:
            return "Domestic"
        
    
def combine_region_and_classification_data(df, labeled):
    if labeled:
        source_df = pd.read_json('../../data/labeled_with_location.jsonl.gz', orient='records', lines=True, compression='gzip')
        breakpoint()
    else:
        source_df = pd.read_json('/data/mourad/narratives/inflation/all_filtered_with_location.jsonl.gz', orient='records', lines=True, compression='gzip')
    source_df = source_df.dropna()
    region_map = utils.region_mapping
    source_df['region'] = source_df.state.apply(lambda x: region_map.get(x.lower(), None))
    source_df = source_df[['text', 'region', 'city', 'state', 'scope']].drop_duplicates()
    
    df = df.merge(source_df, on="text", how="left") # used to be inner
    return df
    # df = df.dropna()

def bar_plot(df, data_type, split):
    plt.figure(figsize=(7, 6))
    df = df.dropna()
    df.label = df.label.astype(str)
    if data_type == "Metadata":
        df.label = df.apply(adjust_value_name, axis=1)
        # ax = sns.histplot(data=df, x="Property", stat='count', discrete=True)
    ax = sns.histplot(data=df, x="label", hue="Property", stat='count', discrete=True)
    ax.yaxis.grid(True)  # Add horizontal grid lines
    plt.title(f"{data_type}")
    plt.xticks(rotation=45)  # Rotate x-axis labels by 45 degrees
    plt.tight_layout()
    plt.savefig(f"../../data/plots/now_all_preds/{split}_{data_type}_barplot.png", dpi=300)
    plt.clf()
    ax.clear()
    
def kde_plot(df, type, ds, inflation_time, split, location=False):
    
    if type in ['Cause', 'Effect']:        
        df['weight'] = df.label.notna().astype(int)
        hue_order = utils.categories[f"{type.lower()}_category"]
    elif type == 'Cause_Effect':
        df['weight'] = df.label.notna().astype(int)
        hue_order = ['cause', 'effect']
    else:
        df['weight'] = 1
        hue_order = None
        
    if split is None:
        if len(df) > len(ds):
            df['month'] = ds['month'] + ds['month']
            df['year'] = ds['year'] + ds['year']
            df['inflation-time'] = inflation_time + inflation_time
        else:    
            df['month'] = ds['month']
            df['year'] = ds['year']
            df['inflation-time'] = inflation_time
        
        df['month_year'] = df.month.astype(str) + '-' + df.year.astype(str)
        df['time'] = df.groupby(['year', 'month'], sort=True).ngroup()
    # if type == 'Cause':
    #     df = df[df['inflation-time'].isin(["present", "future"])]
    # elif type == 'Effect':
    #     df = df[df.narrative_time.isin(["present", "future"])]

    if location:
        # breakpoint()
        df = combine_region_and_classification_data(df, labeled=split is not None)
    breakpoint()
    if split is not None:
        df.to_csv(f"../../data/{split}_{type.lower()}_and_locations.tsv", sep='\t', index=False)
    else:
        df.to_csv(f"../../data/{type.lower()}_and_locations.tsv", sep='\t', index=False)

    return
        # region_plot(df, args.inflation_type)
    
    x_label = "time"
    # breakpoint()
    x = df.time
    xlabels = []
    current_yr = ''
    for m in df.month_year.unique().tolist():
        year = m.split('-')[1]
        if current_yr != year:
            xlabels.append(str(year))
            current_yr = year
        else:
            xlabels.append('')
            
            
    if location:
        df = df.drop('weight', axis=1)
        for word in tqdm(df.label.dropna().unique()):
            plt.figure(figsize=(8, 6))
            indiv_df = df.copy() #[df.label == word]
            indiv_df['match'] = indiv_df.label.apply(lambda x: 1 if x == word else 0)
            # indiv_df['match'] = indiv_df.label == word
            # breakpoint()
            weight = indiv_df.groupby(["time", "region"]).match.agg(np.mean).reset_index()[['time', 'region', 'match']]
            weight = weight.rename({'match':'weight'}, axis=1)
            indiv_df = df[df.label == word].merge(weight, on=['time', 'region'], how='inner')
            # indiv_df['match'] = indiv_df.groupby(["time", "region"]).agg(np.mean)['match']

            # breakpoint()
            ax = sns.kdeplot(data=indiv_df, x=x_label, hue='region', multiple="fill", 
                             bw_adjust=1, clip=(0,len(xlabels)), weights="weight",
                             hue_order=set(utils.region_mapping.values())) #[df.label == word]
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            plt.xticks(list(set(x)), xlabels)
            plt.xlabel("Year")
            plt.ylabel(f"Sentences about {word}")
            plt.savefig(f'../../data/plots/analysis/{type.lower()}/kdefill_{type}_location_alltime_{word.replace("/", "_").lower()}.png', dpi=300, bbox_inches='tight')
            # plt.savefig(f'/data/mourad/narratives/categories/{type}/region/{word.replace("/", "_").lower()}.png', dpi=300)
            plt.clf()
            # breakpoint()
            # sns.histplot(data=df[df.label == word], x='region', stat='count', discrete=True)
            # # plt.xticks(list(set(x)), xlabels)
            # plt.xlabel("Region")
            # plt.ylabel(f"# of Sentences About {word}")
            # plt.savefig(f'/data/mourad/narratives/categories/{type}/region/counts/{word.replace("/", "_").lower()}.png', dpi=300)
            
            # plt.clf()
    else:
        plt.figure(figsize=(8, 6))
        
        ax = sns.kdeplot(data=df, x=x_label, hue='label', multiple="fill",
                        bw_adjust=1, clip=(0,len(xlabels)), 
                        weights="weight",
                        hue_order=hue_order
                        ) #[df.label == word]
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
        
        plt.xticks(list(set(x)), xlabels)
        plt.xlabel("Year")
        plt.ylabel(f"{type} Narratives")
        # plt.savefig(f'../../data/plots/analysis/kdefill_{type}_presentfuture.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'../../data/plots/analysis/kdefill_{type}_alltime.png', dpi=300, bbox_inches='tight')
        plt.clf()
    
    
    

def main(split, model, location, barplot=False, kdeplot=True):
    if model:
        ds = utils.load_hf_dataset(model=model)
        ds = ds.map(lambda x: {"input": clean_completion(x['completion'])}, batch_size=False, load_from_cache_file=False)
        # split = model
    else:
        ds = utils.load_hf_dataset(split=split)
        ds = ds.map(lambda x: {"input": utils.reconstruct_training_input(x)}, batch_size=False, load_from_cache_file=False)
    # if split == 'test':
    #     pred_data = utils.load_hf_dataset(model=model)
    gen = []
    narrative_meta = []
    narrative = []
    for instance in tqdm(ds):
        
        # print(instance['text'])
        
        json_instance = json.loads(instance['input'])
        # print(json_instance)
        # breakpoint()
        gen_counts, narrative_meta_counts, narrative_counts = count_labels(json_instance)
        gen.append(gen_counts)
        narrative_meta.append(narrative_meta_counts)
        narrative.append(narrative_counts)

    gen_df = pd.DataFrame(gen)

    narrative_meta_df = pd.DataFrame(narrative_meta)
    narrative_df = pd.DataFrame(narrative)
    
    narrative_df['id'] = ds['id']
    narrative_df['text'] = ds['text']
    
    
    cause_df = narrative_df[["id", "text","cause"]].explode(['cause'])
    cause_time_df = narrative_df[["cause-time"]].explode(['cause-time'])
    effect_df = narrative_df[["id", "text", "effect"]].explode(['effect'])
    effect_time_df = narrative_df[["effect-time"]].explode(['effect-time'])
    gen_df = pd.melt(gen_df, var_name="Property", value_name="label")
    narrative_meta_df = pd.melt(narrative_meta_df, var_name="Property", value_name="label")
    

    cause_df = pd.melt(cause_df, id_vars=["id", "text"], var_name="Property", value_name="label")
    effect_df = pd.melt(effect_df, id_vars=["id", "text"], var_name="Property", value_name="label")
    cause_time_df = pd.melt(cause_time_df, var_name="Property", value_name="label")
    effect_time_df = pd.melt(effect_time_df, var_name="Property", value_name="label")

 
    cause_df['narrative_time'] = cause_time_df.label
    effect_df['narrative_time'] = effect_time_df.label
    
    breakpoint()
    
    if barplot == True:
        bar_plot(gen_df, "Metadata", split)
        bar_plot(narrative_meta_df, "Narrative Metadata", split)
        bar_plot(cause_df, "Cause", split)
        bar_plot(cause_time_df, "Cause Time", split)
        bar_plot(effect_df, "Effect", split)
        bar_plot(effect_time_df, "Effect Time", split)
        
    if kdeplot == True:
        inflation_time = narrative_meta_df[narrative_meta_df.Property == 'inflation-time'].label

        # cause/effect categories
        kde_plot(effect_df, "Effect", ds, inflation_time, split, location)
        kde_plot(cause_df, "Cause", ds, inflation_time, split, location)
        breakpoint()
        
        # cause vs effect
        cause_effect = pd.concat([cause_df, effect_df], axis=0).reset_index(drop=True)
        cause_effect['label'] = cause_effect.apply(lambda x: x.Property if x.label is not np.nan else np.nan, axis=1)
        kde_plot(cause_effect, "Cause_Effect", ds, inflation_time, split)
        # narrative vs no narrative
        kde_plot(gen_df[gen_df.Property == 'contains-narrative'], "Contains_Narrative", ds, inflation_time, split)
        

        
    
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["claude", "gpt35", "gpt4t", "gpt4", "phi2_ft", "phi2_ft_NOW_filtered_all"])
    # parser.add_argument('--category', choices=['narrative', 'time', 'metadata'], required=True)
    parser.add_argument('--split', choices=['test', 'train'], required=False)
    parser.add_argument('--location', action="store_true")
    # parser.add_argument('--evaluate', action="store_true")
    # parser.add_argument('--combine_preds', action='store_true', help="if generated preds across years for sample eval")
    args = parser.parse_args()

    main(args.split, args.model, args.location)
# def clean_label(label):
#     if label in categories:
#         return label
    
#     pred_cats = []
#     for cat in categories:
#         score = fuzz.partial_ratio(label, cat)
#         if score > 80:
#             # breakpoint()
#             pred_cats.append(cat)
#             # return cat
#     return pred_cats

# def plot(df, type):
#     x_label = "time"
#     x = df.time
#     xlabels = []
#     current_yr = ''
#     for m in df.month_year.unique().tolist():
#         year = m.split('-')[1]
#         if current_yr != year:
#             xlabels.append(str(year))
#             current_yr = year
#         else:
#             xlabels.append('')
#     for word in df.label.dropna().unique():
#         plt.figure(figsize=(8, 6))
#         indiv_df = df.copy() #[df.label == word]
#         indiv_df['match'] = indiv_df.label.apply(lambda x: 1 if x == word else 0)
#         # indiv_df['match'] = indiv_df.label == word
#         indiv_df = indiv_df.groupby("time").agg(np.mean)*100

#         # breakpoint()
#         sns.barplot(data=indiv_df, x=x_label, y="match", width=1, edgecolor='black')
#         # sns.histplot(data=df[df.word == word], x=x_label, stat='probability', discrete=True)
#         plt.xticks(list(set(x)), xlabels)
#         plt.xlabel("Year")
#         plt.ylabel(f"% of sentences about {word}")
#         plt.savefig(f'/data/mourad/narratives/categories/{type}/{word.replace("/", "_").lower()}.png', dpi=300)
#         plt.clf()




    
   
#     return df

# def region_plot(df, type):
#     x_label = "time"
#     x = df.time
#     xlabels = []
#     current_yr = ''
#     for m in df.month_year.unique().tolist():
#         year = m.split('-')[1]
#         if current_yr != year:
#             xlabels.append(str(year))
#             current_yr = year
#         else:
#             xlabels.append('')
#     for word in tqdm(df.label.dropna().unique()):
#         plt.figure(figsize=(8, 6))
#         indiv_df = df.copy() #[df.label == word]
#         indiv_df['match'] = indiv_df.label.apply(lambda x: 1 if x == word else 0)
#         # indiv_df['match'] = indiv_df.label == word

#         weight = indiv_df.groupby(["time", "region"]).agg(np.mean).reset_index()[['time', 'region', 'match']]
#         weight = weight.rename({'match':'weight'}, axis=1)
#         indiv_df = df[df.label == word].merge(weight, on=['time', 'region'], how='inner')
#         # indiv_df['match'] = indiv_df.groupby(["time", "region"]).agg(np.mean)['match']

#         # breakpoint()
#         sns.kdeplot(data=indiv_df, x=x_label, hue='region', multiple="fill", bw_adjust=1, clip=(0,len(xlabels)), weights="weight") #[df.label == word]
#         plt.xticks(list(set(x)), xlabels)
#         plt.xlabel("Year")
#         plt.ylabel(f"Sentences about {word}")
#         plt.savefig(f'/data/mourad/narratives/categories/{type}/region/{word.replace("/", "_").lower()}.png', dpi=300)
#         plt.clf()
#         # breakpoint()
#         sns.histplot(data=df[df.label == word], x='region', stat='count', discrete=True)
#         # plt.xticks(list(set(x)), xlabels)
#         plt.xlabel("Region")
#         plt.ylabel(f"# of Sentences About {word}")
#         plt.savefig(f'/data/mourad/narratives/categories/{type}/region/counts/{word.replace("/", "_").lower()}.png', dpi=300)
#         plt.clf()

# def make_eval_csv(df, model, type):
#     df = df.groupby('label').sample(20, random_state=0)
#     df = df[["id", "text"]]
#     df.to_csv(f"/data/mourad/narratives/categories/eval/{type}.tsv", sep="\t", index=False)



        
    # data_root = Path(f"/data/mourad/narratives/inflation/{args.model}/{args.inflation_type}")
    # dfs = []
    # for file in tqdm(os.listdir(data_root)):
    #     df = pd.read_parquet(data_root / file)
    #     # breakpoint()
    #     df['label'] = df.parse.apply(clean_label)
    #     df['match'] = ~df.label.isna()
    #     df.match = df.match.astype(int)
    #     # df = df.dropna()
    #     dfs.append(df[['id', 'text', 'match', 'month', 'year', 'label']])


    # df = pd.concat(dfs, axis=0)
    # # df = df.sort_values(by=['year', 'month'])
    # df['time'] = df.groupby(['year', 'month']).ngroup()
    # # df = pd.DataFrame(keyword_counts, columns=vocabulary.keys())
    # df['month_year'] = df.month.astype(str) + '-' + df.year.astype(str)
    # df = df.sort_values(by='time')

    # if args.evaluate:
    #     make_eval_csv(df, args.model, args.inflation_type)
    # elif args.location:
    #     df = combine_region_and_classification_data(df)
    #     region_plot(df, args.inflation_type)
    # else:
    #     plot(df, args.inflation_type)
