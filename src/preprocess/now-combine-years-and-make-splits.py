import pandas as pd
from tqdm.auto import tqdm 
import os
import numpy as np

if __name__ == "__main__":
    sentences_dir = '/data/mourad/narratives/inflation/sentences'

    n_per_month = 10

    years = range(2012, 2023)
    dfs = []
    for year in tqdm(years):
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            file_out = os.path.join(sentences_dir, str(year), f'us_now_{year}_{month}.tsv')
            try:
                df = pd.read_csv(file_out, sep='\t')
            except Exception as e:
                print(e)
                continue
            df = df[df.mention_flag == 1]
            df = df.drop(['mention_flag', 'unique_id'], axis=1)
            df['month'] = int(month)
            df['year'] = year
            # df = df.groupby(['id', 'unique_id', ]).agg(list)
            # df = df.drop('mention_flag', axis=1)
            # df.text = df.text.apply(lambda x: ' '.join(x))
            # df['date'] = f"{year}{month}01"

            df = df.reset_index(drop=True)
            dfs.append(df)
    
    df = pd.concat(dfs, axis=0)    
    breakpoint()
    
    # eval_df = df.sample(500, random_state=0)
    eval_df = df.groupby(['year', 'month']).sample(n_per_month, random_state=n_per_month) # for 5/month random state 0
    eval_df = eval_df.sample(frac=1, random_state=0)
    
    # df = df.drop(eval_df.index)
    
    # eval_df = eval_df[["id", "text"]]
    # eval_df['(c)ause/(e)ffect'] = ""
    # eval_df['category'] = ""
    # eval_df['Not Applicable'] = ""
    # eval_df['Foreign'] = ""

    # eval_df['assigned'] = ["mh" if i % 3 == 0 else "qz" if i % 3 == 1 else "az" for i in range(len(eval_df))]

    agreement_data_len = 201
    eval_df_length = len(eval_df) - agreement_data_len
    eval_df['assigned'] = ["all"] * agreement_data_len + ["mh"] * int(eval_df_length * 1/3) + ["qz"] * int(eval_df_length * 1/3) + ["az"] * int(eval_df_length * 1/3)

    mh_df = pd.concat([eval_df[eval_df.assigned == "mh"],eval_df[eval_df.assigned == "all"]], axis=0)
    qz_df = pd.concat([eval_df[eval_df.assigned == "qz"],eval_df[eval_df.assigned == "all"]], axis=0)
    az_df = pd.concat([eval_df[eval_df.assigned == "az"],eval_df[eval_df.assigned == "all"]], axis=0)
    # qz_df = eval_df[eval_df.assigned == "qz"] + eval_df[eval_df.assigned == "all"]
    # az_df = eval_df[eval_df.assigned == "az"] + eval_df[eval_df.assigned == "all"]

    new_df = pd.concat([mh_df, qz_df, az_df], axis=0)

    new_df['assigned'] =    ["mh"] * len(eval_df[eval_df.assigned == "mh"]) + ["mh*"] * agreement_data_len + \
                            ["qz"] * len(eval_df[eval_df.assigned == "qz"]) + ["qz*"] * agreement_data_len + \
                            ["az"] * len(eval_df[eval_df.assigned == "az"]) + ["az*"] * agreement_data_len

    breakpoint()

    # new_df.to_csv(f"/data/mourad/narratives/categories/eval/eval-{n_per_month}_per_month.tsv", sep="\t", index=False)
    
    # active
    # new_df.to_csv(f"../../data/NOW_for_labelstudio_v1.tsv", sep="\t", index=False)
    
   
    df.to_json('/data/mourad/narratives/inflation/all_filtered.jsonl.gz', orient='records', lines=True, compression='gzip')
