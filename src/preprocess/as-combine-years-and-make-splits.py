import pandas as pd
from tqdm.auto import tqdm 
import os
import numpy as np

if __name__ == "__main__":
    dir_path = "/data/mourad/narratives/inflation/americanstories_sentences"
    
    dfs = []
    for file in os.listdir(dir_path):
        df = pd.read_csv(os.path.join(dir_path, file), sep="\t")
        df.text = df.text.str.replace("\n", " ")
        df['year'] = int(file.split(".")[0])
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.reset_index(drop=True)
    df = df[df.mention_flag == 1]
    df['month'] = df.date.apply(lambda x: eval(x)[0].split('-')[1])
    df = df[["id", "text", "month", "year"]]
    
    # n_per_year = 5
    
    # breakpoint()
    # eval_df = df.groupby(['year']).sample(n_per_year, random_state=n_per_year) # for 5/month random state 0
    # eval_df = eval_df.sample(frac=1, random_state=0)
    eval_df = df.copy()
    eval_df = eval_df.sample(frac=1, random_state=0)
    
    # df = df.drop(eval_df.index)
    

    eval_df = eval_df[["id", "text", "month", "year"]]
    # eval_df['(c)ause/(e)ffect'] = ""
    # eval_df['category'] = ""
    # eval_df['Not Applicable'] = ""
    # eval_df['Foreign'] = ""

    # eval_df['assigned'] = ["mh" if i % 3 == 0 else "qz" if i % 3 == 1 else "az" for i in range(len(eval_df))]

    agreement_data_len = 500
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


    # breakpoint()
    # new_df = new_df.sort_values(by='year')

    new_df.to_csv(f"/data/mourad/narratives/inflation/raw_to_label-all-500test.tsv", sep="\t", index=False)

    # for text in df.text.sample(50):
        # print(text)
        # print()
    # print(df.head())
    
