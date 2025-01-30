import pandas as pd
import os
from glob import glob
import utils.utils as utils
import swifter

def preprocess(month, year):
    files = glob(f'/data/now/extracted/{year}/text/*{year[-2:]}*{month}-*.txt')
    # src_fname = '/data/now/extracted/2021/text/21-09-us1.txt'
    # src_fname = f'/data/now/extracted/{year}/text/{year[-2:]}-{month}-us1.txt'
    dfs = []
    for src_fname in files:
        df = pd.read_table(src_fname, sep="\t", encoding= "ISO-8859-1", header=None, on_bad_lines="warn")

        # df = df.head(100)

        df.columns = ['text']
        # text = open('/data/now/extracted/2021/text/21-09-us1.txt').readlines()
        # df = pd.DataFrame({'text': text})

        df['text'] = utils.filter_now(df['text'])

        res = df.text.swifter.apply(utils.extract_id)
        id, text = zip(*res)
        df.text = text
        df = df.sample(100, random_state=0)
        df = df.reset_index(drop=True)
        dfs.append(df)
    return pd.concat(dfs, axis=0)

if __name__ == '__main__':
    dir_to_save = '/data/mourad/narratives/sentences'
    years = range(2012, 2023)
    dfs = []
    for year in years:
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            try:
                df = preprocess(month, str(year))
                df['date'] = f"{year}{month}01"
                dfs.append(df)
                # out_dir = os.path.join(dir_to_save, str(year))
                # if not os.path.exists(out_dir):
                #     os.makedirs(out_dir)
                # df.to_csv(os.path.join(out_dir, f'us_now_{year}_{month}.tsv'), sep='\t', index=False)
            except:
                print('error', year, month)
                continue
            # breakpoint()
    df = pd.concat(dfs, axis=0)
    df = df.reset_index(drop=True)
    df.to_json('/data/mourad/narratives/idea_relations_background.jsonl.gz', orient='records', lines=True, compression='gzip')
    breakpoint()
