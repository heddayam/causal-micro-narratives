import pandas as pd
import swifter
from glob import glob
import spacy
import re
import os
from src.utils import utils
import argparse
from collections import defaultdict

def preprocess(month, year, sentence_only=False, article_only=False):
    files = glob(f'/data/now/extracted/{year}/text/*{year[-2:]}*{month}-*.txt')
    # src_fname = '/data/now/extracted/2021/text/21-09-us1.txt'
    # src_fname = f'/data/now/extracted/{year}/text/{year[-2:]}-{month}-us1.txt'
    dfs = []
    for src_fname in files:
        df = pd.read_table(src_fname, sep="\t", encoding= "ISO-8859-1", header=None, on_bad_lines="warn")
        # df = df.head(100)
        if article_only:
            return df

        df.columns = ['text']
        # text = open('/data/now/extracted/2021/text/21-09-us1.txt').readlines()
        # df = pd.DataFrame({'text': text})
      
        df['text'] = utils.filter_now(df['text'])

        res = df.text.swifter.apply(utils.extract_id)
        id, text = zip(*res)
        df.text = text
        df['id'] = id

        def get_sentences(text):
            return_sents = []
            if len(text) > 1000000:
                for i in range(len(text) // 1000000):
                    return_sents += get_sentences(text[i:1000000*(i+1)])
                return return_sents
            else:
                doc = nlp(text)
                return [sent.text.strip() if ('@' not in sent.text) and (sent.text.strip() != '') else None for sent in doc.sents]
        df.text = df.text.apply(get_sentences)
        df = df.explode('text').reset_index(drop=True)
        df = df.dropna()
        if sentence_only:
            return df

        df = df.groupby('id').apply(get_surrounding_rows)
        if df is not None:
            df = df.reset_index(drop=True)
            dfs.append(df)
    return pd.concat(dfs, axis=0)


def get_surrounding_rows(df, pre1920=False):
    # if not isinstance(df, pd.Series):
    df = df.reset_index(drop=True)
    mask = df['text'].apply(lambda x: utils.check_inflation(x, pre1920=pre1920))

    # indices_to_keep = []
    # unique_id = []
    to_keep = []
    for i, val in enumerate(mask):
        if val:
            # Add the index and the two before and after if they exist
            # pre = len(indices_to_keep)
            # indices_to_keep.extend(range(max(0, i - 2), min(len(df), i + 3)))
            tmp = df.loc[range(max(0, i - 2), min(len(df), i + 3))]
            tmp['mention_flag'] = 0
            tmp.at[i, 'mention_flag'] = 1
            tmp['unique_id'] = i
            to_keep.append(tmp.groupby('id').agg(list).explode(['text', 'mention_flag', 'unique_id']).reset_index())
            # post = len(indices_to_keep)
            # diff = post - pre
            # unique_id += [i]*diff
            # breakpoint()
    if to_keep:
        df = pd.concat(to_keep, axis=0).reset_index(drop=True)
    else: return None
    # indices_to_keep = list(set(indices_to_keep))
    # if len(indices_to_keep) > 0:
    #     breakpoint()
    # df['mention_flag'] = mask.astype(int)
    # df = df.loc[indices_to_keep]
    # breakpoint()
    # try:
        # df['unique_id'] = unique_id
    # except:
    # breakpoint()
    return df


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence_only', action='store_true')
    parser.add_argument('--article_only', action='store_true')
    args = parser.parse_args()

    
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')
    nlp.select_pipes(enable=["sentencizer"])#, "tokenizer"])

    all_sentences = defaultdict(list)
    all_articles = defaultdict(list)
    dir_to_save = '/data/mourad/narratives/inflation/sentences'
    years = range(2012, 2023)
    # years = [2022]
    for year in years:
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            print(f"{year} - {month}")
            out_dir = os.path.join(dir_to_save, str(year))
            file_out =os.path.join(out_dir, f'us_now_{year}_{month}.tsv')
            # if os.path.exists(file_out): continue
            try:
                df = preprocess(month, str(year), sentence_only=args.sentence_only, article_only=args.article_only) 
                if args.article_only:
                    all_articles[year].append(len(df)) # df is dataframe where rows are articles
                elif args.sentence_only:
                    all_sentences[year].append(len(df))
                    print(f"Year {year} has {len(df)} sentences")
                else:
                    breakpoint()
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    df.to_csv(os.path.join(out_dir, f'us_now_{year}_{month}.tsv'), sep='\t', index=False)
            except Exception as e:
                print(e, year, month)
                continue
    if args.sentence_only:
        x = pd.DataFrame(all_sentences)
        x = x.reset_index()
        x = pd.melt(x, id_vars="index", var_name='year', value_name='sentences')
        x = x.rename({'index':'month'}, axis=1)
        x.to_csv("../../data/NOW_sentence_counts.csv", index=False)
    if args.article_only:
        x = pd.DataFrame(all_articles)
        x = x.reset_index()
        x = pd.melt(x, id_vars="index", var_name='year', value_name='articles')
        x = x.rename({'index':'month'}, axis=1)
        x.to_csv("../../data/NOW_article_counts.csv", index=False)
        breakpoint()

