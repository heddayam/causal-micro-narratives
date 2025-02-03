import pandas as pd
from glob import glob
import os
from fuzzywuzzy import fuzz, process
from urllib.parse import urlparse
import pycountry
import swifter
from tqdm.auto import tqdm
import argparse
import csv
from src.utils import utils
from nltk.tokenize.treebank import TreebankWordDetokenizer

def split_url(url):
    domain = urlparse(url).netloc
    domain = domain.replace('www.', '')
    return domain



def check_foreign_country_in_string(input_string):
    # List of all country names
    country_names = [country.name for country in pycountry.countries]

    # Check if any country name is mentioned in the input string
    for country in country_names:
        if country in input_string:
            return country

    return None



# def match_locations(df):
#     locs = pd.read_csv("/data/mourad/narratives/newspaper_locations.csv")
#     df['url'] = df.url.apply(split_url)
#     locs['url'] = locs.link.apply(split_url)
#     x = df.merge(locs, on='url', how='inner')
#     breakpoint()


def load_sources():
    df = pd.read_csv("~/economic-narratives/data/sources.csv")
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", action="store_true", help="add location data to labeled data")
    args = parser.parse_args()
    
    dir = '/data/mourad/narratives/inflation/sentences/'
    sources_dir = '/data/now/extracted/sources/'

    src_loc = load_sources()
    # src_loc = src_loc[src_loc.scope != 'National']
    src_loc = src_loc[["source", "city", "state", "scope"]]
    src_loc = src_loc.replace("no data", None)

    if args.labeled:
        all_df = pd.read_csv("../../data/NOW_for_labelstudio_v1.tsv", sep="\t")
    else:
        all_df = pd.read_json('/data/mourad/narratives/inflation/all_filtered.jsonl.gz', lines=True, orient='records', compression='gzip')

    dfs = []
    for year in tqdm(os.listdir(dir)):
        print(year)
        # if int(year) < 2017: continue
        if int(year) <= 2020:
            src_df = pd.read_table(os.path.join(sources_dir, f'now-sources-{year}.txt'), sep="\t", encoding= "ISO-8859-1", header=None, on_bad_lines="warn", low_memory=True, lineterminator='\n', quoting=csv.QUOTE_NONE)

        for month_file in tqdm(os.listdir(os.path.join(dir, year))):
            month = month_file.split(".")[0][-2:]
            # df = pd.read_csv(os.path.join(dir, year, month_file), sep='\t')
            # df = df[df.mention_flag == 1]
            # df = df.drop(['mention_flag', 'unique_id'], axis=1)
            # df['month'] = int(month)
            # df['year'] = year
            # try:
            df = all_df[(all_df.year == int(year)) & (all_df.month == int(month))]
            # except:
                # df = all_df.copy()
            # df = df.groupby(['id', 'unique_id', ]).agg(list)
            # df = df.drop('mention_flag', axis=1)
            # df.text = df.text.apply(lambda x: ' '.join(x))
            # df['date'] = f"{year}{month}01"
            df = df.reset_index(drop=True)

            # breakpoint()
            try:
                if int(year) > 2020:
                    src_df = pd.read_table(os.path.join(sources_dir, f'sources-{year[-2:]}-{month}.txt'), sep="\t",encoding= "ISO-8859-1", header=None, on_bad_lines="warn",low_memory=True, lineterminator='\n', quoting=csv.QUOTE_NONE) 
                # else:
                #     # src_df = pd.read_table(os.path.join(sources_dir, f'now-sources-{year}.txt'), sep="\t", encoding= "ISO-8859-1", header=None, on_bad_lines="warn", low_memory=True, lineterminator='\n', quoting=csv.QUOTE_NONE)
                
            except: breakpoint()
            # len(src_df.country != '')
            if len(src_df.columns) == 7:
                src_df.columns = ["id", "n_words", "date", "country", "website", "url", "title"]
            else:
                src_df.columns = ["index", "id", "n_words", "date", "country", "website", "url", "title"]

            src_df = src_df[src_df.country == 'US']
            src_df_clean = src_df[["id", "website", "title"]].copy()
            # src_df.country = src_df.title.swifter.apply(check_foreign_country_in_string)
            # src_df = src_df.drop('n_words', axis=1)
            # breakpoint()
            src_df_clean.id = src_df_clean.id.astype(int)
            # breakpoint()
            df = df.merge(src_df_clean, on='id', how='left').dropna()
            df = df.rename({"website": "source"}, axis=1)

            try:
                len_before = len(df)
                df = df.merge(src_loc, on="source", how="inner")
                if len(df) < len_before:
                    print('merged smaller than original')
                    # breakpoint()
            except Exception as e:
                print(e)
                breakpoint()
            dfs.append(df.reset_index(drop=True))
            # match_locations(df)
    df = pd.concat(dfs, axis=0).drop_duplicates()
    # x = pd.read_csv("/data/mourad/narratives/tmp/labeled_test.csv")
    # print(df[df.id.isin(x.id)])
    # print(df[df.id.isin(x.id)].id.nunique())
    if args.labeled:
        detokenizer = TreebankWordDetokenizer()
        df.text = df.text.apply(lambda x: utils.detokenize(x, detokenizer))
        df.to_json('../../data/labeled_with_location.jsonl.gz', orient='records', lines=True, compression='gzip')
    else:
        # df.to_json('/data/mourad/narratives/idea_relations.jsonl.gz', orient='records', lines=True, compression='gzip')
        df.to_json('/data/mourad/narratives/inflation/all_filtered_with_location.jsonl.gz', orient='records', lines=True, compression='gzip')
