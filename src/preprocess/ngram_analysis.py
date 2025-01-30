from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import utils.utils as utils
import argparse
import swifter
from tqdm import tqdm
import old_prompts
import seaborn as sns
import pickle
from collections import defaultdict
from nltk.stem import WordNetLemmatizer

def vectorize_corpus(df, vocab, ngram_range=(1,1)):
    corpus = df.text
    # vec = CountVectorizer(tokenizer=utils.LemmaTokenizer(),
    #                             strip_accents = 'unicode', # works 
    #                             stop_words = None, # works
    #                             lowercase = True,
    #                             token_pattern=None,
    #                             vocabulary=vocab) # works
                                # max_df = 0.5, # works
                                # min_df = 10) # works
    # vec = CountVectorizer(ngram_range=ngram_range, stop_words='english', lowercase=True, vocabulary=vocab)
    vec = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', lowercase=True, vocabulary=vocab, use_idf=False, norm=None, binary=True)

    # Transform the corpus into a bag-of-words sparse matrix and sum the occurrences
    bow = vec.fit_transform(corpus)
    # breakpoint()
    # bow = bow.toarray().mean(axis=0) * 100
    # return bow
    return pd.DataFrame(bow.toarray())


def get_counts(key_index, bow, keyword='inflation'):
    # key_index = list(feature_names).index(keyword)
    key_counts = bow.toarray()[:, key_index].sum()
    return key_counts
    
def plot_lineplot(df): #, x, month_year):

    wnl = WordNetLemmatizer()
    x_label = "time"
    x = df.time
    feature_df = df[list(vocabulary.keys())]
    unique_kw = [wnl.lemmatize(t) for t in feature_df.columns]
    feature_df.columns = unique_kw
    feature_df = feature_df.groupby(feature_df.columns, axis=1).agg(sum)
    features = feature_df.columns
    feature_df[x_label] = x
    indiv_df = df.groupby('time').agg(np.mean)*100
    indiv_df = indiv_df.reset_index()
    breakpoint()
    xlabels = []
    current_yr = ''
    for m in df.month_year.unique().tolist():
        year = m.split('-')[1]
        if current_yr != year:
            xlabels.append(str(year))
            current_yr = year
        else:
            xlabels.append('')
    for word in features:
        plt.figure(figsize=(8, 6))
        # breakpoint()
        sns.barplot(data=indiv_df, x=x_label, y=word, width=1, edgecolor='black')
        # sns.histplot(data=df[df.word == word], x=x_label, stat='probability', discrete=True)
        plt.xticks(list(set(x)), xlabels)
        plt.xlabel("Year")
        plt.ylabel(f"% of articles that mention {word}")
        plt.savefig(f'/data/mourad/narratives/ngrams/{save_mod}/{word}.png', dpi=300)
        plt.clf()

    df = pd.melt(df, id_vars=[x_label], value_vars=features, var_name="word", value_name='counts')
    df = df[df.counts != 0]
    plt.figure(figsize=(8, 6))
    sns.kdeplot(data=df[df.word != 'inflation'], x=x_label, hue='word', multiple="fill", bw_adjust=1, clip=(0,len(xlabels)))
    plt.xticks(list(set(x)), xlabels)
    plt.xlabel("Year")
    plt.ylabel(f"Words related to inflation")    
    plt.savefig(f'/data/mourad/narratives/ngrams/{save_mod}/compare.png', dpi=300)
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-rw", "--rewrite", action='store_true', help='rewrite count vectorized corpus')
    parser.add_argument("-f", "--filtered", action='store_true', help='use sentences filtered by inflation')
    parser.add_argument("-d", "--debug", action='store_true', help="debug don't save")
    args = parser.parse_args()

    if args.filtered:
        save_mod = 'inflation_filtered'
    else:
        save_mod = 'all'

    # if args.filtered:
    #     vectorizer_base = f"/data/mourad/narratives/ngrams/count_vectorizer/inflation_filtered/"
    #     if args.rewrite:
    #         os.system("rm /data/mourad/narratives/ngrams/count_vectorizer/inflation_filtered/*")
    # else:
    #     vectorizer_base = f"/data/mourad/narratives/ngrams/count_vectorizer/all/"
    #     if args.rewrite:
    #         os.system("rm /data/mourad/narratives/ngrams/count_vectorizer/all/*")

    vocabulary = {
        "inflation": 0,
        "demand": 1,
        "supply": 2,
        "supplies": 3,
        "cost": 4,
        "costs": 5,
        "wage": 6, 
        "wages": 7,
        "uncertainty": 8,
        "uncertainties": 9,
        "saving": 10,
        "savings": 11,
        "investment": 12,
        "investments": 13
        # "interest rate": 7
    }

    if args.filtered:
        df = pd.read_json('/data/mourad/narratives/inflation/all_filtered.jsonl.gz', orient='records', lines=True, compression='gzip')
        df = df.sort_values(['year', 'month'])
        df['time'] = df.groupby(['year', 'month']).ngroup()
        counts = vectorize_corpus(df, vocabulary)
        counts.columns = vocabulary.keys()
        counts = counts.astype(int)
        month_year = df.month.astype(str) + '-' + df.year.astype(str)
        counts['time'] = df.time
        counts['month_year'] = month_year
        plot_lineplot(counts) #, df.time, month_year.unique().tolist())
    else:
        dfs = []
        for i, (month, year, df) in enumerate(utils.read_now_by_month(years=None, simple=True, iter_only=False)):
            if args.debug and year == "2013": break
            if df is None: continue
            month_year_label = f"{month}-{year[-2:]}"
            print(month_year_label)
            vectorizer_base = f"/data/mourad/narratives/ngrams/count_vectorizer/{save_mod}/"
            if args.rewrite:
                os.system(f"rm /data/mourad/narratives/ngrams/count_vectorizer/{save_mod}/*")

            vectorizer_path = os.path.join(vectorizer_base, f"{month_year_label}.pkl")
            if (not os.path.exists(vectorizer_path) or args.rewrite) and not args.debug:
                print('rewrite')
                counts = vectorize_corpus(df, vocabulary)
                counts.columns = vocabulary.keys()
                counts = counts.astype(int)
                counts['year'] = int(year)
                counts['month'] = int(month)
                counts = counts.sort_values(['year', 'month'])
                counts['time'] = i
                with open(vectorizer_path, "wb") as f:
                    pickle.dump(counts, f)
            else:
                print('existing')
                with open(vectorizer_path, "rb") as f:
                    counts = pickle.load(f)
            dfs.append(counts)
            # keyword_counts.append(bow)
            # months.append(month_year_label)
        
        df = pd.concat(dfs, axis=0)
        # df = pd.DataFrame(keyword_counts, columns=vocabulary.keys())
        df['month_year'] = df.month.astype(str) + '-' + df.year.astype(str)
        plot_lineplot(df)