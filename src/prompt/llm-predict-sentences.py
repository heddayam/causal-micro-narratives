from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
import pandas as pd
import numpy as np
import utils.utils as utils
import argparse
import swifter
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm
# import prompt
from pathlib import Path

tqdm.pandas()

def get_completion(sentence, cache, prompt, model, claude_or_gpt):
    model_str = utils.model_mapping.get(model)

    if claude_or_gpt == 'claude':
        completion = cache.generate(
            model=model_str,
            max_tokens_to_sample=500,
            temperature=0,
            prompt=prompt.format(HUMAN_PROMPT=HUMAN_PROMPT.strip(), AI_PROMPT=AI_PROMPT, SENTENCE=sentence),
        )
        completion = completion.completion.strip()
    else:
        completion = cache.generate(
            model=model_str, #gpt-3.5-turbo-1106
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt.format(HUMAN_PROMPT="", AI_PROMPT="", SENTENCE=sentence)}]
        )
        completion = completion['choices'][0]['message']['content'].strip()

    answer_indicator = "Answer:"
    answer_idx = completion.find(answer_indicator)
    if answer_idx != -1:
        return completion[answer_idx + len(answer_indicator):].strip()
    else:
        return None


def init_and_query_llm(df, type, model):
    claude_or_gpt = 'claude' if model.startswith('claude') else 'gpt'
    cache = utils.init_llm(model)
    prompt = Path(f"./prompts/classify/{claude_or_gpt}/inflation-{type}.txt").read_text()
    df['completion'] = df.text.progress_apply(get_completion, args=(cache, prompt, model, claude_or_gpt))


def extract_top_ngrams(corpus, ngram_range=(1, 1), top_n=10):
    """
    Extracts top n-grams from the given corpus based on their frequency.

    Parameters:
        corpus (str): The text from which to extract n-grams.
        ngram_range (tuple): The range of n-values for the n-grams. E.g., (1, 1) for unigrams, (1, 2) for unigrams and bigrams, etc.
        top_n (int): The number of top n-grams to return.

    Returns:
        list: A list of tuples where each tuple is (ngram, frequency).
    """
    # Initialize CountVectorizer with the desired ngram_range
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_df = 0.6).fit(corpus)
    # vec = TfidfVectorizer(ngram_range=ngram_range, stop_words='english', max_df=0.4).fit(corpus)

    # Transform the corpus into a bag-of-words sparse matrix and sum the occurrences
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 

    # Extract n-gram and their counts
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    # Return the top n n-grams
    return words_freq[:top_n]

def filter_parses(parse):
    if not isinstance(parse, list): return None
    sents = []
    for pair in parse:
        try:
            cause = pair['cause']
            effect = pair['effect']
            combined = f"{cause} , {effect}"
        except:
            breakpoint()
        if 'coronavirus' in combined or 'pandemic' in combined: continue
        sents.append(combined)

    return ' .'.join(sents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-y", "--year", type=int, required=False, default=None, help="Year to filter on")
    parser.add_argument("-rw", "--rewrite", action='store_true', help='rewrite the gpt4 output? cached results will still be used')
    parser.add_argument('--model', choices=['claude', 'gpt35', 'gpt4t', "gpt4"], default='claude')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--do_eval', action='store_true', help="whether to do eval on manually annotated data")
    parser.add_argument('--eval_path', default="../data/eval/annotated/effect-done-mourad.tsv", help="path to manually annotated data for eval")
    parser.add_argument('--inflation_type', choices=['cause', 'effect'])
    parser.add_argument("--downsample", type=int, default=10000, help="Downsample # of sentences to analyze with LLM to this, -1 to not downsample")
    parser.add_argument('--out_dir', default='/data/mourad/narratives/inflation')
    # parser.add_argument("--prompt_path", default="")
    # Parse the arguments
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    out_dir = out_base / args.model / args.inflation_type 
    if args.do_eval:
        out_path = out_dir /  "eval.parquet"
    else:
        out_path = out_dir /  f"{args.year}.parquet"
    
    if os.path.exists(out_path) and not args.rewrite:
        df = pd.read_parquet(out_path)
    else:
        df = pd.read_json(out_base / 'all_filtered.jsonl.gz', orient='records', lines=True, compression='gzip')

        if not args.do_eval and args.year:
            df = df[df.year == args.year]

        if args.debug:
            df = df.sample(10)
        elif args.do_eval:
            sample_ids = pd.read_csv(args.eval_path, sep="\t")[['id', 'text']]#.id.tolist()
            df = df.merge(sample_ids, on=['id', 'text'], how='inner')
        elif args.downsample != -1:
            df = df.sample(args.downsample, random_state=42)
        init_and_query_llm(df, args.inflation_type, args.model)
        # df = utils.explode_create_cause_cols(df)
        # breakpoint()
        
        if args.do_eval:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            df.to_parquet(out_path)
            breakpoint()
        elif not args.debug:
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            df.to_parquet(out_path)

        

    
    # df = df.dropna()
    # df['doc'] = df.parse.apply(filter_parses)
    # top_bigrams = extract_top_ngrams(df.doc, ngram_range=(2, 2), top_n=100)
    # print(top_bigrams)
   
    # Don't forget to include all implied and secondary cause/effect pairs in the list.