import pandas as pd
import swifter
from glob import glob
import spacy
import re
import os
from src.utils import utils
from datasets import load_dataset
from tqdm import tqdm
from preprocess_now import get_surrounding_rows
import multiprocessing as mp
import concurrent.futures

tqdm.pandas()

def apply_get_surrounding_rows(args):
    group, pre1920 = args
    # group_id, group_data = group
    # result = group.apply(lambda x: get_surrounding_rows(x, pre1920=pre1920), axis=1)
    result = get_surrounding_rows(group, pre1920=pre1920)
    return result #group_id, 

# def apply_parallel(df, func, pre1920, n_jobs=-1):
#     groups = df.groupby('id')
#     args = [(group, pre1920) for group in groups]  # Prepare arguments for each group
#     with mp.Pool(processes=n_jobs) as pool:
#         results = list(tqdm(pool.imap(func, args), total=len(groups)))  # Pass the prepared list of arguments
    
#     results_dict = dict(results)
#     return df.groupby('id').apply(lambda x: results_dict[x.name])

def get_sentences(text, nlp):
    return_sents = []
    if len(text) > 1000000:
        for i in range(len(text) // 1000000):
            return_sents += get_sentences(text[i:1000000*(i+1)], nlp)
        return return_sents
    else:
        doc = nlp(text)
        return [sent.text.strip() if (sent.text.strip() != '') else None for sent in doc.sents]

def apply_get_sentences(args):
    text, nlp = args
    return text.apply(lambda x:get_sentences(x, nlp))

def main():
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('sentencizer')
    nlp.select_pipes(enable=["sentencizer"])#, "tokenizer"])

    dir_to_save = '/data/mourad/narratives/inflation/americanstories_sentences'

    all_lens = 0
    total_articles = 0
    for year in range(1900, 1910): #skip 1774 - bad data - 1893
        try:
            dataset = load_dataset("dell-research-harvard/AmericanStories", 
                                    "all_years", #"all_years_content_regions",
                                    year_list=[str(year)],
                                    cache_dir="/data/mourad/huggingface"
                                    )
        except Exception as e:
            print(f"Year {year} not found")
            continue
        
        df = dataset[str(year)].to_pandas()
        df = df.rename({'article_id': 'id', 'article': 'text'}, axis=1)
        
        total_articles += len(df)
        # df.text = df.text.apply(get_sentences)
        
        # Get the number of available CPU cores
        num_cores = mp.cpu_count()-10
        
        # Split the DataFrame into chunks based on the number of cores
        chunk_size = len(df) // num_cores
        chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Create a multiprocessing pool
        pool = mp.Pool(processes=num_cores)
        
        # Apply the get_sentences function to each chunk using the pool with a progress bar
        results = []
        with tqdm(total=len(chunks), desc='Processing') as pbar:
            for result in pool.imap(apply_get_sentences, [(chunk['text'], nlp) for chunk in chunks]):
                results.append(result)
                pbar.update(1)
        
        # Close the pool to release resources
        pool.close()
        pool.join()
        
        # Concatenate the results back into a single DataFrame
        df['text'] = pd.concat(results)
        #########
        
        df = df.explode('text').reset_index(drop=True)
        df = df.dropna()
        
 
        # breakpoint()
        
        # df.loc[0, 'text'] = "~ he TrE1-L --c-5---s -zs CD--ge-e era-' l bT\n\n\n= iIl7-era-e nea-o. =hot= zhc.e Gock r zcquir-\n-4 k1ou'e5zs l-zrc=lT -rr=15-d beyond tie\n-zsch of a fsgzr csne : AfIer r. departure ct\nS7nt-s1-zr, thz .o.zl Command III re CzPe de-\nFo.vsd IL TcDEz.nt-; Rzvm-n1, ]: IS tire inflation tf1=-e."
        if year < 1915:
            pre1920 = True
        else:
            pre1920 = False
            
         
        ##########   
        # Split the DataFrame into chunks based on the number of cores
       
        grouped = df.groupby("id")[df.columns]
        chunks = [group for name, group in grouped]
        # chunk_size = len(grouped) // num_cores
        # chunks = [list(grouped)[i:i+chunk_size] for i in range(0, len(grouped), chunk_size)]
       
        # Create a multiprocessing pool
        pool = mp.Pool(processes=num_cores)
        # breakpoint()
        # Apply the get_sentences function to each chunk using the pool with a progress bar
        results = []
        with tqdm(total=len(chunks), desc='Processing') as pbar:
            for result in pool.imap(apply_get_surrounding_rows, [(chunk, pre1920) for chunk in chunks]):
                if result is not None:
                    results.append(result)
                pbar.update(1)
        
        # Close the pool to release resources
        pool.close()
        pool.join()
        
        # Concatenate the results back into a single DataFrame
        if len(results) > 0:
            df = pd.concat(results)
        else:
            df = None
        
        ###########
            
            
            
        # df = apply_parallel(df, process_group, pre1920, n_jobs=num_cores)

        # df = df.groupby('id')[df.columns].progress_apply(lambda x: get_surrounding_rows(x, pre1920=pre1920), include_groups = True)
        
        if df is not None:
            df = df.reset_index(drop=True)
            # breakpoint()
            # dfs.append(df)
        
            for t in df[df.mention_flag == 1].text:
                print(t.replace("\n", " ").strip())
                print("____________________")
            print(len(df[df.mention_flag == 1]))
            print(year)
            all_lens += len(df[df.mention_flag == 1])
            
            df.to_csv(os.path.join(dir_to_save, f"{year}.tsv"), sep="\t")
        # breakpoint()

    print(all_lens)
    print(total_articles)
    
if __name__ == "__main__":
    main()