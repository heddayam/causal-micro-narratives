import openai
import anthropic
import json5
import re
import swifter
import pandas as pd
import xml.etree.ElementTree as ET
from glob import glob
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from datasets import load_from_disk, Dataset, DatasetDict
import subprocess
import json
import os

PROQUEST_MAX_LEN = 150

synonyms = set(['recession', 'depression', 'slump', 'contraction', 'downturn', 'slowdown', 'decline', 'trough', 
                'collapse', 'bad times', 'hard times', 'crash', 'slide', 'downswing', 'downtrend', 'shrinking', 'withdrawal', 
                'abatement', 'lull', 'recede', 'crunch', 'tailspin', 'crisis'])

econ_synonyms = set(['economy', 'economic'])

regions = ['national', 'southeast', 'west', 'northeast', 'midwest', 'southwest']

state_abbreviation_to_name = {
    'al': 'alabama', 'ala': 'alabama',
    'ak': 'alaska', 'als': 'alaska',
    'az': 'arizona', 'ari': 'arizona',
    'ar': 'arkansas', 'ark': 'arkansas',
    'ca': 'california', 'cal': 'california',
    'co': 'colorado', 'col': 'colorado',
    'ct': 'connecticut', 'con': 'connecticut',
    'dc': 'district of columbia', 'dis': 'district of columbia',
    'de': 'delaware', 'del': 'delaware',
    'fl': 'florida', 'fla': 'florida',
    'ga': 'georgia', 'geo': 'georgia',
    'hi': 'hawaii', 'haw': 'hawaii',
    'id': 'idaho', 'ida': 'idaho',
    'il': 'illinois', 'ill': 'illinois',
    'in': 'indiana', 'ind': 'indiana',
    'ia': 'iowa', 'iow': 'iowa',
    'ks': 'kansas', 'kan': 'kansas',
    'ky': 'kentucky', 'ken': 'kentucky',
    'la': 'louisiana', 'lou': 'louisiana',
    'me': 'maine', 'mai': 'maine',
    'md': 'maryland', 'mar': 'maryland',
    'ma': 'massachusetts', 'mas': 'massachusetts',
    'mi': 'michigan', 'mic': 'michigan',
    'mn': 'minnesota', 'min': 'minnesota',
    'ms': 'mississippi', 'mis': 'mississippi',
    'mo': 'missouri', 'mis': 'missouri',
    'mt': 'montana', 'mon': 'montana',
    'ne': 'nebraska', 'neb': 'nebraska',
    'nv': 'nevada', 'nev': 'nevada',
    'nh': 'new hampshire', 'nha': 'new hampshire',
    'nj': 'new jersey', 'nje': 'new jersey',
    'nm': 'new mexico', 'nme': 'new mexico',
    'ny': 'new york', 'nyo': 'new york',
    'nc': 'north carolina', 'nca': 'north carolina',
    'nd': 'north dakota', 'nda': 'north dakota',
    'oh': 'ohio', 'ohi': 'ohio',
    'ok': 'oklahoma', 'okl': 'oklahoma',
    'or': 'oregon', 'ore': 'oregon',
    'pa': 'pennsylvania', 'pen': 'pennsylvania',
    'ri': 'rhode island', 'rhi': 'rhode island',
    'sc': 'south carolina', 'sca': 'south carolina',
    'sd': 'south dakota', 'sda': 'south dakota',
    'tn': 'tennessee', 'ten': 'tennessee',
    'tx': 'texas', 'tex': 'texas',
    'ut': 'utah', 'uta': 'utah',
    'vt': 'vermont', 'ver': 'vermont',
    'va': 'virginia', 'vir': 'virginia',
    'wa': 'washington', 'was': 'washington',
    'wv': 'west virginia', 'wva': 'west virginia',
    'wi': 'wisconsin', 'wis': 'wisconsin',
    'wy': 'wyoming', 'wyo': 'wyoming'
}


region_mapping = {
    'alabama': 'southeast',
    'alaska': 'west',
    'arizona': 'west',
    'arkansas': 'southeast',
    'california': 'west',
    'colorado': 'west',
    'connecticut': 'northeast',
    'delaware': 'northeast',
    'florida': 'southeast',
    'georgia': 'southeast',
    'hawaii': 'west',
    'idaho': 'west',
    'illinois': 'midwest',
    'indiana': 'midwest',
    'iowa': 'midwest',
    'kansas': 'midwest',
    'kentucky': 'southeast',
    'louisiana': 'southeast',
    'maine': 'northeast',
    'maryland': 'northeast',
    'massachusetts': 'northeast',
    'michigan': 'midwest',
    'minnesota': 'midwest',
    'mississippi': 'southeast',
    'missouri': 'midwest',
    'montana': 'west',
    'nebraska': 'midwest',
    'nevada': 'west',
    'new hampshire': 'northeast',
    'new jersey': 'northeast',
    'new mexico': 'west',
    'new york': 'northeast',
    'north carolina': 'southeast',
    'north dakota': 'midwest',
    'ohio': 'midwest',
    'oklahoma': 'southwest',
    'oregon': 'west',
    'pennsylvania': 'northeast',
    'rhode island': 'northeast',
    'south carolina': 'southeast',
    'south dakota': 'midwest',
    'tennessee': 'southeast',
    'texas': 'southwest',
    'utah': 'west',
    'vermont': 'northeast',
    'virginia': 'southeast',
    'washington': 'west',
    'west virginia': 'southeast',
    'wisconsin': 'midwest',
    'wyoming': 'west',
    'district of columbia': 'northeast'
}

missing_cities = {
    'tempe': 'arizona',
    'corpus christi': 'texas',
    'annapolis': 'maryland',
    'newport beach': 'california',
    'new brunswick': 'new jersey',
    'metairie': 'louisiana',
    'purchase': 'new york',
    'boulder': 'colorado',
    'norcross': 'georgia',
    'arlington': 'virginia',
    'carlsbad': 'california',
    'dublin': 'new hampshire',
    'winchester': 'virginia',
    'vermillion': 'south dakota',
    'charlotte': 'north carolina',
    'chicopee': 'massachusetts',
    'southfield': 'michigan',
    'austin': 'texas',
    'littleton': 'colorado',
    'centennial': 'colorado',
    'williams bay': 'wisconsin',
    'princeton': 'new jersey',
    'youngstown': 'ohio',
    "east hartford": "connecticut",
    "gulfport": "mississippi",
    "dover": "delaware",
    "waupaca": "wisconsin",
    "dewitt": "new york",
    "palm desert": "california",
    "bellows falls": "vermont",
    "pueblo": "colorado",
    "new york city": "new york",
    "gretna": "louisiana",
    "cromwell": "connecticut",
    "princeton": "new jersey",
    "aberdeen": "maryland",
    "fort wayne": "indiana",
    "longwood": "florida",
    "walnut creek": "california",
    "rocky river": "ohio",
    "port chester": "new york",
    "myrtle beach": "south carolina",
    "boyne city": "michigan",
    "north canton": "ohio",
    "ft lauderdale": "florida",
    "akron": "ohio",
    "méxico city": "mexico",
    "neeah": "wisconsin",
    "redondo beach": "california",
    "duluth": "minnesota",
    "stockton": "california",
    "saint louis": "missouri",
    "pembroke": "massachusetts",
    "huntington": "new york",
    "beverly hills": "california",
    "charlotte": "north carolina"
}

time = ['past', 'present', 'future', 'general']
causes = ['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause']
effects = ['cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect']

categories = {
    "cause_category": causes,
    "effect_category": effects,
    "contains-narrative": [
        False, True
    ],
    "foreign": [
          False, True
    ],
    "inflation-time": time,
    "counter-narrative": [
         False, True
    ],
    "cause_time": time,
    "effect_time": time
    # "cause_effect": ['cause', 'effect'],
    
    # "category": causes+effects
}

model_mapping = {
    'gpt4': 'gpt-4',
    'gpt4t': 'gpt-4-1106-preview',
    'gpt4o': 'gpt-4o',
    'gpt35': 'gpt-3.5-turbo-1106',
    'o1-mini': 'o1-mini',
    'claude2': 'claude-2',
    'claude': 'claude-3-opus-20240229'
}

def michigan_survey_state_to_region():
    df = pd.read_csv("../../data/economic-indicators/michigan-survey-region-map.csv")
    region_map = pd.Series(df.Abbreviation.values,index=df.State).to_dict()
    return region_map

def reconstruct_training_input(instance):
    template = instance['template'].split("#")
    data = instance['data'].split("#")
    
    # breakpoint()

    # if str(instance['id']) == "23232402":
    #     breakpoint()
    interleaved = [item for pair in zip(template, data) for item in pair]
    input = "".join(interleaved) + template[-1]
    
    return input


def scp_file(local_file,  remote_path, remote_host='dsi', local_host=None):
    if remote_host == 'dsi':
        remote_host = 'fe01.ds.uchicago.edu'

    if local_host:
        local_file = f"mourad@{local_host}:{local_file}"

    try:
        # Construct the scp command
        scp_command = [
            'scp',
            '-r',
            local_file,
            f'mourad@{remote_host}:{remote_path}'
        ]

        # Execute the scp command
        subprocess.run(scp_command, check=True)
        print(f'Successfully copied {local_file} to {remote_host}:{remote_path}')
    except subprocess.CalledProcessError as e:
        print(f'Error: {e}')

def load_hf_dataset(path="/data/mourad/narratives/sft_data", split=None, model=None, binary=False, dataset='now', test_ds=None, fewshot_seed=None):
    if model:
        if binary:
            path = f"/data/mourad/narratives/model_json_binary_preds/{model}"
        else:
            path = f"/data/mourad/narratives/model_json_preds/{model}"
    
    # breakpoint()
    if dataset and test_ds is None:
        path += f"_{dataset}"
    else:
        path += f"_train-{dataset}_test-{test_ds}"
    if fewshot_seed is not None:
        path += f"_seed{fewshot_seed}"
    print(path)
    dataset = load_from_disk(path)
    if split:
        return dataset[split]
    return dataset

def read_all_data(path="/data/mourad/narratives/inflation", location=True):
    if location:
        df = pd.read_json(os.path.join(path, 'all_filtered_with_location.jsonl.gz'), orient='records', lines=True, compression='gzip')
    else:
        df = pd.read_json(os.path.join(path, 'all_filtered.jsonl.gz'), orient='records', lines=True, compression='gzip')
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds
 
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

def check_parse(parse, cause_effect):
    if isinstance(parse, dict):
        if cause_effect in parse:
            return parse[cause_effect]
    else:
        return None

def explode_create_cause_cols(df):
    df = pd.concat([df, df.parse.apply(pd.Series)], axis=1)
    df = df.drop('parse', axis=1)
    df = df.explode('causes')
    df = df.reset_index(drop=False)
    df = df.rename({'index': "sentence_id"}, axis=1)
    return df

def create_cause_effect_cols(df):
    df = df.reset_index(drop=True)
    df['cause'] = df.parse.apply(lambda x: check_parse(x, 'cause'))
    df['effect'] = df.parse.apply(lambda x: check_parse(x, 'effect'))
    df = df.dropna()
    df = df.drop('parse', axis=1)
    return df.reset_index(drop=True)

def explode_create_cause_effect_cols(df):
    breakpoint()
    df = df.explode('parse')
    df = df.reset_index(drop=True)
    df['cause'] = df.parse.apply(lambda x: check_parse(x, 'cause'))
    df['effect'] = df.parse.apply(lambda x: check_parse(x, 'effect'))
    df = df.dropna()
    df = df.drop('parse', axis=1)
    return df.reset_index(drop=True)

def check_recession_syns(line):
    pattern = re.compile(r'^(?=.*\b({0})\b)(?=.*\b({1})\b).*$'.format('|'.join(econ_synonyms), '|'.join(synonyms)), re.IGNORECASE)
    matches = re.findall(pattern, line)
    if matches: 
        return True
    else:
        return False

def check_inflation(line, pre1920):
    if pre1920:
        import inflection as inf
        inflation_alts = {'appreciation', 'depreciation', 'devaluation', 'debasement'}#, 'high'] #'appreciation'
        currency_alts = {"currency", "money", "note", "wage", "payment"}# "price"]
        # pattern = r'\b(debasement|depreciation|appreciation|devaluation|rising\sprices)\b'
        # s = set(inflation_alts)
        # matches = sum(1 for x in re.finditer(r'\b\w+\b',line) if x.group().lower() in s) > 0
        
       # Process the line once
        words = set(inf.singularize(word.lower()) for word in line.split())
        
        # Initialize flags to check presence of at least one word from each category
        found_inflation = found_currency = False
        
        # Check for matches in a single pass
        for word in words:
            if word in inflation_alts:
                found_inflation = True
            if word in currency_alts:
                found_currency = True
            # Early exit if both categories are matched
            if found_inflation and found_currency:
                return True
        
        return False
    else:
        pattern = r'\binflation\b'
        matches = re.search(pattern, line, re.IGNORECASE)
    if matches:
        return True
    else:
        return False

def extract_id(text):
    text_split = text.strip().split(' ')
    if len(text_split) > 1:
        return text_split[0].replace('@@', ''), ' '.join(text_split[1:])
    return '' ,text

def filter_now(text):
    tmp = text
    tmp = tmp.str.replace("@ @ @ @ @ @ @ @ @ @", "@")
    tmp = tmp.str.replace("<p> Advertisement <p>", "")
    tmp = tmp.str.replace("<h>", "")
    tmp = tmp.str.replace("<p>", "")
    return tmp

def detokenize(text, treebank_detokenizer):
    text = treebank_detokenizer.detokenize(text.split()).strip()
    # Remove spaces before and after punctuation
    text = re.sub(r'\s+([.,;!?])', r'\1', text)
    text = re.sub(r'([.,;!?])\s+"', r'\1"', text)
    # Split the sentence into parts by quote
    parts = text.split('"')

    # Remove leading space from every second part (starting from index 0)
    for i in range(1, len(parts), 2):
        parts[i] = parts[i].lstrip()

    # Join the parts back together
    detokenized_text = '"'.join(parts)
    return detokenized_text

def read_now_by_month(years=None, simple=False, iter_only=False):
    if years is None:
        years = range(2012, 2023)
    years = [str(y) for y in years]
    # years = [2022]
    for year in years:
        for month in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            if iter_only:
                yield month, year, 0
            files = glob(f'/data/now/extracted/{year}/text/*{year[-2:]}*{month}-*.txt')
            # src_fname = '/data/now/extracted/2021/text/21-09-us1.txt'
            # src_fname = f'/data/now/extracted/{year}/text/{year[-2:]}-{month}-us1.txt'
            dfs = []
            for src_fname in files:
                df = pd.read_table(src_fname, sep="\t", encoding= "ISO-8859-1", header=None, on_bad_lines="warn")

                # df = df.head(100)

                df.columns = ['text']
                if simple:
                    dfs.append(df)
                    continue
                # text = open('/data/now/extracted/2021/text/21-09-us1.txt').readlines()
                # df = pd.DataFrame({'text': text})
            
                df['text'] = filter_now(df['text'])

                res = df.text.swifter.apply(extract_id)
                id, text = zip(*res)
                df.text = text
                df['id'] = id
                
                dfs.append(df)
            if len(dfs) > 0:
                df = pd.concat(dfs, axis=0)
            else: 
                df = None 
            yield month, year, df

def init_llm(model):
    redis_port = 6379
    if model.startswith('gpt') or model.startswith('o1'):
        from src.utils.llm_cache import OpenAIAPICache
        # openai.api_key_path = "./openai_key.txt"
        openai.api_key = "API_KEY"
        cache = OpenAIAPICache(mode="chat", port=redis_port)
    elif model.startswith('claude'):
        from src.utils.llm_cache import ClaudeAPICache
        anthropic_client = anthropic.Anthropic(api_key="API_KEY")
        cache = ClaudeAPICache(anthropic_client, port=redis_port)
    return cache

def extract_narrative_category(pred):
    cats = []
    if pred['contains-narrative']:
        if len(pred['inflation-narratives']['narratives']) > 0:
            narrs = pred['inflation-narratives']['narratives'][0]
            if isinstance(narrs, str):
                narrs = re.sub(r'\s', '', narrs)
                try:
                    narrs = eval(narrs)
                except:
                    print("UNPARSABLE", narrs)
                    return None
            if isinstance(narrs, tuple) or isinstance(narrs, list):
                cats = [list(narr.values())[0] for narr in narrs]
            else:
                cats = [list(narrs.values())[0]]
            
    cats = [cat for cat in cats if cat in causes+effects]
    if len(cats) > 0:
        cats.append('contains_narrative')
    return list(set(cats))


def results_to_latex(results, narratives, title, dep_var_note=None, additional_notes=None, panel=False):
    """
    Convert statsmodels results object to a LaTeX table string.
    
    Parameters:
    -----------
    results : (narrative_name, regression object) dict of statsmodels.regression.linear_model.RegressionResultsWrapper. 
        Values to to include as columns
    title : str
        Table title
    dep_var_note : str, optional
        Note about dependent variable
    additional_notes : list of str, optional
        Additional notes to add below the table
        
    Returns:
    --------
    str : LaTeX code for the table
    """
    
    latex = []
    
    # Begin table environment
    # latex.append(r'\begin{table}[htbp]')
    # latex.append(r'\centering')
    # latex.append(r'\caption{' + title + '}')
    latex.append(r'\begin{tabular}{l' + 'c' * len(results) + '}')
    latex.append(r'\toprule')
    
    # Column headers
    latex.append(r' & ' + ' & '.join([f'({narr})' for narr in narratives]) + r' \\')
    latex.append(r'\midrule')
    
    # Get all variable names
    all_vars = set()
    # breakpoint()
    for narr_name, result in results.items():
        all_vars.update(result.params.index)
    
    # Add each variable's coefficients and standard errors
    for var in all_vars:
        coef_line = var
        se_line = ''  # For standard errors
        
        for narr_name, result in results.items():
            if var in result.params.index:
                coef = result.params[var]
                if panel:
                    stderr = result.std_errors[var]
                else:
                    stderr = result.bse[var]
                
                tstat = result.tstats[var]
                t_threshold = 1.96  

                if tstat > t_threshold:
                    coef_line += f' & \\textbf{{{coef:.4f}}}'
                    se_line += f' & (\\textbf{{{stderr:.4f}}})' 
                else:
                    coef_line += f' & {coef:.4f}'
                    se_line += f' & ({stderr:.4f})'  
                # coef_line += f' & {coef:.4f}'
                # se_line += f' & ({stderr:.4f})'
            else:
                coef_line += ' & '
                se_line += ' & '
        
        latex.append(coef_line + r' \\')
        latex.append(r'\vspace{0.2cm}')
        latex.append(se_line + r' \\')
        
    
    # Add regression statistics
    latex.append(r'\midrule')
    
    # breakpoint()
    
    # Number of observations
    obs_line = 'Observations'
    # for i in range(len(all_vars)):
    for narr_name, result in results.items():
        obs_line += f' & {result.nobs}'
    latex.append(obs_line + r' \\')
    
    # R-squared
    r2_line = '$R^2$'
    r2_overall_line = '$R^2_{overall}$'
    # for i in range(len(all_vars)):
    for narr_name, result in results.items():
        if panel:
            r2_line += f' & {result.rsquared:.3f}'
            r2_overall_line += f' & {result.rsquared_overall:.3f}'
        else:
            r2_line += f' & {result.rsquared:.3f}'
            r2_overall_line += f' & {result.rsquared_adj:.3f}'
    latex.append(r2_line + r' \\')
    latex.append(r2_overall_line + r' \\')
    latex.append(r'\bottomrule')
    
    # Add notes
    if additional_notes or dep_var_note:
        latex.append(r'\multicolumn{' + str(len(results) + 1) + '}{p{\textwidth}}{')
        if additional_notes:
            for note in additional_notes:
                latex.append(note + r'\\')
        if dep_var_note:
            latex.append(dep_var_note)
        latex.append('}')
    
    # End table environment
    latex.append(r'\end{tabular}')
    # latex.append(r'\end{table}')
    
    return '\n'.join(latex)



def separate_results_to_latex(results, title, dep_var_note=None, additional_notes=None):
    """
    Convert statsmodels results object to a LaTeX table string.
    
    Parameters:
    -----------
    results : list of statsmodels.regression.linear_model.RegressionResultsWrapper
        List of regression results to include as columns
    title : str
        Table title
    dep_var_note : str, optional
        Note about dependent variable
    additional_notes : list of str, optional
        Additional notes to add below the table
        
    Returns:
    --------
    str : LaTeX code for the table
    """
    
    latex = []
    
    # Begin table environment
    # latex.append(r'\begin{table}[htbp]')
    # latex.append(r'\centering')
    # latex.append(r'\caption{' + title + '}')
    latex.append(r'\begin{tabular}{l' + 'c' * len(results) + '}')
    latex.append(r'\toprule')
    
    # Column headers
    latex.append(r' & ' + ' & '.join([f'({i})' for i in range(len(results))]) + r' \\')
    latex.append(r'\midrule')
    
    # Get all variable names
    all_vars = []
    meta_params = set()
    for result in results:
        print(result.summary())
        breakpoint()
        param = result.params.index
        for p in param:
            if 'lag' in p or p == 'const' or p == 'time':
                meta_params.add(p)
            else:
                all_vars.append(p)
    
    all_vars.extend(list(meta_params))
    
    # breakpoint()
    # Add each variable's coefficients and standard errors
    for var in all_vars:
        coef_line = var
        se_line = ''  # For standard errors
        # breakpoint()
        for result in results:
            if var in result.params.index:
                coef = result.params[var]
                stderr = result.bse[var]
                
                coef_line += f' & {coef:.4f}'
                se_line += f' & ({stderr:.4f})'
            else:
                coef_line += ' & '
                se_line += ' & '
        
        latex.append(coef_line + r' \\')
        latex.append(r'\vspace{0.2cm}')
        latex.append(se_line + r' \\')
        
    
    # Add regression statistics
    latex.append(r'\midrule')
    
    # Number of observations
    obs_line = 'Observations'
    for result in results:
        obs_line += f' & {result.nobs}'
    latex.append(obs_line + r' \\')
    
    # R-squared
    r2_line = 'Adjusted $R^2$'
    for result in results:
        r2_line += f' & {result.rsquared_adj:.3f}'
    latex.append(r2_line + r' \\')
    latex.append(r'\bottomrule')
    
    # Add notes
    if additional_notes or dep_var_note:
        latex.append(r'\multicolumn{' + str(len(results) + 1) + '}{p{\textwidth}}{')
        if additional_notes:
            for note in additional_notes:
                latex.append(note + r'\\')
        if dep_var_note:
            latex.append(dep_var_note)
        latex.append('}')
    
    # End table environment
    latex.append(r'\end{tabular}')
    # latex.append(r'\end{table}')
    
    return '\n'.join(latex)

# def query_gpt(text, cache, prompt2,model):
   
#     test_sents = [
#         "The anchor who questioned Warren seemed to suggest that the Massachusetts senator 's tax proposal would lead to national economic collapse as the wealthiest Americans fled for tax havens .",
#         "Hoping to rebuild their savings after the economic crisis , we may see older Americans looking to rejoin the workforce , bringing with them years of experience and expertise ."
#     ]

#     query_prompt = f"Sentence: {text}\n\n" + PROMPT

#     if model == 'gpt4':
#         model_string = 'gpt-4'
#     elif model == 'gpt4t':
#         model_string = 'gpt-4-1106-preview'
#     elif model == 'gpt35':
#         model_string = "gpt-3.5-turbo-1106"

#     try:
#         completion = cache.generate(
#             model=model_string, #gpt-3.5-turbo-1106
#             max_tokens=500,
#             temperature=0,
#             messages=[{"role": "user", "content": prompt2.format(HUMAN_PROMPT="", AI_PROMPT="", SENTENCE=text)}]
#             # prompt=prompt2.format(HUMAN_PROMPT=HUMAN_PROMPT.strip(), AI_PROMPT=AI_PROMPT, SENTENCE=text),
#         )
        
#         completion = completion['choices'][0]['message']['content'].strip()
#         answer_idx = completion.rfind("Answer:")
#         if answer_idx != -1:
#             return completion[answer_idx:]
#         else:
#            return None
#         root = ET.fromstring('<root>'+completion+'</root>')
#         # print(completion.completion.strip())
        
#         # task1 = root.find('task1').text.strip()
#         # task2 = root.find('task2').text.strip()
#         # task1 = json5.loads(''.join(task1.split('\n')))
#         # task2 = json5.loads(''.join(task2.split('\n')))
#         # res = task1+task2
        
#         ans = root.find('answer')
#         if ans is None or ans.text is None:
#             return None
#         res = ans.text.strip()
#         if res == 'None':
#             return None
#         else:
#             return res
#         # bracket_index = res.find('[')
#         # res = res[bracket_index:] if bracket_index != -1 else res
#         # res = json5.loads(res)
#         # return res
#     except Exception as e:
#         print(e)
#         print(text)
#         return None
#     # return res
#     # print(res)

# def query_claude(text, cache, prompt1, prompt2):
#     from anthropic import HUMAN_PROMPT, AI_PROMPT
#     res=''
#     text = text.replace("\"", "'")
#     try:
#         # completion = cache.generate(
#         #     model="claude-2",
#         #     max_tokens_to_sample=10,
#         #     temperature=0,
#         #     prompt=prompt1,
#         #     )
#         completion = cache.generate(
#             model="claude-2",
#             max_tokens_to_sample=5,
#             temperature=0,
#             stop_sequences=["]"],
#             prompt=prompt1.format(HUMAN_PROMPT=HUMAN_PROMPT.strip(), AI_PROMPT=AI_PROMPT, SENTENCE=text), #{AI_PROMPT}
#         )

#         relevant = completion.completion
#         # if foreign != 'True' and economic == 'True':
#         if relevant[0].lower() == 'y':
#             completion = cache.generate(
#                 model="claude-2",
#                 max_tokens_to_sample=500,
#                 temperature=0,
#                 prompt=prompt2.format(HUMAN_PROMPT=HUMAN_PROMPT.strip(), AI_PROMPT=AI_PROMPT, SENTENCE=text),
#             )

#             root = ET.fromstring('<root>'+completion.completion.strip()+'</root>')
#             # print(completion.completion.strip())
            
#             # task1 = root.find('task1').text.strip()
#             # task2 = root.find('task2').text.strip()
#             # task1 = json5.loads(''.join(task1.split('\n')))
#             # task2 = json5.loads(''.join(task2.split('\n')))
#             # res = task1+task2
#             res = root.find('answer').text.strip()
#             # bracket_index = res.find('[')
#             # res = res[bracket_index:] if bracket_index != -1 else res
#             res = json5.loads(res)
#             return res
#         else:
#             return None
#     except Exception as e:
#         print(e)
#         print(text)
#         print(res)
#         return None
    # print(completion.completion)