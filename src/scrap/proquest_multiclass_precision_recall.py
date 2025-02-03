from src.utils import utils
import argparse
import json
from jsondiff import diff as jdiff
import numpy as np
import pandas as pd
# from nltk.metrics import masi_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, average_precision_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from collections import defaultdict, Counter
from sklearn.metrics import f1_score
import re
import dirtyjson
from datasets import Dataset, DatasetDict
from seaborn import heatmap
from matplotlib.colors import LogNorm, Normalize


time = ['past', 'present', 'future', 'general']
causes = ['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause']
effects = ['cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect']

categories = {
    # "cause_category": causes,
    # "effect_category": effects,
    "cause_effects": causes + effects + ["none", "no narrative"],
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
    "effect_time": time,
    # "cause_effect": ['cause', 'effect'],
    
    # "category": causes+effects
}
#     "cause": [
#         'demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause'
#         ],
#     "effect": [
#         'cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect'
#     ]
#     # "narrative-time": time,
# }

# cause_effect_categories = categories['category']

def clean_completion(completion):
    pattern = r"```json(.*?)```"
    completion = completion.replace("not-applicable", "general")
    completion = completion.replace("savings/investment", "savings")
    completion = completion.replace("(", "[")
    completion = completion.replace(")", "]")
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        json_completion = matches[0].strip()
        # fixed = re.sub(r'\s(\w+)(,?)', r'"\1"\2', json_completion)
        fixed = re.sub(r'\s(false|true)(,?)', r'"\1"\2', json_completion)
    
        return fixed
    else:
        return completion
    # if completion [0] != "`":
    #     completion =  completion.split("```json")[1]
    # else:
    #     completion = completion.replace("```json\n", "")
    #     completion = completion.split("```")[0]
    # # completion = completion.replace("\n```", "")
    # return completion

def classify_error_types(diff, gold, pred, text):

    gold_meta, gold_narrs = flatten_json(gold)
    pred_meta, pred_narrs = flatten_json(pred)

    return gold_meta, pred_meta, gold_narrs, pred_narrs
    
    # meta_masi = masi_distance(set(gold_meta), set(pred_meta))
    # narr_masi = masi_distance(set(gold_narrs), set(pred_narrs))

    breakpoint()

def calculate_true_positives(gold, preds):
    correct_preds, all_preds, all_instances = Counter(), Counter(), Counter()
    mistakes = defaultdict(list)

    breakpoint()

    gold = flatten_json(gold)
    preds = flatten_json(preds)
    for pred in preds:
        all_preds[pred] += 1
        if pred in gold:
            correct_preds[pred] += 1
    for g in gold:
        all_instances[g] += 1
    
    gold_diff = list(set(gold).difference(preds))
    pred_diff = list(set(preds).difference(gold))
    for pred in pred_diff:
        mistakes[pred] += gold_diff 
    return correct_preds, all_preds, all_instances, mistakes

def find_majority_labels(array_list):
    # Stack the arrays vertically
    stacked = np.vstack(array_list)
    
    # Sum along the vertical axis
    sum_array = np.sum(stacked, axis=0)
    
    # breakpoint()
    # Create a binary array where sum is greater than 1
    result = (sum_array > 1).astype(int)
    
    
    if result.sum() == 0: return None
    
    return result
        

# def flatten_json(data):
#     # meta = []
#     narratives = []
#     # if data['foreign']:
#     #     meta.append('foreign')
#     # else:
#     #     meta.append('domestic')
#     # if data['contains-narrative']:
#     #     meta.append('has_narrative')
#     # else:
#     #     meta.append('no_narrative')

#     if data['contains-narrative']:
#         # if data['inflation-narratives']['counter-narrative']:
#         #     narratives.append('inflation_direction')
#         # else:
#         #     narratives.append('normal_narrative')
#         # narratives.append(data['inflation-narratives']['inflation-time'])
#         for narr in data['inflation-narratives']['narratives']:
#             tmp = list(narr.values())
#             narratives.append(tmp[0])
#             # narratives.append(tmp[0] + "-" + tmp[1])
    

    return narratives

def check_equality(lhs, rhs):
    if lhs == rhs:
        return "correct"
    else:
        return "incorrect"

def check_couter_narrative_contains(item, lst):
    # if item == 'inflation_direction': breakpoint()
    if item == "inflation_direction" and item in lst:
        return "correct"
    elif item == "normal_narrative" and item not in lst:
        return "correct"
    else:
        return "incorrect"
    
def check_contains(item, lst):
    if item in lst:
        return "correct"
    else:
        return "incorrect"
    
def calc_precision(correct_preds, all_preds):
    prec = {}
    for k, v in correct_preds.items():
        p = v / all_preds[k]
        prec[k] = f"{round(p, 2)} ({all_preds[k]})"
    return prec

def calc_recall(correct_preds, all_instances):
    recall = {}
    for k, v in correct_preds.items():
        r = v / all_instances[k]
        recall[k] = f"{round(r, 2)} ({all_instances[k]})"
    return recall


def flatten_gold_json(id, data):
    def convert_to_json(row):
        if isinstance(row, str):
            try:
                row = f"[{row}]"
                row = json.loads(row)
            except:
                breakpoint()
        return row
        
    def extract_narrative_info(row):
        try:
            cause_effect = list(row.keys())[0]
        except:
            breakpoint()
        category = row[cause_effect]
        category_time = row['time']
        cause_category = ""
        effect_category = ""
        cause_time = ""
        effect_time = ""
        if cause_effect == "cause":
            cause_category = category
            cause_time = category_time
        else:
            effect_category = category
            effect_time = category_time
        return {"cause_effect": cause_effect, "cause_category": cause_category, "effect_category": effect_category, "effect_time": effect_time, "cause_time": cause_time}
    
    # breakpoint()
    try:
        df = pd.DataFrame(data['inflation-narratives'])
    except:
        df = pd.DataFrame(None)
        # breakpoint()
    df['id'] = id
    
    if isinstance(data['contains-narrative'], str):
        data['contains-narrative'] = "true" in data['contains-narrative'].lower()
    if isinstance(data['foreign'], str):
        data['foreign'] = "true" in data['foreign'].lower()
    df['foreign'] = data['foreign']
    df['contains-narrative'] = data['contains-narrative']
    annotation_cols = ['inflation-time', 'counter-narrative', 'cause_effect', 'cause_category', 'effect_category', 'cause_time', 'effect_time']
    if df.empty or not data['contains-narrative']:
        df = pd.DataFrame(data, index=[0])
        df['id'] = id
        df['foreign'] = data['foreign']
        df['contains-narrative'] = data['contains-narrative']
        for ac in annotation_cols:
            df[ac] = [""]
        return df
    # breakpoint()
    df['narratives'] = df['narratives'].apply(convert_to_json)
    if isinstance(df.narratives.iloc[0], list):
        df['narratives'] =  df['narratives'].explode('narratives')
    narr_data = df['narratives'].apply(extract_narrative_info)
    narr_df = narr_data.apply(pd.Series)
    df = pd.concat([df, narr_df], axis=1)
    df = df.drop('narratives', axis=1)
    # df = df.rename({'inflation-time': 'inflation_time', 'counter-narrative': 'inflation_direction', 'contains-narrative': 'contains_narrative'}, axis=1)
    return df


def flatten_pred_json(id, data):
    def extract_cause_effect(causes_effects):
        cats = []
        ts = []
        pattern = r"\[(\w+)\]"
        for ce in causes_effects:
            matches = re.findall(pattern, ce[0], re.DOTALL)
            if matches:
                cats.append(matches[0])
            else:
                cats.append(ce[0])
            ts.append(ce[1])
        return cats, ts
    
    annotation_cols = ['inflation-time', 'counter-narrative', 'cause_effect', 'cause_category', 'effect_category', 'cause_time', 'effect_time']
    df = pd.DataFrame(data, index=[0])
    df = df.drop('inflation-narratives', axis=1)
    df['id'] = id
    if df.empty or not data['contains-narrative']:   
        df['foreign'] = data['foreign']
        df['contains-narrative'] = data['contains-narrative']
        for ac in annotation_cols:
            df[ac] = [""]
    else:

    # df = pd.DataFrame(data['inflation-narratives'])
    # df['id'] = id
        if isinstance(data['contains-narrative'], str):
            data['contains-narrative'] = "true" in data['contains-narrative'].lower()
        if isinstance(data['foreign'], str):
            data['foreign'] = "true" in data['foreign'].lower()
        # if isinstance(data['inflation-narratives']['counter-narrative'], str):
        #     data['inflation-narratives']['counter-narrative'] = "true" in data['inflation-narratives']['counter-narrative'].lower()
        # if isinstance(data['inflation-narratives']['inflation-time'], str):
        #     data['inflation-narratives']['inflation-time'] = data['inflation-narratives']['inflation-time'].lower()
        # df['foreign'] = data['foreign']
        # df['contains-narrative'] = data['contains-narrative']
            
        df['inflation-time'] = data['inflation-narratives']['inflation-time']
        df['counter-narrative'] = data['inflation-narratives']['counter-narrative']
        
        df['cause_category'] = None
        df['effect_category'] = None
        df['cause_time'] = None
        df['effect_time'] = None
        if 'causes' in data['inflation-narratives']:
            causes, ctimes = extract_cause_effect(data['inflation-narratives']['causes'])
            df['cause_category'] = [causes]
            df['cause_time'] = [ctimes]
        if 'effects' in data['inflation-narratives']:
            effects, etimes= extract_cause_effect(data['inflation-narratives']['effects'])
            df['effect_category'] = [effects]
            df['effect_time'] = [etimes]
        df = df.explode(['cause_category', 'cause_time'])
        df = df.explode(['effect_category', 'effect_time'])
        # narr_data = df['narratives'].apply(extract_narrative_info)
    # narr_df = narr_data.apply(pd.Series)
    # df = pd.concat([df, narr_df], axis=1)
    # df = df.drop('narratives', axis=1)
    # df = df.rename({'inflation-time': 'inflation_time', 'counter-narrative': 'inflation_direction', 'contains-narrative': 'contains_narrative'}, axis=1)
    df.foreign = df.foreign.astype(bool)
    df['contains-narrative'] = df['contains-narrative'].astype(bool)
    df['counter-narrative'] = df['counter-narrative'].astype(bool)
    df = df.fillna("", inplace=False)
    return df

def calc_scores(gold, pred, label, types):
    gold_cats = gold['cause_category'] + gold['effect_category']
    gold_cats = [x for x in gold_cats if x != ""]
    if gold_cats == []:
        gold_cats = ["no narrative"]
    gold_bin = label_binarize(gold_cats, classes=types).sum(axis=0)
    if gold_bin.max() > 1:
        gold_bin[(gold_bin > 1) & (gold_bin != 0)] = 1
    
    if pred is not None:
        if isinstance(pred, list):
            pred_cats = pred
        else:
            pred_cats = pred['cause_category'] + pred['effect_category']
        pred_cats = [x for x in pred_cats if x != ""]
        
        if pred_cats == []:
            pred_cats = ["no narrative"]
        pred_bin = label_binarize(pred_cats, classes=types).sum(axis=0)
        
        if pred_bin.max() > 1:
            pred_bin[(pred_bin > 1) & (pred_bin != 0)] = 1
    else:
        pred_bin = None
        
            
    return gold_bin, pred_bin

    # for col in gold.columns:
    #     if col in ['id', 'gold']:
    #         continue
    #     res[col] = {}
    #     breakpoint()

  
def save_training_data():
    data = utils.load_hf_dataset(split="train")
    df = data.to_pandas()
    df = df.sample(200, random_state=42)
    df['inp'] = df.apply(lambda x: f"<sentence>{x['text']}</sentence>\n" + utils.reconstruct_training_input(x), axis=1)
    with open("/data/mourad/narratives/prompting_data/train.txt", "w") as f:
        for inst in df.inp:
            f.write(inst + "\n\n")

def load_claude_project_predictions():
    with open("../../data/claude-project-predictions/predictions.txt", 'r') as f:
        data = f.readlines()
    
    texts = []
    preds = []
    for i in range(0, len(data), 2):
        text = data[i].strip()
        # if text[0] == "\"":
            # text = text[1:]
        # if text[-1] == "\"":
            # text = text[:-1]
        pred = data[i+1].strip()
        texts.append(text)
        preds.append(pred)
        
    df = pd.DataFrame({'text': texts, 'completion': preds})
    df['id'] = range(len(df))
    dataset = Dataset.from_pandas(df)
    
    return dataset

def convert_to_std_format(prediction_data, types):
    global causes_effects
    
    completion = prediction_data["completion"]
    completion = json.loads(completion)
    formatted = {"causes": [], "effects": []}
    flattened = []
    for narr, cats in completion.items():
        if cats == "":
            formatted[narr] = []
        else:
            # breakpoint()
            fixed_cats = cats.replace("\"", "").split(",") 
            formatted[narr] = fixed_cats
            flattened.extend(fixed_cats)
    prediction_data["prediction"] = formatted
    
    if len(flattened) == 0:
        flattened = ["no narrative"]
        
   
    prediction_data['bin_prediction'] = label_binarize(flattened, classes=types).sum(axis=0) 
    return prediction_data
        
def make_prediction_confusion_matrix(pred_gold_pairs, types, model, annotator, oracle, train_ds, test_ds, seed, agreement=False):
    # unzip all_mistakes
    
    annotator_initials = [['az', 'qz'],['qz', 'mh'], ['az', 'mh']]
    
    if len(pred_gold_pairs[0]) > 2:
        p1 = [] # az, qz
        p2 = [] # qz, mh
        p3 = [] # az, mh
        for pair in pred_gold_pairs:
            # twopairs.append([[pair[0], pair[1]], [pair[1], pair[2]], [pair[0], pair[2]]])
            p1.append([pair[0], pair[1]])
            p2.append([pair[1], pair[2]])
            p3.append([pair[0], pair[2]])
        # pred_gold_pairs = twopairs
        all = [p1, p2, p3]
    else:
        all = [pred_gold_pairs]
            
    heatmaps = []       
    for ai, pg in enumerate(all):
        # breakpoint()
        preds = []
        golds = []

        for pair in pg:
            # breakpoint()
            preds.append(pair[0])
            golds.append(pair[1])
            
        # preds = all_mistakes[0]
        # golds = all_mistakes[1]
        
        df = pd.DataFrame({'pred': preds, 'gold': golds})
        
        # breakpoint()
        match_dict = defaultdict(list)
        for row in df.itertuples(index=False):
            # breakpoint()
            gold = set(row.gold)
            pred = set(row.pred)

            gold_diff = gold.difference(pred)
            pred_diff = pred.difference(gold)
            pred_intersection = pred.intersection(gold) 
            # breakpoint()
            for p in pred_diff:
                if len(gold) == 0:
                    match_dict[p].append('no narrative')
                else:
                    if len(gold_diff) == 0:
                        # breakpoint()
                        match_dict[p].append('none')
                    else:
                        match_dict[p] += list(gold_diff)
            for g in gold_diff:
                if len(pred_diff) == 0 and len(pred) != 0:
                    # breakpoint()
                    match_dict['none'].append(g)
            for p in pred_intersection:
                match_dict[p].append(p)
            # if len(gold) > 1: 
            #     breakpoint()
            # breakpoint()
        # df = df.explode('pred')
            
        # df = df.groupby('pred').agg(sum)
        
        # breakpoint()
        match_dict = {k: dict(Counter(v)) for k, v in match_dict.items()}
        # pred_common_errors = df.gold.apply(lambda x: Counter(x))
        # breakpoint()
        # print(f"Common {prop} errors")
        cls_errors = 0
        bin_errors = 0
        heatmap_data = []
        for cat_pred in types:
            pred_row = []
            for cat_gold in types:
                if cat_pred in match_dict and cat_gold in match_dict[cat_pred]:
                    pred_row.append(match_dict[cat_pred][cat_gold])
                    if cat_pred != cat_gold and cat_pred not in ['none', 'no narrative'] and cat_gold not in ['none', 'no narrative']:
                        cls_errors += match_dict[cat_pred][cat_gold]
                    elif cat_pred != cat_gold:
                        bin_errors += match_dict[cat_pred][cat_gold]
                else:
                    pred_row.append(0)
            heatmap_data.append(pred_row)
        heatmap_data = np.array(heatmap_data).T
        heatmaps.append(heatmap_data)
        # print(f"Total {cls_errors} classification errors\n")
        # print(f"Total {bin_errors} binary errors\n\n")
    
    
    # breakpoint()     
    # if len(heatmaps) > 1:
    #     heatmap_data = np.sum(heatmaps, axis=0) //3 
    # else:
    #     heatmap_data = heatmaps[0]
    
        fig, ax = plt.subplots(figsize=(10, 8.5))
        heatmap(heatmap_data, xticklabels=types, yticklabels=types, annot=True, norm=LogNorm(), cmap="Blues", fmt='g')

        if 'llama' in model:
            ylabel = 'Fine-tuned Llama 3.1 8B Prediction'
        elif 'gpt' in model:
            ylabel = 'GPT-4o Few-Shot Prediction'
            
        if annotator == 'majority':
            xlabel = 'Majority Vote Human Annotation'
        else:
            xlabel = 'Human Annotation'
            
        fontsize=16
        plt.xlabel(ylabel, fontsize=fontsize)
        plt.ylabel(xlabel, fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        if oracle:
            plt.savefig(f'../../data/error-analysis/{model}-{annotator}-oracle.png', dpi=300, bbox_inches = "tight")
        elif agreement:
            plt.xlabel(f"{annotator_initials[ai][0].upper()} Annotations")
            plt.ylabel(f"{annotator_initials[ai][1].upper()} Annotations")
            plt.savefig(f'../../data/error-analysis/{train_ds}/annotator-majority-{ai}-cm.pdf', format='pdf', dpi=300, bbox_inches = "tight")
        else:
            if seed:
                seed = f"-seed{seed}"
            else:
                seed = "-"    
            plt.savefig(f'../../data/error-analysis/{train_ds}/{model}-{annotator}-test{test_ds}{seed}.pdf', format='pdf', dpi=300, bbox_inches = "tight")
    # for k, v in pred_common_errors.items():
    #     for type

def main(model, system, annotator, binary, oracle, train_ds, test_ds, seed, metaphor):
    # save_training_data()
    # breakpoint()
    if model == 'claude_project':
        prediction_data = load_claude_project_predictions()
    # elif test_ds == 'proquest' and model == 'phi2_ft':
    #     prediction_data = utils.load_hf_dataset(path="/data/mourad/narratives/model_json_preds/proquest_basic/phi2_ft_test_sample_0", dataset=None)
    elif model == 'human_f1':
        print("calculating human f1 average based on majority annotation.")
    else:
        prediction_data = utils.load_hf_dataset(model=model, dataset=train_ds, test_ds=test_ds, fewshot_seed = seed)
       
    # breakpoint()
    # if annotator == 'all':
    #     ann_dfs = []
    #     for ann in ['test_az', 'test_qz', 'test_mh']:
    #         gold_data = utils.load_hf_dataset(path="/data/mourad/narratives/sft_data_proquest")[ann]
    #         gold_data = gold_data.sort("text")
    #         ann_dfs.append(gold_data.to_pandas())
    #     annotator = 'test'

    gold_data = utils.load_hf_dataset(path=f"/data/mourad/narratives/sft_data", dataset=test_ds if test_ds is not None else train_ds)
    
    # breakpoint()
    if metaphor:
        metaphor_data = pd.read_csv("/data/mourad/narratives/metaphor_label.csv")
        metaphor_data = metaphor_data[metaphor_data.metaphor == True]
        metaphors = metaphor_data.text.tolist()
        gold_data = gold_data.filter(lambda x: x['text'] in metaphors)
        prediction_data = prediction_data.filter(lambda x: x['text'] in metaphors)

    if annotator == 'majority':
        az = gold_data['test_az'].sort("text")
        qz = gold_data['test_qz'].sort("text")
        mh = gold_data['test_mh'].sort("text")
        all_annotators = [az, qz, mh]
        
        # az = az.rename({'data': 'data_az'}, axis=1)
        # az['data_mh'] = gold_data['test_mh']['data']
        # az['data_qz'] = gold_data['test_qz']['data']
        # gold_data = Dataset.from_pandas(az)
        # breakpoint()
    else:
        gold_data = gold_data[annotator]
        gold_data = gold_data.sort("text")
        all_annotators = [gold_data]
        
        
    if model == 'human_f1':
        prediction_data = qz
        
    prediction_data = prediction_data.sort("text")

    if system:
        phi2_pred_data = utils.load_hf_dataset(model="phi2_ft", binary=True)

    
    annotator_maj_pairs = []
    
    pred_gold_pairs =[] #defaultdict(list)
    has_narrative = []
    
    ids_to_keep = []
    
    for label, types in categories.items():
        print("Label:", label)
        gold_cats = []
        pred_cats = []
        # breakpoint()
        if model == 'phi2_ft' and test_ds=='proquest':
            prediction_data = prediction_data.map(convert_to_std_format, fn_kwargs={"types": types})
        for inst_id, instance in enumerate(prediction_data):
            if train_ds == 'now_and_proquest' and instance['source'] != 'now': continue
            golds = []
            for gold_data in all_annotators:
                # breakpoint()
                assert instance['text'] == gold_data[inst_id]['text']
                gold = json.loads(utils.reconstruct_training_input(gold_data[inst_id]))
            
                try:
                    if model == 'human_f1':
                        pred = json.loads(utils.reconstruct_training_input(prediction_data[inst_id]))
                    else:
                        pred = json.loads(clean_completion(instance['completion']))
                    if system:
                        phi2_pred = json.loads(phi2_pred_data[inst_id]['completion'])
                        if phi2_pred['contains-narrative'] == "Yes":
                            phi2_pred = True
                        else:
                            phi2_pred = False
                        # phi2_pred = json.loads(clean_completion(phi2_pred_data[inst_id]['completion']))
                except:
                    breakpoint()


                diff = jdiff(gold, pred)
                gold = flatten_gold_json(instance['id'], gold)
                
                # if len(diff) != 0:
                #     print(instance['text'])
                #     print()
                #     print(gold)
                #     print()
                #     print(pred)
                #     breakpoint()
                # print(instance['text'])
                # print(gold)
                # breakpoint()
                if model == 'claude':
                    pred = flatten_pred_json(instance['id'], pred)
                elif model == 'phi2_ft' and test_ds == 'proquest':
                    pred = None
                elif model == 'human_f1':
                    pred = flatten_gold_json(instance['id'], pred)
                else:
                    # pass
                    # breakpoint()
                    # try:
                    if 'base' in model:
                        pass
                    else:
                        pred = flatten_gold_json(instance['id'], pred)
                    # breakpoint()
                    # except:
                        # breakpoint()
                if system:
                    pass
                    # phi2_pred = flatten_gold_json(instance['id'], phi2_pred)
                # pred['gold'] = 0
                
                # gold['gold'] = 1
                # breakpoint()
                # df = pd.concat([gold, pred], axis=0)
                # print(df)
                gold_bin, pred_bin = calc_scores(gold, pred, label, types)
                
                golds.append(gold_bin)
             

            
            if model == 'phi2_ft' and test_ds == 'proquest':
                pred_bin = np.array(instance['bin_prediction'])
            
            if len(golds) > 1:
                # 
                majority = find_majority_labels(golds)
                # breakpoint()
                if majority is None: # or np.any(np.all(majority == golds, axis=1)) == False:
                    continue
                else:
                    gold_bin = majority
                    all_anns = []
                    for g in golds:
                        g[g > 1] = 1
                        g = np.array(types)[np.array(g, dtype=bool)]
                        all_anns.append(g)
                    annotator_maj_pairs.append(all_anns)
                # if (golds[0] != golds[1]).any() or (golds[1] != golds[2]).any() or (golds[0] != golds[2]).any():
                #     continue
                # annotator_scores = [np.abs(pred_bin - g).sum() for g in golds]
                # # get index of minimum score
                # min_score_idx = np.argmin(annotator_scores)
                # gold_bin = golds[min_score_idx]
            
            # breakpoint()
        
            
            diff = np.abs(gold_bin - pred_bin)
            if diff.sum() > 0:
                breakpoint()
            
            # breakpoint()
            # if 1 in diff:
            pred_list = np.array(types)[np.array(pred_bin, dtype=bool)]
            tmp_wrong = np.array(types)[np.array(gold_bin, dtype=bool)]
            # (gold.cause_category + gold.effect_category).tolist() 
            
            # breakpoint()
            # if len(set(pred_list).difference(tmp_wrong)) < len(pred_list): breakpoint()
            # only keep incorrect preds
            # pred_unique = list(set(pred_list).difference(tmp_wrong))
            # tmp_wrong = list(set(tmp_wrong).difference(pred_list))
            
            if len(tmp_wrong) == 0 or tmp_wrong[0] == "":
                tmp_wrong = ["no narrative"]
            if len(pred_list) == 0 or pred_list[0] == "":
                pred_list = ["no narrative"]
         

            # breakpoint()
            if oracle and (gold_bin[-1] == 1 or pred_bin[-1] == 1):#(gold_bin[-1] == 1 or (gold_bin[-1] == 0 and pred_bin[-1] == 1)): #"category" in label and (gold_bin.sum() == 0 or pred_bin.sum() == 0): 
                continue
            # breakpoint()
            if system:
                if phi2_pred in gold['contains-narrative'].tolist():
                    has_narrative.append(1)
                else:
                    has_narrative.append(0)
                if phi2_pred == False:
                    continue
            
            # binary
            if binary:
                gold_bin = gold_bin[-1] == 1
                pred_bin = pred_bin[-1] == 1
            # breakpoint()
            # if gold['cause_effect'].iloc[0] == "cause":
            #     continue
            
            ids_to_keep.append(instance['id'])
            
            pred_gold_pairs.append([list(pred_list), tmp_wrong])
                
            gold_cats.append(gold_bin)
            pred_cats.append(pred_bin)
            # breakpoint()
            # dfs.append(df)
        # df = pd.concat(dfs, axis=0)
        # 
        if len(types) == 2:
            avg = 'binary'
        else:
            avg = 'weighted' #'micro'
            
        if binary:
            avg ='binary'
        # breakpoint()
        f1_macro = f1_score(gold_cats, pred_cats, average='macro')
        f1_micro = f1_score(gold_cats, pred_cats, average=avg)
        f1_weighted = f1_score(gold_cats, pred_cats, average='weighted')
        acc = accuracy_score(gold_cats, pred_cats)
        avg_prec = precision_score(gold_cats, pred_cats, average=avg)
        avg_recall = recall_score(gold_cats, pred_cats, average=avg)

        avg_prec_macro = precision_score(gold_cats, pred_cats, average='macro')
        avg_recall_macro = recall_score(gold_cats, pred_cats, average='macro')

        # print(f"F1 Macro: {round(f1_macro, 2)}")
        # print(f"F1 Weighted: {round(f1_weighted, 2)}\n")

        print(f"Average Micro Precision: {round(avg_prec, 2)}")
        print(f"Average Micro Recall: {round(avg_recall, 2)}")
        print(f"F1 Micro: {round(f1_micro, 2)}\n")
        
        # breakpoint()
        prediction_data = prediction_data.filter(lambda x: x['id'] in ids_to_keep)
        pred_df = prediction_data.to_pandas()[["text"]]
        pred_df[['pred', 'gold']] = pd.DataFrame(pred_gold_pairs)
        pred_df.to_csv(f"../../data/error-analysis/{model}-{annotator}-train_{train_ds}-test_{test_ds}.tsv", index=False)
        print(len(gold_cats))

        # make_prediction_confusion_matrix(annotator_maj_pairs, types, model, annotator, oracle, dataset, agreement=True)
        # # breakpoint()
        make_prediction_confusion_matrix(pred_gold_pairs, types, model, annotator, oracle, train_ds, test_ds, seed)

       
        break
    
        # print(f"Average Macro Precision: {round(avg_prec_macro, 2)}")
        # print(f"Average Macro Recall: {round(avg_recall_macro, 2)}")
        # print(f"F1 Macro: {round(f1_macro, 2)}\n")


        print(f"Accuracy: {round(acc, 2)}\n")
        # types_col = types if len(types) > 2 else [label]
        # pred_df = pd.DataFrame(pred_cats, columns=types_col)
        # gold_df = pd.DataFrame(gold_cats, columns=types_col)
        # breakpoint()

        # continue
        # diff = jdiff(gold, pred)
        # cp, ap, ai, mistakes = calculate_true_positives(gold, pred)
        # correct_preds.update(cp)
        # all_preds.update(ap)
        # all_instances.update(ai)
        # for pred, mistake in mistakes.items():
        #     all_mistakes[pred] += mistake
        # # gold_meta, pred_meta, gold_narr, pred_narr = classify_error_types(diff, gold, pred, instance['text'])
        # scores.append(len(diff) == 0)
        # # gold_metas.append(gold_meta)
        # # pred_metas.append(pred_meta)
        # # gold_narrs.append(gold_narr)
        # # pred_narrs.append(pred_narr)
        # # breakpoint()
        # errors.append(errs)
    # acc = np.mean(scores) 
    # precision = calc_precision(correct_preds, all_preds)
    # recall = calc_recall(correct_preds, all_instances)

    # for k, v in all_mistakes.items():
    #     all_mistakes[k] = dict(Counter(v))
    #     print(k, ":", all_mistakes[k])

    # breakpoint()

    # df = pd.DataFrame({'precision': list(precision.values()), 'recall': list(recall.values())}, index=precision.keys())
    # print(df[df.index.isin(categories['cause'])].T.to_latex(float_format="%.2f"))
    # print(df[df.index.isin(categories['effect'])].T.to_latex(float_format="%.2f"))
    # breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parse_args.add_argument("--split", type=str, default="test")
    parser.add_argument('--model', choices=['human_f1', 'claude', 'gpt35', 'gpt4t', 'gpt4o', "gpt4", "phi2_ft", 'phi2_base', "phi2_ft_600s", "phi2_ft_1000s", "mistral_ft", "llama31_ft", 'llama31_base',  "llama31_ft_test_300s", "llama31_ft_test_600s", "claude_project"], default='gpt4t')
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--system", action="store_true", help="use phi2 to screen for narratives then claude3 for classification")
    parser.add_argument("--annotator", type=str, default="test", choices=['test', 'test_az', 'test_qz', 'test_mh', 'majority', 'avg'])
    parser.add_argument("--binary", action="store_true")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--train_ds", choices=['proquest', 'now', 'now_and_proquest'], required=True)
    parser.add_argument("--test_ds", choices=['proquest', 'now', 'now_and_proquest'], required=False)
    parser.add_argument("--ckpt", type=str, default=None) # llama is 300, phi is 500
    parser.add_argument("--fewshot_seed", choices=['1',2,3,4,5, '4_v2'], required=False)
    parser.add_argument("--metaphor", action="store_true")
    
    args = parser.parse_args()
    
    if args.ckpt is not None:
        args.model += f"_{args.ckpt}s"
    
    # if args.test_ds == None:
    #     args.test_ds = args.train_ds 

    if args.annotator == 'avg':
        for ann in ['test_mh', 'test_qz', 'test_az']:
            main(args.model, args.system, ann, args.binary, args.oracle, args.dataset)
    else:
        main(args.model, args.system, args.annotator, args.binary, args.oracle, args.train_ds, args.test_ds, args.fewshot_seed, args.metaphor)

    


    # what do i wanna know?
    # 1. what was missed (recall)
    # 2. what was wrong (precision)
