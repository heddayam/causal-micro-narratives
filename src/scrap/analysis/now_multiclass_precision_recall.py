from src.utils import utils
import argparse
import json
from jsondiff import diff as jdiff
import numpy as np
import pandas as pd
# from nltk.metrics import masi_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, accuracy_score, average_precision_score, recall_score, precision_score
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


time = ['past', 'present', 'future', 'general']
causes = ['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause']
effects = ['cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect']

categories = {
    # "cause_category": causes,
    # "effect_category": effects,
    "cause_effects": causes + effects + ["none"],
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
#         #     narratives.append('counter_narrative')
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
    # if item == 'counter_narrative': breakpoint()
    if item == "counter_narrative" and item in lst:
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
    
    df = pd.DataFrame(data['inflation-narratives'])
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
    narr_data = df['narratives'].apply(extract_narrative_info)
    narr_df = narr_data.apply(pd.Series)
    df = pd.concat([df, narr_df], axis=1)
    df = df.drop('narratives', axis=1)
    # df = df.rename({'inflation-time': 'inflation_time', 'counter-narrative': 'counter_narrative', 'contains-narrative': 'contains_narrative'}, axis=1)
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
    # df = df.rename({'inflation-time': 'inflation_time', 'counter-narrative': 'counter_narrative', 'contains-narrative': 'contains_narrative'}, axis=1)
    df.foreign = df.foreign.astype(bool)
    df['contains-narrative'] = df['contains-narrative'].astype(bool)
    df['counter-narrative'] = df['counter-narrative'].astype(bool)
    df = df.fillna("", inplace=False)
    return df

def calc_scores(gold, pred, label, types):
    gold_cats = gold['cause_category'] + gold['effect_category']
    pred_cats = pred['cause_category'] + pred['effect_category']
    
    gold_cats = [x for x in gold_cats if x != ""]
    pred_cats = [x for x in pred_cats if x != ""]
    
    if gold_cats == []:
        gold_cats = ["none"]
    if pred_cats == []:
        pred_cats = ["none"]

    
    gold_bin = label_binarize(gold_cats, classes=types).sum(axis=0)
    pred_bin = label_binarize(pred_cats, classes=types).sum(axis=0)
    

    if pred_bin.max() > 1:
        pred_bin[(pred_bin > 1) & (pred_bin != 0)] = 1
    if gold_bin.max() > 1:
        gold_bin[(gold_bin > 1) & (gold_bin != 0)] = 1
        
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


def main(model, system):
    # save_training_data()
    # breakpoint()
    prediction_data = utils.load_hf_dataset(model=model)
    gold_data = utils.load_hf_dataset(path="/data/mourad/narratives/sft_data_proquest")[annotator]
    if system:
        phi2_pred_data = utils.load_hf_dataset(model="phi2_ft", binary=True)

    scores = []
    errors = []
    gold_metas = []
    pred_metas = []
    gold_narrs = []
    pred_narrs = []
    
    all_mistakes = defaultdict(list)
    dfs = []
    correct_preds, all_preds, all_instances = Counter(), Counter(), Counter()
    has_narrative = []
    approx = 0
    approx_narr = 0
    for label, types in categories.items():
        print("Label:", label)
        gold_cats = []
        pred_cats = []
        for inst_id, instance in enumerate(prediction_data):
            gold = json.loads(utils.reconstruct_training_input(instance))
            try:
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
            # if len(diff) != 0:
            #     print(instance['text'])
            #     print(gold)
            #     print(pred)
            #     breakpoint()
            # print(instance['text'])
            # print(gold)
            # breakpoint()
            if model == 'claude':
                pred = flatten_pred_json(instance['id'], pred)
            else:
                # pass
                pred = flatten_gold_json(instance['id'], pred)
            if system:
                pass
                # phi2_pred = flatten_gold_json(instance['id'], phi2_pred)
            pred['gold'] = 0
            gold = flatten_gold_json(instance['id'], gold)
            gold['gold'] = 1
            # breakpoint()
            df = pd.concat([gold, pred], axis=0)
            # breakpoint()
            
            gold_bin, pred_bin = calc_scores(gold, pred, label, types)
            # breakpoint()
            if (gold_bin + pred_bin).max() > 1:
                # print(df)
                # breakpoint()
                approx += 1
                approx_narr += gold['contains-narrative'].iloc[0]
                # approx_nonarr = 
            # oracle
            if gold['contains-narrative'].iloc[0] and pred['contains-narrative'].iloc[0]: #"category" in label and (gold_bin.sum() == 0 or pred_bin.sum() == 0): 
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
            # gold_bin = gold_bin[-1] == 1
            # pred_bin = pred_bin[-1] == 1
                
            gold_cats.append(gold_bin)
            pred_cats.append(pred_bin)
            # breakpoint()
            # dfs.append(df)
        # df = pd.concat(dfs, axis=0)
        # 
        if len(types) == 2:
            avg = 'binary'
        else:
            avg = 'micro'
        
        # avg ='binary'
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
        breakpoint()
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
    parser.add_argument('--model', choices=['claude', 'gpt35', 'gpt4t', "gpt4", "phi2_ft", "mistral_ft", "llama3_300steps"], default='gpt4t')
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--system", action="store_true", help="use phi2 to screen for narratives then claude3 for classification")
    args = parser.parse_args()

    main(args.model, args.system)

    


    # what do i wanna know?
    # 1. what was missed (recall)
    # 2. what was wrong (precision)
