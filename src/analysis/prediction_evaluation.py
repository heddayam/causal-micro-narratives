from src.utils import utils
import argparse
import json
# from jsondiff import diff as jdiff
import numpy as np
import pandas as pd
# from nltk.metrics import masi_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, accuracy_score, average_precision_score, recall_score, precision_score
from sklearn.preprocessing import label_binarize, LabelBinarizer

import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from collections import defaultdict, Counter
from sklearn.metrics import f1_score
import re
# import dirtyjson

causes = ['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause']
effects = ['cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect']
causes_effects = causes + effects + ['none']

def convert_to_std_format(prediction_data, lb):
    global causes_effects
    
    completion = prediction_data["completion"]
    completion = json.loads(completion)
    formatted = {"causes": [], "effects": []}
    flattened = []
    for narr, cats in completion.items():
        if cats == "":
            formatted[narr] = []
        else:
            fixed_cats = cats.replace("\"", "").split(",") 
            formatted[narr] = fixed_cats
            flattened.extend(fixed_cats)
    prediction_data["prediction"] = formatted
    
    if len(flattened) == 0:
        flattened = ["none"]
        
   
    prediction_data['bin_prediction'] = binarize_labels(flattened, lb)
    return prediction_data

def binarize_labels(flattened, lb):
    bin = lb.transform(flattened).sum(axis=0)
    if bin.max() > 1:
        bin[(bin > 1) & (bin != 0)] = 1
    return bin

def convert_gold_to_std_format(gold_data, lb):
    global causes_effects
    
    gold = json.loads(utils.reconstruct_training_input(gold_data))
    flattened = gold['causes'] + gold['effects']

    if len(flattened) == 0:
        flattened = ["none"]
    
    # gold_bin = lb.transform(flattened).sum(axis=0)
    # if gold_bin.max() > 1:
    #     gold_bin[(gold_bin > 1) & (gold_bin != 0)] = 1
    gold_data['bin_prediction'] =  binarize_labels(flattened, lb)
    return gold_data

def main(split, oracle, binary):
    
    lb = LabelBinarizer()
    lb.fit(causes_effects)
    
    data_path = f"/data/mourad/narratives/model_json_preds/proquest_basic/phi2_ft_test_sample_0"
    prediction_data = utils.load_hf_dataset(path=data_path)
    prediction_data = prediction_data.map(convert_to_std_format, fn_kwargs={"lb": lb})
    
    gold_data = utils.load_hf_dataset(path="/data/mourad/narratives/sft_data_proquest_basic", split=split)
    gold_data = gold_data.map(convert_gold_to_std_format, fn_kwargs={"lb": lb})
    
    gold_cats = gold_data['bin_prediction']
    pred_cats = prediction_data['bin_prediction']
    
    none_idx = list(lb.classes_).index('none')
    if oracle:
        oracle_gold = []
        oracle_pred = []
        for i, gold in enumerate(gold_cats):
            if gold[none_idx] == 0:
                if pred_cats[i][none_idx] == 0:
                    breakpoint()
                    oracle_gold.append(gold)
                    oracle_pred.append(pred_cats[i])
            #     pred_cats[i] = gold
            # elif pred_cats[i][none_idx] == 1:
            #     pred_cats[i] = gold
        gold_cats = oracle_gold
        pred_cats = oracle_pred
                
    if binary:
        # breakpoint()
        gold_cats = np.array(gold_cats)[:,none_idx]
        pred_cats = np.array(pred_cats)[:,none_idx]
            
    
    
    breakpoint()
    
    f1_macro = f1_score(gold_cats, pred_cats, average='macro')
    f1_micro = f1_score(gold_cats, pred_cats, average='micro')
    f1_weighted = f1_score(gold_cats, pred_cats, average='weighted')
    acc = accuracy_score(gold_cats, pred_cats)
    avg_prec = precision_score(gold_cats, pred_cats, average='micro')
    avg_recall = recall_score(gold_cats, pred_cats, average='micro')

    avg_prec_macro = precision_score(gold_cats, pred_cats, average='macro')
    avg_recall_macro = recall_score(gold_cats, pred_cats, average='macro')

    # print(f"F1 Macro: {round(f1_macro, 2)}")
    # print(f"F1 Weighted: {round(f1_weighted, 2)}\n")

    print(f"Average Micro Precision: {round(avg_prec, 2)}")
    print(f"Average Micro Recall: {round(avg_recall, 2)}")
    print(f"F1 Micro: {round(f1_micro, 2)}\n")
    
    # breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--binary", action="store_true")
    # parser.add_argument('--model', choices=['claude', 'gpt35', 'gpt4t', "gpt4", "phi2_ft", "mistral_ft"], default='gpt4t')
    # parser.add_argument("--debug", type=bool, default=False)
    # parser.add_argument("--system", action="store_true", help="use phi2 to screen for narratives then claude3 for classification")
    args = parser.parse_args()

    main(args.split, args.oracle, args.binary)