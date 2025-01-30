from src.utils import utils
import argparse
import json
from jsondiff import diff as jdiff
import numpy as np
import pandas as pd
# from nltk.metrics import masi_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import math
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from collections import defaultdict

colors_raw = [
    # Full set of Plotly's built-in sequential colorscales
    px.colors.sequential.Plasma,
    px.colors.sequential.Viridis,
    px.colors.sequential.Cividis,
    px.colors.sequential.Turbo,
    px.colors.sequential.Jet,

    # Some qualitative colorscales
    px.colors.qualitative.Plotly,
    px.colors.qualitative.Light24,
    px.colors.qualitative.D3,

    # Additional named colors 
    "indigo", 
    "teal", 
    "darkgoldenrod",
    "lightsalmon",
    "forestgreen",
    "mediumorchid",
    "cornflowerblue",
    "slategray"
]
colors = []
for c in colors_raw:
    if isinstance(c, list):
        colors += c
    else:
        colors.append(c)


time = ['past', 'present', 'future', 'general']

categories = {
    "contains-narrative": [
        True, False
    ],
    "foreign": [
        True, False
    ],
    "inflation-time": time,
    "counter-narrative": [
        True, False
    ],
    "cause": [
        'demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause'
        ],
    "effect": [
        'cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect'
    ]
    # "narrative-time": time,
}

def clean_completion(completion):
    completion = completion.replace("```json\n", "")
    completion = completion.replace("\n```", "")
    return completion

def classify_error_types(diff, gold, pred, text):

    gold_meta, gold_narrs = flatten_json(gold)
    pred_meta, pred_narrs = flatten_json(pred)

    return gold_meta, pred_meta, gold_narrs, pred_narrs
    
    # meta_masi = masi_distance(set(gold_meta), set(pred_meta))
    # narr_masi = masi_distance(set(gold_narrs), set(pred_narrs))

    breakpoint()

def flatten_json(data):
    meta = []
    narratives = []
    if data['foreign']:
        meta.append('foreign')
    else:
        meta.append('domestic')
    if data['contains-narrative']:
        meta.append('has_narrative')
    else:
        meta.append('no_narrative')

    if data['contains-narrative']:
        if data['inflation-narratives']['counter-narrative']:
            narratives.append('counter_narrative')
        # else:
        #     narratives.append('normal_narrative')
        narratives.append(data['inflation-narratives']['inflation-time'])
        for narr in data['inflation-narratives']['narratives']:
            tmp = list(narr.values())
            narratives.append(tmp[0])
            narratives.append(tmp[0] + "-" + tmp[1])
    

    return meta, narratives

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

def make_sankey(gold_metas, pred_metas, gold_narrs, pred_narrs, model):
    val = 0.1

    source, narr_source = [], []
    target, narr_target = [], []
    value, narr_value = [], []

    category_color_map = {}
    cats = ['incorrect', 'correct', 'has-narrative', 'no-narrative', 'foreign', 'domestic', 'inflation-time', 'counter_narrative?', 'narrative', "cause", "effect"]
    colors = ['#F2A2A2', '#C2EABD', '#0000FF', '#0000FF', '#FFD700', '#FFD700', '#800080', '#FFA500', '#00FFFF', '#808080', '#FF00FF']
    general_color = "gray"
    for i, cat in enumerate(cats):
        category_color_map[cat] = colors[i]

    src_trgt_ctr = defaultdict(int)
    color_map = {}
    
    color, narr_color = [], []
    for i, gold_meta in enumerate(gold_metas):
        pred_meta = pred_metas[i]

        foreign_pred = pred_meta[0]
        foreign_gold = gold_meta[0]
        t = check_equality(foreign_pred, foreign_gold)
        color.append(category_color_map[t])
        src_trgt_ctr[(foreign_pred, f"(gold) {foreign_gold}")] += 1
        color_map[(foreign_pred, f"(gold) {foreign_gold}")] = category_color_map[t]


        narrative_pred = pred_meta[1].replace("_", "-")
        narrative_gold = gold_meta[1].replace("_", "-")

        t = check_equality(narrative_pred, narrative_gold)
        color.append(category_color_map[t])
        value.append(val)
        src_trgt_ctr[(narrative_pred, f"(gold) {narrative_gold}")] += 1
        color_map[(narrative_pred, f"(gold) {narrative_gold}")] = category_color_map[t]
        
       
        if pred_meta[1] == "has_narrative":
            pred_narr = pred_narrs[i]
            gold_narr = gold_narrs[i]

            pred_inflation_time_idx = 0 
            if pred_narr[0] == "counter_narrative":
                pred_counter_narrative = "counter_narrative"
                pred_inflation_time_idx = 1
            else:
                pred_counter_narrative = "normal_narrative"
            gold_inflation_time_idx = 0
            if not gold_narr:
                continue
            if gold_narr[0] == "counter_narrative":
                gold_counter_narrative = "counter_narrative"
                gold_inflation_time_idx = 1
            else:
                gold_counter_narrative = "normal_narrative"
            
            t = check_equality(pred_counter_narrative, gold_counter_narrative)   
            src_trgt_ctr[("has-narrative", pred_counter_narrative)] += 1
            color_map[("has-narrative", pred_counter_narrative)] = category_color_map[t]
           
            src_trgt_ctr[(pred_counter_narrative, f"(gold) {gold_counter_narrative}")] += 1
            color_map[(pred_counter_narrative, f"(gold) {gold_counter_narrative}")] = category_color_map[t]

            pred_time = pred_narr[pred_inflation_time_idx]
            gold_time = gold_narr[gold_inflation_time_idx]
            t = check_equality(pred_time, gold_time)
            src_trgt_ctr[("inflation-time", pred_time)] += 1
            src_trgt_ctr[(pred_time, f"(gold) {gold_time}")] += 1
            color_map[("inflation-time", pred_time)] = general_color#category_color_map["inflation-time"]
            color_map[(pred_time, f"(gold) {gold_time}")] = category_color_map[t]
            continue

            # s = "inflation-time"
            # source.append(s) #has-narrative
            # color.append(category_color_map[s])
            # target.append(pred_narr[inflation_time_idx])

            for i in range(inflation_time_idx+1, len(pred_narr), 2):
                narr_label = pred_narr[i]

                s = 'narrative'
                narr_source.append(s) #pred_narr[inflation_time_idx])
                narr_color.append(category_color_map[s])
                # narr_target.append(narr_label)
                narr_value.append(val)

                # breakpoint()
                if narr_label in categories["cause"]:
                    narr_target.append("cause")
                    narr_source.append("cause")
                    narr_color.append(category_color_map["cause"])
                else:
                    narr_target.append("effect")
                    narr_source.append("effect")
                    narr_color.append(category_color_map["effect"])

                narr_target.append(pred_narr[i])
                narr_value.append(val)

                src_target_ctr[()] += 1


                narr_source.append(pred_narr[i])
                t = check_contains(pred_narr[i], gold_narr)
                narr_target.append(t)
                narr_color.append(category_color_map[t])
                narr_value.append(val)
        elif gold_meta[1] == "has_narrative":
            gold_narr = gold_narrs[i]
            if gold_narr[0] == "counter_narrative":
                gold_counter_narrative = "counter_narrative"
                gold_inflation_time_idx = 1
            else:
                gold_counter_narrative = "normal_narrative"    
    
            src_trgt_ctr[("(gold) has-narrative", f"(gold) {gold_counter_narrative}")] += 1
            color_map[("(gold) has-narrative", f"(gold) {gold_counter_narrative}")] = category_color_map["incorrect"]


    src_trgt = list(src_trgt_ctr.keys())
    value = list(src_trgt_ctr.values())
    source = []
    target = []
    color = []
    for s, t in src_trgt:
        if s == 'inflation-time': continue
        source.append(s)
        target.append(t)
        color.append(color_map[(s, t)])

    # source, target = zip(*src_trgt)

    le = LabelEncoder()
    le.fit(source+target)
    source = le.transform(source)
    target = le.transform(target)
    label = le.classes_
    # color = []
    # for s in source:
    #     color.append(colors[s])
    # breakpoint()

    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 40,
        line = dict(color = "black", width = 0.5),
        label = label, #["A1", "A2", "B1", "B2", "C1", "C2"],
        color = "gray"
        ),
        link = dict(
        source = source, #[0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A1, B1, ... 
        target = target, #[2, 3, 3, 4, 4, 5],
        value = value,#[8, 4, 2, 8, 4, 2],
        color= color
    ))])

    fig.update_layout(title_text=f"{model} Sankey Diagram", font_size=20)
    fig.write_html(f"/data/mourad/narratives/plots/sankey_{model}.html")

    le = LabelEncoder()
    le.fit(narr_source+narr_target)
    source = le.transform(narr_source)
    target = le.transform(narr_target)
    label = le.classes_


    fig = go.Figure(data=[go.Sankey(
        node = dict(
        pad = 15,
        thickness = 40,
        line = dict(color = "black", width = 0.5),
        label = label, #["A1", "A2", "B1", "B2", "C1", "C2"],
        color = "gray"
        ),
        link = dict(
        source = source, #[0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A1, B1, ... 
        target = target, #[2, 3, 3, 4, 4, 5],
        value = narr_value,#[8, 4, 2, 8, 4, 2],
        color= narr_color
    ))])

    fig.update_layout(title_text=f"{model} Narratives Sankey Diagram", font_size=20)
    fig.write_html(f"/data/mourad/narratives/plots/sankey_{model}_narratives.html")

def main(model):
    prediction_data = utils.load_hf_dataset(model=model)
    scores = []
    errors = []
    gold_metas = []
    pred_metas = []
    gold_narrs = []
    pred_narrs = []
    for instance in prediction_data:
        gold = json.loads(utils.reconstruct_training_input(instance))
        pred = json.loads(clean_completion(instance['completion']))
        diff = jdiff(gold, pred)
        gold_meta, pred_meta, gold_narr, pred_narr = classify_error_types(diff, gold, pred, instance['text'])
        scores.append(len(diff) == 0)
        gold_metas.append(gold_meta)
        pred_metas.append(pred_meta)
        gold_narrs.append(gold_narr)
        pred_narrs.append(pred_narr)
        # errors.append(errs)
    acc = np.mean(scores) 
    make_sankey(gold_metas, pred_metas, gold_narrs, pred_narrs, model)


    # for i, (gold, pred) in enumerate([(gold_metas, pred_metas), (gold_narrs, pred_narrs)]):
    #     vectorizer = CountVectorizer(analyzer=lambda x: x, binary=True)
    #     vectorizer = vectorizer.fit(gold+pred)
    #     gold = vectorizer.transform(gold).toarray()
    #     pred = vectorizer.transform(pred).toarray()
    #     make_sankey(gold, pred, vectorizer, 'meta' if i == 0 else 'narrative', model)

   


    # plt.figure(figsize=(14, 10))
    # # Display confusion matrix as a heatmap
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues') #, xticklabels=vectorizer.get_feature_names_out(), yticklabels=vectorizer.get_feature_names_out())
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()


    
    # # breakpoint()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parse_args.add_argument("--split", type=str, default="test")
    parser.add_argument('--model', choices=['claude', 'gpt35', 'gpt4t', "gpt4", "phi2_ft"], default='gpt4t')
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    main(args.model)

    


    # what do i wanna know?
    # 1. what was missed (recall)
    # 2. what was wrong (precision)