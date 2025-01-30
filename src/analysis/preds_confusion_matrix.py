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
    
def make_cm(gold, pred, vectorizer, type, model):
    cm = multilabel_confusion_matrix(gold, pred) 
    if type == 'meta':
        keep = ['foreign', 'has_narrative']
    else:
        # keep = [f for f in vectorizer.get_feature_names_out() if '-' not in f]
        feats = list(vectorizer.get_feature_names_out())
        keep_time = ['present', 'past', 'future', 'general']
        keep = set(feats).difference(keep_time)
        keep = [f for f in keep if '-' not in f]
        # breakpoint()
    f, axes = plt.subplots(math.ceil(len(keep)/5), len(keep) if len(keep) < 5 else 5, figsize=(25, 15))
    axes = axes.ravel()
    # for i in range(4):
    kept_idx = 0
    for i, feature in enumerate(vectorizer.get_feature_names_out()):
        if feature not in keep: continue
        # breakpoint()
        disp = ConfusionMatrixDisplay(cm[i])#, display_labels=[0, i]) # cm(gold_metas[:, i], pred_metas[:, i]),                  
        disp.plot(ax=axes[kept_idx], values_format='.4g')
        disp.ax_.set_title(f'{feature}')
        if kept_idx<12 and len(keep) > 5:
            disp.ax_.set_xlabel('')
        if kept_idx%5!=0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()
        kept_idx+=1

    plt.subplots_adjust(wspace=0.10, hspace=0.3)
    f.colorbar(disp.im_, ax=axes)

    plt.savefig(f"/data/mourad/narratives/plots/confusion_matrix_{type}_{model}.png", dpi=300)
    plt.clf()

def main(model):
    prediction_data = utils.load_hf_dataset(model=model)
    scores = []
    narrative_scores = []
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
        if pred_meta[1] == 'has_narrative':
            narrative_scores.append(len(diff) == 0)
            if len(diff) != 0:
                print(instance['text'])
                print(gold)
                print(pred)
                print(diff)
                print()
        gold_metas.append(gold_meta)
        pred_metas.append(pred_meta)
        gold_narrs.append(gold_narr)
        pred_narrs.append(pred_narr)
        # errors.append(errs)
    acc = np.mean(scores) 
    narrative_acc = np.mean(narrative_scores)
    print("general acc = ", acc)
    print("narrative acc = ", narrative_acc)
    # for i, (gold, pred) in enumerate([(gold_metas, pred_metas), (gold_narrs, pred_narrs)]):
    #     vectorizer = CountVectorizer(analyzer=lambda x: x, binary=True)
    #     vectorizer = vectorizer.fit(gold+pred)
    #     gold = vectorizer.transform(gold).toarray()
    #     pred = vectorizer.transform(pred).toarray()
    #     make_cm(gold, pred, vectorizer, 'meta' if i == 0 else 'narrative', model)

   


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