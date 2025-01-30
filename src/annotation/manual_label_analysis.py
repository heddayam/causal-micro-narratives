import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from itertools import combinations
import nltk
from nltk.metrics import agreement
from nltk.metrics.distance import masi_distance
from src.utils import utils
import json
import argparse
import datasets

def confusion(ref, pred):
    cm = confusion_matrix(ref, pred)
    plt.figure(figsize=(14, 10))
    # Display confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(df.label_manual), yticklabels=np.unique(df.label))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    plt.savefig(f"/data/mourad/narratives/categories/eval/{args.model}_effect_confusion_matrix.png", dpi=300)

# def calculate_masi_distance(data):
#     # calculate pairwise distances between all annotators
#     distances = []
#     breakpoint()
#     for pair in combinations(range(3), 2):
#         d = masi_distance(set(data[pair[0]]),
#                         set(data[pair[1]]))
#         distances.append(d)


#     weights = [1 - d for d in distances]
#     alpha_value = alpha(reliability_data=data, distance=masi_distance, weights=weights)
#     # calculate average distance
#     avg_distance = np.mean(distances)
#     return avg_distance
    
def manual_compare(df):
    df = df.groupby('id')
    records = []
    for name, group in df:
        # if group.narrative.nunique() > 1:
        #     breakpoint()
        #     record[]
        # record = {row['assigned'][:-1]:row['category'] for name, row in group.iterrows()}
        # record.update({f"{row['assigned'][:-1]}_type":row['(c)ause/(e)ffect'] for name, row in group.iterrows()})
        record = {row['assigned'][:-1]:", ".join(list(row['narrative'])) for name, row in group.iterrows()}
        record['id'] = name
        record['sentence'] = group.iloc[0]['text']
        records.append(record)
    df = pd.DataFrame(records)
    

    diffs = df[(df.mh != df.az) | (df.qz != df.mh) | (df.az != df.qz)]
    diffs.to_csv("../../data/eval/annotated/annotator_diffs_proquest.tsv", sep='\t', index=False)
    breakpoint()

def make_comparison_set(annotation, binary=False):
    # breakpoint()
    if annotation['contains-narrative'] == False:
        return frozenset(['none'])
    elif binary:
        return frozenset(['narrative'])
    
    data = []
    if annotation['inflation-narratives'] is None:
        breakpoint()
    for narr in annotation['inflation-narratives']['narratives']:
        data.append(list(narr.values())[0])

    return frozenset(data)

def calc_human_f1(ds):
    az = ds['test_az'].to_pandas()
    qz = ds['test_qz'].to_pandas()
    mh = ds['test_mh'].to_pandas()
    gold = json.loads(utils.reconstruct_training_input(gold_data[inst_id]))
    breakpoint()


def main(split=None):
    # df = pd.read_csv('../../data/eval/annotated/eval-5_per_month-annotated.tsv', sep='\t')
    # df = df[df.assigned.str.endswith("*")]
    # df.category = df.category.str.replace("~", "")

    # ds = utils.load_hf_dataset(path="/data/mourad/narratives/annotation_data", split=split)
    # ds = utils.load_hf_dataset(path=" vsft_data_proquest")#, split=split)
    ds = utils.load_hf_dataset(path=f"/data/mourad/narratives/sft_data", dataset='now_and_proquest')
    calc_human_f1(ds)
    breakpoint()
    
    if split is None:
        mh = ds['test_mh'].to_pandas().sort_values(by='text')
        mh['id'] = range(len(mh))
        qz = ds['test_qz'].to_pandas().sort_values(by='text')
        qz['id'] = range(len(qz))
        az = ds['test_az'].to_pandas().sort_values(by='text')
        az['id'] = range(len(az))
        df = pd.concat([mh, qz, az], axis=0)
        # breakpoint()
    else:
        df = ds.to_pandas()
        
    df['og'] = df.apply(utils.reconstruct_training_input, axis=1)
    df['og'] = df.og.apply(json.loads)
    
    breakpoint()
    df['narrative'] = df.og.apply(lambda x:make_comparison_set(x, binary=True))

    # breakpoint()
    tmp = df.explode("narrative")
    tmp = tmp.drop_duplicates(subset=['text'])
    # ds = datasets.Dataset.from_pandas(tmp, preserve_index=False)
    # ds.save_to_disk(f"/data/mourad/narratives/narrative_category_data/{split}")

    # manual_compare(df.copy())
        
    
    
    # df = df[df.assigned.isin(['az*', 'qz*'])]
    for annotation in ['narrative']: #, 'category']:
        # if annotation == 'category':
        #     df = df.dropna(subset=[annotation])
        # df[annotation] = df[annotation].fillna("na")
        # df[annotation] = df[annotation].apply(lambda x: frozenset(x.split(',')))
        df = df.sort_values(by='id')

        data = df[['assigned', 'id', annotation]].to_records(index=False).tolist()
        # breakpoint()
        masi_task = nltk.AnnotationTask(distance=masi_distance)
        masi_task.load_array(data)

        print(masi_task.alpha())
        

        


    # breakpoint()



   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, required=False, choices=['train', 'test', 'dev'])
    args = parser.parse_args()

    main(args.split)