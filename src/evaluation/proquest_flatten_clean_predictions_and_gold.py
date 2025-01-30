def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import re
import json
import pandas as pd
from src.utils import utils
import argparse
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from src.utils.utils import time, causes, effects, categories

# time = ['past', 'present', 'future', 'general']
# causes = ['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause']
# effects = ['cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect']

# categories = {
#     "cause_category": causes,
#     "effect_category": effects,
#     "contains-narrative": [
#         False, True
#     ],
#     "foreign": [
#           False, True
#     ],
#     "inflation-time": time,
#     "counter-narrative": [
#          False, True
#     ],
#     "cause_time": time,
#     "effect_time": time,
#     # "cause_effect": ['cause', 'effect'],
    
#     # "category": causes+effects
# }

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

        if isinstance(data['contains-narrative'], str):
            data['contains-narrative'] = "true" in data['contains-narrative'].lower()
        if isinstance(data['foreign'], str):
            data['foreign'] = "true" in data['foreign'].lower()
            
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

    df.foreign = df.foreign.astype(bool)
    df['contains-narrative'] = df['contains-narrative'].astype(bool)
    df['counter-narrative'] = df['counter-narrative'].astype(bool)
    df = df.fillna("", inplace=False)
    return df

def standardize_claude(data):
    global effects
    global causes
    
    narrs = []
    if isinstance(data, list):
        fmt_data = {'contains-narrative': True, 
                    'foreign': False,
                    'inflation-narratives': {'narratives': [],
                                             'inflation-time': 'general',
                                             'counter-narrative': False}}
        for n in data:
            n = n.strip()
            if n == 'none' and len(data) == 1:
                fmt_data['contains-narrative'] = False
            elif n in effects:
                narrs.append({'effect': n, 'time': 'general'})
            elif n in causes:
                narrs.append({'cause': n, 'time': 'general'})
        fmt_data['inflation-narratives']['narratives'] = narrs
        return fmt_data
    
    
    if data['contains-narrative'] == 'false':
        # breakpoint()
        data['inflation-narratives'] = None
        return data
        
    # narrs = []
    if 'effects' in data['inflation-narratives']:
        for e in data['inflation-narratives']['effects']:
            cat = re.findall(r"\[(.*?)\]", e[0])
            if cat:
                cat = cat[0]
                if cat in effects:
                    narrs.append({'effect': cat, 'time': e[1]})
        del data['inflation-narratives']['effects']
    if 'causes' in data['inflation-narratives']:
        for c in data['inflation-narratives']['causes']:
            cat = re.findall(r"\[(.*?)\]", c[0])
            if cat:
                cat = cat[0]
                if cat in causes:
                    narrs.append({'cause': cat, 'time': c[1]})
        del data['inflation-narratives']['causes']

    
    
    data['inflation-narratives']['narratives'] = narrs
    # breakpoint()
    return data
    # breakpoint()

def main(model, system, do_oracle, annotator, avg_type):
    prediction_data = utils.load_hf_dataset(model=model)
    # gold_data = utils.load_hf_dataset()[annotator]
    gold_data = utils.load_hf_dataset(path="/data/mourad/narratives/sft_data_proquest")[annotator]
    if system:
        phi2_pred_data = utils.load_hf_dataset(model="phi2_ft", binary=True)

    # for label, types in categories.items():
    #     print("Label:", label)
    #     gold_cats = []
    #     pred_cats = []
    category = []
    category_time = []
    contains_narrative = []
    general_time = []
    general_foreign = []
    general_counter = []
    wrong_preds = {}
    truth_disagree = pd.read_csv("../../data/eval/annotated/annotator_diffs.tsv", sep='\t')

    category_mlb = MultiLabelBinarizer().fit([categories['cause_category']+categories['effect_category']])
    time_mlb = MultiLabelBinarizer().fit([time])
    binary_mlb = MultiLabelBinarizer().fit([[True, False]])

    for inst_id, instance in enumerate(prediction_data):
        # breakpoint()
        gold = json.loads(utils.reconstruct_training_input(instance))
        # assert instance['text'] == gold_data[inst_id]['text']
        # gold = json.loads(utils.reconstruct_training_input(gold_data[inst_id]))
        # gold = json.loads(utils.reconstruct_training_input(instance))
        # breakpoint()
        # print(instance['completion'])
        # continue
        try:
            # pred = json.loads(clean_completion(instance['completion']))
            pred = clean_completion(instance['completion']).split(",")
            if system:
                phi2_pred = json.loads(phi2_pred_data[inst_id]['completion'])
                if phi2_pred['contains-narrative'] == "Yes":
                    phi2_pred = True
                else:
                    phi2_pred = False
        except Exception as e:
            print(e)
            breakpoint()
        
        # breakpoint()
        if model == 'claude':
            # pred = flatten_pred_json(instance['id'], pred)
            # pred = flatten_pred_json(instance['id'], pred)
            pred = standardize_claude(pred)
        # else:
        #     # pass
        breakpoint()
        pred_general, pred_narrative = flatten_gold_json(instance['id'], pred)
        if system:
            pass
        # pred['gold'] = 0
        gold_general, gold_narrative = flatten_gold_json(instance['id'], gold)

        # oracle
        if do_oracle:
            if not gold_general['contains-narrative'].iloc[0] or not pred_general['contains-narrative'].iloc[0]:
                continue

        pred_narrative = pred_narrative.drop_duplicates(subset=['category'])
        # if instance['id'] == 3446908:
        #     breakpoint()
        if set(pred_narrative.category.tolist()) != set(gold_narrative.category.tolist()):
            # print(pred_narrative)
            # print(gold_narrative)
            # breakpoint()
            if pred_narrative.category.iloc[0] == "":
                wrong_preds[instance['text']] = ['none']
            else:
                wrong_preds[instance['text']] = pred_narrative.category.tolist()
            # non_match_ids.append(instance['id'])
        # set(gold_narrative)

        cats = tuple(category_mlb.transform([gold_narrative.category.tolist(), pred_narrative.category.tolist()]))
        cats_time = tuple(time_mlb.transform([gold_narrative.time.tolist(), pred_narrative.time.tolist()]))
        # contains_narr = (int(gold_general['contains-narrative'].item()), int(pred_general['contains-narrative'].item()))
        contains_narr = tuple(binary_mlb.transform([gold_general['contains-narrative'].tolist(), pred_general['contains-narrative'].tolist()]))
        inflation_time = tuple(time_mlb.transform([gold_general['inflation-time'].tolist(), pred_general['inflation-time'].tolist()]))
        foreign = tuple(binary_mlb.transform([gold_general['foreign'].tolist(), pred_general['foreign'].tolist()]))
        counter_narrative = tuple(binary_mlb.transform([gold_general['counter-narrative'].tolist(), pred_general['counter-narrative'].tolist()]))

        category.append(cats)
        category_time.append(cats_time)
        contains_narrative.append(contains_narr)
        general_time.append(inflation_time)
        general_foreign.append(foreign)
        general_counter.append(counter_narrative)

    
    # truth_disagree = pd.read_csv("../../data/eval/annotated/annotator_diffs.tsv", sep='\t')
    # truth_disagree['pred'] = truth_disagree.sentence.apply(lambda x: wrong_preds.get(x, None))
    # truth_disagree['gold_alt'] = truth_disagree.apply(lambda x: set([it.strip() for it in f"{x.az}, {x.mh}, {x.qz}".split(",")]), axis=1)
    # truth_disagree.pred = truth_disagree.pred.fillna(-1)
    # print(truth_disagree[['pred', 'az', 'mh', 'qz']].head(50))
    # overlap = set(truth_disagree.sentence.tolist()).intersection(set(list(wrong_preds.keys())))
    # print(len(overlap))
    # breakpoint()

    # avg_type = None #'micro'
    if avg_type == 'none':
        avg_type = None
    cat_gold, cat_preds = list(zip(*category))
    cat_f1 = f1_score(np.array(cat_gold), np.array(cat_preds), average=avg_type)
    cat_time_gold, cat_time_preds = list(zip(*category_time))
    cat_time_f1 = f1_score(np.array(cat_time_gold), np.array(cat_time_preds), average=avg_type)
    contains_gold, contains_preds = list(zip(*contains_narrative))
    contains_f1 = f1_score(np.array(contains_gold), np.array(contains_preds), average=avg_type)
    time_gold, time_preds = list(zip(*general_time))
    time_f1 = f1_score(np.array(time_gold), np.array(time_preds), average=avg_type)
    foreign_gold, foreign_preds = list(zip(*general_foreign))
    foreign_f1 = f1_score(np.array(foreign_gold), np.array(foreign_preds), average=avg_type)
    counter_gold, counter_preds = list(zip(*general_counter))
    counter_f1 = f1_score(np.array(counter_gold), np.array(counter_preds), average=avg_type)

    if avg_type is None:
        df = pd.DataFrame({cls: cat_f1[i] for i, cls in enumerate(category_mlb.classes_)},  index=[f"f1_{model}"]).T
        df = df.sort_index()
        # breakpoint()
    else:
        df = pd.DataFrame({"contains_narrative": contains_f1, "foreign": foreign_f1, "time": time_f1, "counter_narrative": counter_f1, "category": cat_f1, "category_time": cat_time_f1}, index=[f"f1_{model}"]).T
    df = df.round(2)
    # breakpoint()
    cats_score = pd.DataFrame({"gold": category_mlb.inverse_transform(np.array(cat_gold)), "pred": category_mlb.inverse_transform(np.array(cat_preds))})
    cats_score[annotator] = cats_score.apply(lambda x: np.array_equal(x.gold, x.pred), axis=1)
    if not do_oracle:
        # cats_score = cats_score[[annotator]]
        cats_score['text'] = gold_data['text']
        # breakpoint()

    return df, cats_score
    # breakpoint()

        # enc.fit(X)
        # gold['gold'] = 1

        # df = pd.concat([gold, pred], axis=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parse_args.add_argument("--split", type=str, default="test")
    parser.add_argument('--model', choices=['claude', 'gpt35', 'gpt4t', "gpt4", "phi2_ft", "phi3_ft", "mistral_ft", "llama3_300steps", "all"], default='gpt4t')
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--system", action="store_true", help="use phi2 to screen for narratives then claude3 for classification")
    parser.add_argument("--oracle", action="store_true", help="only evaluate on instances where both models have a narrative")
    parser.add_argument("--annotator", type=str, default="test", choices=['test', 'test_az', 'test_qz', 'test_mh', 'all'])
    parser.add_argument('--avg_type', default='micro')
    args = parser.parse_args()

    if args.model == 'all':
        dfs = []
        if args.annotator == 'all':
            args.annotator = 'test'
        for m in ['gpt4t', "phi2_ft", "phi3_ft", "mistral_ft"]:
            print("MODEL: ", m)
            df, _ = main(m, args.system, args.oracle, args.annotator, args.avg_type)
            dfs.append(df)
        
        # if args.avg_type == 'none':
        #     df = pd.concat(dfs, axis=0)
        # else:
        df = pd.concat(dfs, axis=1)
        print(df)
        print(df.to_latex(float_format="%.2f"))
    elif args.annotator == 'all':
        dfs = []
        for a in ['test']: # ['test_az', 'test_qz', 'test_mh']:
            print("ANNOTATOR: ", a)
            df, cats_score = main(args.model, args.system, args.oracle, a, args.avg_type)
            dfs.append(cats_score)
        all_annotators = pd.concat(dfs, axis=1)
        # all_annotators['best'] = all_annotators.apply(lambda x: x.any(), axis=1)
        x = all_annotators[all_annotators['test'] == False]
        x[['text', 'pred', 'gold']]
        breakpoint()
        print(df)

    else:
        df, labels = main(args.model, args.system, args.oracle, args.annotator, args.avg_type)
       
        preds = utils.load_hf_dataset(model='claude')
        labels['pred'] = preds['completion']
        labels.pred = labels.pred.apply(lambda x: x.replace("effects: ", ""))
        labels.pred = labels.pred.apply(lambda x: x.replace("causes: ", ""))
        labels.pred = labels.pred.apply(lambda x: x.replace("none", ""))
        labels.pred = labels.pred.apply(lambda x: x.replace(" ", ""))
        labels['gold'] = labels.gold.apply(lambda x: ",".join([el for el in x]))
        labels['test'] = labels.gold == labels.pred
        breakpoint()
        for name, row in labels.iterrows():
            if not row.test:
                print(row.text)
                print("gold: ", row.gold)
                print("pred: ", row.pred)
                print()
                breakpoint()
        print(df)
