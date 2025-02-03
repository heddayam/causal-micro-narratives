"""
This script evaluates model predictions for inflation narrative classification tasks.
It processes model predictions and gold labels, computing various metrics including:
- Category classification (causes/effects)
- Time classification
- Narrative presence detection
- Foreign content classification

The script supports evaluation across different models, datasets, and annotation types,
with options for oracle evaluation and different averaging methods for F1 scores.

Examples:
    Evaluate Phi-2 model trained and tested on ProQuest data:
    ```
    python process_predictions_and_evaluate.py --model phi2_ft --test_ds proquest --train_ds proquest
    ```

    Evaluate Llama model trained on ProQuest and tested on NOW corpus:
    ```
    python process_predictions_and_evaluate.py --model llama31_ft_300s --test_ds NOW --train_ds proquest
    ```

Arguments:
    --model: Model to evaluate (phi2_ft, llama31_ft_300s, gpt4t, etc.)
    --test_ds: Test dataset (NOW, proquest, NOW_and_proquest)
    --train_ds: Training dataset (NOW, proquest, NOW_and_proquest)
    --oracle: Only evaluate on instances where both models have a narrative
    --annotator: Which annotator's labels to use (test, test_az, test_qz, test_mh)
    --avg_type: Type of averaging for F1 scores (micro, macro, none)
"""

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import re
import json
import pandas as pd
from src.utils import utils
import argparse
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from src.utils.utils import time, causes, effects, categories
import os
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

def flatten_json_to_df(json_input):
    # Handle both string and dict inputs
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError:
            # Clean up escaped quotes and try again
            json_input = json_input.replace('\\"', '"')
            data = json.loads(json_input)
    else:
        data = json_input
        
    # Extract metadata
    metadata = {
        'foreign': data.get('foreign', False),
        'contains-narrative': data.get('contains-narrative', False),
        'inflation-time': '',
        'inflation-direction': ''
    }
    
    # Extract inflation narratives data
    narratives_data = data.get('inflation-narratives', {})
    if narratives_data is not None:
        metadata['inflation-time'] = narratives_data.get('inflation-time', '')
        metadata['inflation-direction'] = narratives_data.get('inflation-direction', '')
        
        # Handle narratives
        narratives = narratives_data.get('narratives', [])
        if not narratives:
            # If no narratives, return dataframe with none category
            narrative_df = pd.DataFrame([{'category': 'none', 'type': '', 'time': ''}])
            return pd.DataFrame([metadata]), narrative_df
        
        # Process each narrative
        rows = []
        for narrative in narratives:
            # If narrative is a string, parse it
            if isinstance(narrative, str):
                try:
                    # Check if the string starts with '[' indicating it's a list
                    if narrative.startswith('['):
                        narratives_list = json.loads(narrative)
                        # Return first narrative from the list for now
                        narrative = narratives_list[0]
                    else:
                        # Try parsing as a single object first
                        try:
                            narrative = json.loads(narrative)
                        except json.JSONDecodeError:
                            # If it fails, try wrapping in brackets to parse as array
                            narrative = json.loads(f"[{narrative}]")[0]
                except json.JSONDecodeError:
                    breakpoint()
                    # Try cleaning up escaped quotes if initial parse fails
                    cleaned = narrative.replace('\\"', '"')
                    if cleaned.startswith('['):
                        narratives_list = json.loads(cleaned)
                        narrative = narratives_list[0]
                    else:
                        narrative = json.loads(f"[{cleaned}]")[0]
            
            row = {}
            # Add time from the narrative
            row['time'] = narrative.get('time', '')
            
            # Determine if it's a cause or effect and add appropriate columns
            if 'cause' in narrative:
                row['type'] = 'cause'
                row['category'] = narrative['cause']
            elif 'effect' in narrative:
                row['type'] = 'effect'
                row['category'] = narrative['effect']
            else:
                row['type'] = ''
                row['category'] = ''
            
            rows.append(row)
        
        # If no narratives, return dataframe with none category
        narrative_df = pd.DataFrame(rows)
        return pd.DataFrame([metadata]), narrative_df
    
    # If no inflation narratives or if it's None, return empty narrative DataFrame
    narrative_df = pd.DataFrame([{'category': 'none', 'type': '', 'time': ''}])
    return pd.DataFrame([metadata]), narrative_df

def compute_metrics(gold_narrative, pred_narrative, gold_general, pred_general, category_mlb, time_mlb, binary_mlb):
    """Compute all evaluation metrics for a single instance."""
    # Category metrics
    cats = tuple(category_mlb.transform([gold_narrative.category.tolist(), pred_narrative.category.tolist()]))
    cats_time = tuple(time_mlb.transform([gold_narrative.time.tolist(), pred_narrative.time.tolist()]))
    
    # General metrics
    contains_narr = tuple(binary_mlb.transform([gold_general['contains-narrative'].tolist(), pred_general['contains-narrative'].tolist()]))
    inflation_time = tuple(time_mlb.transform([gold_general['inflation-time'].tolist(), pred_general['inflation-time'].tolist()]))
    foreign = tuple(binary_mlb.transform([gold_general['foreign'].tolist(), pred_general['foreign'].tolist()]))
    
    return {
        'category': cats,
        'category_time': cats_time,
        'contains_narrative': contains_narr,
        'general_time': inflation_time,
        'general_foreign': foreign
    }

def calculate_f1_scores(metrics_list, category_mlb, avg_type=None):
    """Calculate F1 scores for all metric types."""
    # Unzip the metrics
    metrics_by_type = {
        metric_type: list(zip(*[m[metric_type] for m in metrics_list]))
        for metric_type in metrics_list[0].keys()
    }
    
    # Calculate F1 scores
    f1_scores = {}
    for metric_type, (gold, preds) in metrics_by_type.items():
        f1_scores[metric_type] = f1_score(np.array(gold), np.array(preds), average=avg_type)
    
    return f1_scores

def format_results(f1_scores, model, category_mlb, avg_type=None):
    """Format results into a DataFrame."""
    if avg_type is None:
        df = pd.DataFrame(
            {cls: f1_scores['category'][i] for i, cls in enumerate(category_mlb.classes_)},
            index=[f"f1_{model}"]
        ).T
        return df.sort_index()
    
    df = pd.DataFrame({
        "contains_narrative": f1_scores['contains_narrative'],
        "foreign": f1_scores['general_foreign'],
        "time": f1_scores['general_time'],
        "category": f1_scores['category'],
        "category_time": f1_scores['category_time']
    }, index=[f"f1_{model}"]).T
    
    return df.round(2)

def track_wrong_predictions(instance, pred_narrative, gold_narrative, wrong_preds):
    """Track instances where predictions don't match gold labels."""
    if set(pred_narrative.category.tolist()) != set(gold_narrative.category.tolist()):
        if pred_narrative.empty or pred_narrative.category.iloc[0] == "":
            wrong_preds[instance['text']] = ['none']
        else:
            wrong_preds[instance['text']] = pred_narrative.category.tolist()

def main(model, do_oracle, annotator, avg_type, train_ds, test_ds):
    """Main evaluation function."""
    # Initialize data and binarizers
    prediction_data = utils.load_model_preds(model, train_ds, test_ds)
    gold_data = utils.load_labeled_data(dataset=test_ds)[annotator]
    
    category_mlb = MultiLabelBinarizer().fit([categories['cause_category']+categories['effect_category']])
    time_mlb = MultiLabelBinarizer().fit([time])
    binary_mlb = MultiLabelBinarizer().fit([[True, False]])
    
    # Process predictions
    metrics_list = []
    wrong_preds = {}
    
    failed_to_parse = 0
    for instance in prediction_data:
        # Parse gold and predicted data
        try:
            # TODO fix now_and_proquest template processing
            # TODO add gpt support
            gold = json.loads(utils.reconstruct_training_input(instance))
            pred = clean_completion(instance['completion'])
        except Exception as e:
            print(e)
            failed_to_parse += 1
            continue
            
    
        pred_general, pred_narrative = flatten_json_to_df(pred)
        gold_general, gold_narrative = flatten_json_to_df(gold)
        
        # Skip if oracle mode and conditions not met
        if do_oracle and (not gold_general['contains-narrative'].iloc[0] or 
                         not pred_general['contains-narrative'].iloc[0]):
            continue
        
        # Remove duplicate categories
        pred_narrative = pred_narrative.drop_duplicates(subset=['category'])
        
        # Track wrong predictions
        track_wrong_predictions(instance, pred_narrative, gold_narrative, wrong_preds)
        
        # Compute metrics for this instance
        instance_metrics = compute_metrics(
            gold_narrative, pred_narrative, gold_general, pred_general,
            category_mlb, time_mlb, binary_mlb
        )
        metrics_list.append(instance_metrics)
    
    # Calculate and format results
    if avg_type == 'none':
        avg_type = None
    
    f1_scores = calculate_f1_scores(metrics_list, category_mlb, avg_type)
    results_df = format_results(f1_scores, model, category_mlb, avg_type)
    
    # Prepare category scores for detailed analysis
    cat_gold, cat_preds = list(zip(*[m['category'] for m in metrics_list]))
    cats_score = pd.DataFrame({
        "gold": category_mlb.inverse_transform(np.array(cat_gold)),
        "pred": category_mlb.inverse_transform(np.array(cat_preds))
    })
    cats_score[annotator] = cats_score.apply(lambda x: np.array_equal(x.gold, x.pred), axis=1)
    
    if not do_oracle:
        cats_score['text'] = gold_data['text']
    
    return results_df, cats_score, failed_to_parse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['gpt35', 'gpt4t', "gpt4", "phi2_ft", "llama31_ft_300s", "llama31_ft_600s", "all"], default='gpt4t')
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--oracle", action="store_true", help="only evaluate on instances where both models have a narrative")
    parser.add_argument("--annotator", type=str, default="test", choices=['test', 'test_az', 'test_qz', 'test_mh'])
    parser.add_argument('--avg_type', default='micro')
    parser.add_argument("--test_ds", type=str, choices=['NOW', 'proquest', 'NOW_and_proquest'], required=True)
    parser.add_argument("--train_ds", type=str, choices=['NOW', 'proquest', 'NOW_and_proquest'], required=True)
    args = parser.parse_args()

    if args.model == 'all':
        dfs = []
        for m in ["phi2_ft", "llama31_ft_300s", "gpt4t"]:
            print("MODEL: ", m)
            df, _ = main(m, args.oracle, args.annotator, args.avg_type, args.train_ds, args.test_ds)
            dfs.append(df)
        
        df = pd.concat(dfs, axis=1)
        print(df)
        print(df.to_latex(float_format="%.2f"))

    else:
        df, labels, failed_to_parse = main(args.model, args.oracle, args.annotator, args.avg_type, args.train_ds, args.test_ds)
       
        print("MODEL: ", args.model)
        print("ANNOTATOR: ", args.annotator)
        print("F1 AVERAGING: ", args.avg_type)
        print("TRAIN DATASET: ", args.train_ds)
        print("TEST DATASET: ", args.test_ds)
        print("FAILED TO PARSE: ", failed_to_parse)
        print(df)

        # save results
        out_dir = "output/gold_vs_preds"
        os.makedirs(out_dir, exist_ok=True)
        labels.to_pickle(f"{out_dir}/{args.model}_{args.avg_type}_{args.train_ds}_{args.test_ds}.pkl")

