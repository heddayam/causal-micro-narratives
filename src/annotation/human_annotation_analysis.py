"""
Analyze inter-annotator agreement and differences between human annotations of narratives.

This module provides functionality to:
1. Load and process human annotations from multiple annotators
2. Calculate inter-annotator agreement scores using MASI distance
3. Compare annotations between annotators to identify differences

The analysis can be run in binary mode (narrative vs non-narrative) or with full narrative 
categorization. Results are saved to output files for further analysis.

Example usage:
    python human_annotation_analysis.py --binary --split test
"""


import argparse
import json
from typing import Dict, FrozenSet, List
import nltk
import pandas as pd
from nltk.metrics import agreement
from nltk.metrics.distance import masi_distance

from src.utils import utils

def compare_annotator_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare annotations between different annotators and save differences.
    
    Args:
        df: DataFrame containing annotations
        output_path: Path to save the differences
    
    Returns:
        DataFrame containing only the differing annotations
    """
    df = df.groupby('id')
    records = []
    
    for name, group in df:
        record = {
            row['assigned'][:-1]: ", ".join(list(row['narrative'])) 
            for _, row in group.iterrows()
        }
        record['id'] = name
        record['sentence'] = group.iloc[0]['text']
        records.append(record)
    
    result_df = pd.DataFrame(records)
    diffs = result_df[
        (result_df.mh != result_df.az) | 
        (result_df.qz != result_df.mh) | 
        (result_df.az != result_df.qz)
    ]
    return diffs


def process_annotation(annotation: Dict, binary: bool = False) -> FrozenSet[str]:
    """
    Process a single annotation into a comparable format.
    
    Args:
        annotation: Dictionary containing annotation data
        binary: If True, return binary narrative/non-narrative classification
    
    Returns:
        FrozenSet of narrative categories
    """
    if not annotation['contains-narrative']:
        return frozenset(['none'])
    
    if binary:
        return frozenset(['narrative'])
    
    if annotation['inflation-narratives'] is None:
        return frozenset()
        
    narratives = []
    for narr in annotation['inflation-narratives']['narratives']:
        narratives.append(list(narr.values())[0])
    
    return frozenset(narratives)


def calculate_agreement(df: pd.DataFrame, annotation_column: str = 'narrative') -> float:
    """
    Calculate inter-annotator agreement using MASI distance.
    
    Args:
        df: DataFrame containing annotations
        annotation_column: Column name containing the annotations to compare
    
    Returns:
        Alpha agreement score
    """
    df = df.sort_values(by='id')
    data = df[['assigned', 'id', annotation_column]].to_records(index=False).tolist()
    
    masi_task = nltk.AnnotationTask(distance=masi_distance)
    masi_task.load_array(data)
    return masi_task.alpha()


def load_and_prepare_data(dataset_path: str, binary: bool, split: str) -> pd.DataFrame:
    """
    Load and prepare the dataset for analysis.
    
    Args:
        dataset_path: Path to the dataset
        binary: Whether to use binary narrative/non-narrative classification
        split: Optional split to load (train/test/dev)
    
    Returns:
        Prepared DataFrame
    """
    ds = utils.load_hf_dataset(path=dataset_path, dataset='now_and_proquest')
    
    if split is None:
        # Combine test sets from different annotators
        dfs = []
        for annotator in ['mh', 'qz', 'az']:
            df = ds[f'test_{annotator}'].to_pandas().sort_values(by='text')
            df['id'] = range(len(df))
            dfs.append(df)
        df = pd.concat(dfs, axis=0)
    else:
        df = ds.to_pandas()
    
    # Process annotations
    df['og'] = df.apply(utils.reconstruct_training_input, axis=1)
    df['og'] = df.og.apply(json.loads)
    df['narrative'] = df.og.apply(lambda x: process_annotation(x, binary=binary))
    
    return df


def main(binary: bool, split: str):
    """
    Main function to run the annotation analysis.
    
    Args:
        binary: Whether to use binary narrative/non-narrative classification
        split: Optional dataset split to analyze
    """
    dataset_path = "/data/mourad/narratives/sft_data"
    output_path = "output/"
    df = load_and_prepare_data(dataset_path, binary, split)
    
    # Calculate and print agreement score
    agreement_score = calculate_agreement(df)
    print(f"Inter-annotator agreement score: {agreement_score:.3f}")
    
    # Compare annotator outputs
    diffs = compare_annotator_outputs(df)
    diffs.to_csv(output_path + "annotator_diffs.tsv", sep='\t', index=False)
    print(f"Found {len(diffs)} differing annotations, saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze human annotations for narrative detection.')
    parser.add_argument("--split", type=str, required=False, 
                      choices=['train', 'test', 'dev'],
                      help="Dataset split to analyze")
    parser.add_argument("--binary", action='store_true',
                      help="Whether to use binary narrative/non-narrative classification")
    args = parser.parse_args()
    
    main(args.binary, args.split) 