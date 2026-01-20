"""
Process batch predictions from language models and prepare data for analysis.

This script processes predictions from language models on news articles,
extracts narrative categories, and prepares the data for downstream analysis.

Usage:
    conda activate mourad-econ-py310
    python -m src.finetuning.process_batched_predictions \
        --dataset full_proquest \
        --model llama31_ft__600s \
        --train_ds now_and_proquest \
        --updated

Note: If predictions were generated on a remote cluster, transfer them first:
    scp -r mourad@fe01.ds.uchicago.edu:/net/projects/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/llama31_ft__600s_train-now_and_proquest_*_2010-2025_updated /home/mourad/causal-micro-narratives/tmp_batched/
"""

import argparse
import json
import logging
import re
from glob import glob
from os import path
from typing import List
import swifter

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm

from src.utils import utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Substrings to match against title for national news classification
# Uses case-insensitive substring matching (e.g., "Wall Street Journal" matches "Wall Street Journal (Online)")
national_news_outlets = [
    "New York Times",
    "Wall Street Journal",
    "Washington Post",
    "USA Today",
    "Los Angeles Times",
    "Christian Science Monitor",
    "Newsday",
    "U.S. Newswire",
    "Voice of America",
    "Politico",
    "ProPublica",
    "New York Post",
    "NPR",
    "PR Newswire",
    "Business Wire",
    "Knight Ridder",
    "McClatchy",
    "Gannett News Service",
    "Targeted News Service",
    "Newhouse News Service",
    "Cox News Service",
    "University Wire",
    "Noticias",
    "Religion News Service",
    "CNN",
    "Axios",
    "Barron's",
    "Insider",
    # NPR shows (titles don't contain "NPR")
    "Morning Edition",
    "All Things Considered",
    "Planet Money",
]


def is_national_outlet(title: str) -> bool:
    """Check if title matches any national news outlet (case-insensitive substring match)."""
    if pd.isna(title):
        return False
    title_lower = title.lower()
    return any(outlet.lower() in title_lower for outlet in national_news_outlets)
def process_dataset_predictions(
    dataset_name: str,
    model: str,
    train_ds: str,
    updated: bool = False
) -> pd.DataFrame:
    """
    Process model predictions for a specific dataset.

    Args:
        dataset_name: Name of the dataset ('proquest', 'now', 'fed', or 'full_fed')
        model: Name of the model used for predictions
        train_ds: Training dataset identifier
        updated: If True, use the 2010-2025_updated prediction paths

    Returns:
        DataFrame containing processed predictions
    """
    logger.info(f"Processing predictions for {dataset_name} dataset")

    # Determine paths based on dataset type
    if dataset_name.lower() in ['fed', 'full_fed']:
        # Fed data paths - check both remote and local
        dir_base = "/net/projects/chai-lab/mourad/narratives-data/model_json_preds/fed/full_fed"
        batch_files = glob(path.join(dir_base, f"{model}_train-{train_ds}_sample_*"))
        # Also check local tmp_batched for transferred files
        if not batch_files:
            dir_base = "/home/mourad/causal-micro-narratives/tmp_batched"
            batch_files = glob(path.join(dir_base, f"{model}_train-{train_ds}_sample_*_fed"))
    elif updated:
        dir_base = f"/home/mourad/causal-micro-narratives/tmp_batched"
        batch_files = glob(path.join(dir_base, f"{model}_train-{train_ds}_sample_*_2010-2025_updated"))
    else:
        dir_base = f"/data/mourad/narratives/model_json_preds/{dataset_name}"
        batch_files = glob(path.join(dir_base, f"{model}_train-{train_ds}_sample_*"))

    if not batch_files:
        logger.warning(f"No prediction files found for {dataset_name}")
        return None

    all_dfs = []
    for batch_file in tqdm(batch_files, desc=f"Processing {dataset_name} batches"):
        # Load and process each batch
        dataset = load_from_disk(batch_file)
        dataset = dataset.map(lambda e: {'prediction': json.loads(e['completion'])})

        data = dataset.to_pandas()
        data['lens'] = data['text'].apply(lambda x: len(x.split()))
        data = data[data['lens'] <= utils.PROQUEST_MAX_LEN]
        all_dfs.append(data)

    # Combine all batches
    dataset_df = pd.concat(all_dfs)
    dataset_df['source'] = dataset_name

    # Clean up columns based on dataset
    if dataset_name.lower() == 'now':
        dataset_df = dataset_df.rename({'id': 'file_id'}, axis=1)
        dataset_df['national'] = dataset_df['title'].apply(is_national_outlet)
    elif dataset_name.lower() in ['fed', 'full_fed']:
        # Fed data doesn't have title/city/state columns like ProQuest
        # Set defaults for consistency with downstream processing
        if 'title' not in dataset_df.columns:
            dataset_df['title'] = dataset_df.get('doc_type', 'Federal Reserve')
        if 'city' not in dataset_df.columns:
            dataset_df['city'] = 'washington'
        if 'state' not in dataset_df.columns:
            dataset_df['state'] = 'district of columbia'
        if 'region' not in dataset_df.columns:
            dataset_df['region'] = 'northeast'
        # Fed documents are inherently national
        dataset_df['national'] = True
    else:
        # ProQuest and other news data
        dataset_df['national'] = dataset_df['title'].apply(is_national_outlet)

    dataset_df = dataset_df.drop(['lens'], axis=1)
    return dataset_df

def extract_narrative_details(pred):
    """
    Extract detailed narrative information from prediction.

    Returns:
        tuple: (inflation_time, inflation_direction, list of (narrative_type, narrative_time) tuples)
    """
    if not pred.get('contains-narrative') or not pred.get('inflation-narratives'):
        return None, None, []

    infl_narr = pred['inflation-narratives']
    inflation_time = infl_narr.get('inflation-time')
    inflation_direction = infl_narr.get('inflation-direction')

    narratives_with_time = []
    raw_narratives = infl_narr.get('narratives', [])

    if len(raw_narratives) > 0:
        narrs = raw_narratives[0]
        if isinstance(narrs, str):
            narrs = re.sub(r'\s', '', narrs)
            try:
                narrs = eval(narrs)
            except:
                return inflation_time, inflation_direction, []

        if isinstance(narrs, (tuple, list)):
            for narr in narrs:
                if isinstance(narr, dict):
                    # Get narrative type (cause or effect value)
                    narr_type = narr.get('cause') or narr.get('effect')
                    narr_time = narr.get('time')
                    if narr_type and narr_type in utils.causes + utils.effects:
                        narratives_with_time.append((narr_type, narr_time))
        elif isinstance(narrs, dict):
            narr_type = narrs.get('cause') or narrs.get('effect')
            narr_time = narrs.get('time')
            if narr_type and narr_type in utils.causes + utils.effects:
                narratives_with_time.append((narr_type, narr_time))

    return inflation_time, inflation_direction, narratives_with_time


def process_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and process narrative categories from predictions."""
    logger.info("Processing narrative categories...")

    # Extract detailed narrative info
    df['_narrative_details'] = df['prediction'].swifter.apply(extract_narrative_details)
    df = df[~df._narrative_details.isna()]

    # Unpack the details
    df['inflation_time'] = df['_narrative_details'].apply(lambda x: x[0] if x else None)
    df['inflation_direction'] = df['_narrative_details'].apply(lambda x: x[1] if x else None)
    df['narrative_with_time'] = df['_narrative_details'].apply(lambda x: x[2] if x else [])
    df = df.drop('_narrative_details', axis=1)

    # For backward compatibility, also extract just narrative types
    df['narrative'] = df['narrative_with_time'].apply(
        lambda x: [n[0] for n in x] + (['contains_narrative'] if len(x) > 0 else [])
    )
    df['contains'] = df['narrative'].apply(lambda x: len(x) > 0).astype(int)

    # Filter out specific cities if needed
    df = df[~df.city.isin(['missoula, vermillion'])]

    if '__index_level_0__' in df.columns:
        df = df.drop('__index_level_0__', axis=1)

    return df

def prepare_final_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare final datasets for analysis (sentence-level, not aggregated)."""
    logger.info("Preparing final datasets...")

    # Explode on narrative_with_time to get (narrative_type, narrative_time) per row
    analysis_df = df.explode('narrative_with_time')

    # Extract narrative type and time from tuples
    # For rows without narratives (empty list becomes None after explode), set to 'none'
    analysis_df['narrative'] = analysis_df['narrative_with_time'].apply(
        lambda x: x[0] if isinstance(x, tuple) else 'none'
    )
    analysis_df['narrative_time'] = analysis_df['narrative_with_time'].apply(
        lambda x: x[1] if isinstance(x, tuple) else None
    )
    analysis_df = analysis_df.drop('narrative_with_time', axis=1)

    # Remove 'contains_narrative' marker rows (keep 'none' rows for sentences without narratives)
    analysis_df = analysis_df[analysis_df.narrative != 'contains_narrative']

    # Deduplicate same text with same narrative in same time/location
    rows_before = len(analysis_df)
    dedup_cols = ['text', 'year', 'month', 'city', 'state', 'narrative']
    dedup_cols = [c for c in dedup_cols if c in analysis_df.columns]
    analysis_df = analysis_df.drop_duplicates(subset=dedup_cols)
    rows_removed = rows_before - len(analysis_df)
    logger.info(f"Deduplicated {rows_removed} rows ({rows_removed/rows_before*100:.2f}%) - same text+narrative in same time/location")

    # Clean up columns - keep year_month, title, loc, inflation_time, inflation_direction, narrative_time
    drop_cols = ['file_id', 'contains', 'completion', 'prediction', 'text', 'source']
    drop_cols = [c for c in drop_cols if c in analysis_df.columns]
    analysis_df = analysis_df.drop(drop_cols, axis=1)

    # Create dummy variables for narrative types (sentence-level, no aggregation)
    analysis_df = pd.get_dummies(
        analysis_df,
        prefix="",
        prefix_sep="",
        columns=['narrative'],
        dtype=int
    )

    return df, analysis_df

def main(dataset: str, model: str, train_ds: str, updated: bool = False):
    """Main function to process batch predictions."""
    logger.info(f"Starting batch prediction processing for {model} on {dataset}")

    # Process predictions for each dataset type
    all_predictions = []
    for dataset_name in [dataset.lower()]: #['proquest', 'now']:
        df = process_dataset_predictions(dataset_name, model, train_ds, updated=updated)
        if df is not None:
            all_predictions.append(df)

    if not all_predictions:
        logger.error("No predictions were processed successfully")
        return

    # Combine and process all predictions
    combined_df = pd.concat(all_predictions)
    processed_df = process_narratives(combined_df)

    # Prepare final datasets
    full_df, analysis_df = prepare_final_data(processed_df)

    # Determine output paths based on dataset type
    if dataset.lower() in ['fed', 'full_fed']:
        full_data_path = "/data/mourad/narratives/regression_data/fed_data_llama_preds.csv"
        analysis_path = "/data/mourad/narratives/regression_data/fed_data_llama_preds_for_regression.csv"
    else:
        full_data_path = "/data/mourad/narratives/regression_data/all_news_data_llama_preds.csv"
        analysis_path = "/data/mourad/narratives/regression_data/all_news_data_llama_preds_for_regression.csv"

    # Save full dataset
    full_df.to_csv(full_data_path, index=False)
    logger.info(f"Saved full dataset to {full_data_path}")

    # Save analysis dataset
    analysis_df.to_csv(analysis_path, index=False)
    logger.info(f"Saved analysis dataset to {analysis_path}")

    logger.info("Batch prediction processing completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process batch predictions from language models")
    parser.add_argument(
        "--dataset",
        choices=['NOW', 'PROQUEST', 'full_proquest', 'FED', 'full_fed'],
        help="Dataset to combine batch predictions for",
        required=True
    )
    parser.add_argument(
        "--model",
        choices=['phi2', 'llama31_ft__600s'],
        default='llama31_ft__600s',
        help="Model to combine batch predictions for"
    )
    parser.add_argument(
        '--train_ds',
        choices=['now', 'proquest', 'now_and_proquest'],
        default='now_and_proquest',
        help="Training dataset of model used"
    )
    parser.add_argument(
        '--updated',
        action='store_true',
        help="Use the 2010-2025_updated prediction paths"
    )

    args = parser.parse_args()
    main(args.dataset, args.model, args.train_ds, updated=args.updated) 