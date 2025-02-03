"""
Process batch predictions from language models and prepare data for analysis.

This script processes predictions from language models on news articles,
extracts narrative categories, and prepares the data for downstream analysis.
"""

import argparse
import json
import logging
from glob import glob
from os import path
from typing import List

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

def process_dataset_predictions(
    dataset_name: str,
    model: str,
    train_ds: str
) -> pd.DataFrame:
    """
    Process model predictions for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset ('proquest' or 'now')
        model: Name of the model used for predictions
        train_ds: Training dataset identifier
    
    Returns:
        DataFrame containing processed predictions
    """
    logger.info(f"Processing predictions for {dataset_name} dataset")
    
    dir_base = f"/data/mourad/narratives/model_json_preds/full_{dataset_name}"
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
    if dataset_name == 'now':
        dataset_df = dataset_df.rename({'id': 'file_id'}, axis=1)
    if dataset_name == 'proquest':
        dataset_df = dataset_df.drop(['year_month', 'title', 'loc'], axis=1)
    
    dataset_df = dataset_df.drop(['lens'], axis=1)
    return dataset_df

def process_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and process narrative categories from predictions."""
    logger.info("Processing narrative categories...")
    
    df['narrative'] = df['prediction'].apply(utils.extract_narrative_category)
    df = df[~df.narrative.isna()]
    df['contains'] = df['narrative'].apply(lambda x: len(x) > 0).astype(int)
    
    # Filter out specific cities if needed, these are manually selected. They have too few data points to be useful.
    df = df[~df.city.isin(['missoula, vermillion'])]
    
    if '__index_level_0__' in df.columns:
        df = df.drop('__index_level_0__', axis=1)
    
    return df

def prepare_final_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare final datasets for analysis."""
    logger.info("Preparing final datasets...")
    
    # Prepare data for analysis
    analysis_df = df.explode('narrative')
    analysis_df = analysis_df[analysis_df.narrative != 'contains_narrative']
    analysis_df.narrative = analysis_df.narrative.fillna('none')
    
    # Clean up columns
    drop_cols = ['file_id', 'contains', 'completion', 'prediction']
    analysis_df = analysis_df.drop(drop_cols, axis=1)
    analysis_df = analysis_df.drop_duplicates()
    
    # Create dummy variables and aggregate
    meta_cols = analysis_df.columns.drop('narrative').tolist()
    analysis_df = pd.get_dummies(
        analysis_df, 
        prefix="", 
        prefix_sep="", 
        columns=['narrative'], 
        dtype=int
    )
    
    analysis_df = analysis_df.groupby(meta_cols, sort=False, dropna=False).sum().reset_index()
    
    return df, analysis_df

def main(dataset: str, model: str, train_ds: str):
    """Main function to process batch predictions."""
    logger.info(f"Starting batch prediction processing for {model} on {dataset}")
    
    # Process predictions for each dataset type
    all_predictions = []
    for dataset_name in ['proquest', 'now']:
        df = process_dataset_predictions(dataset_name, model, train_ds)
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
    
    # Save full dataset
    full_data_path = "/data/mourad/narratives/regression_data/all_news_data_llama_preds.csv"
    full_df.to_csv(full_data_path)
    logger.info(f"Saved full dataset to {full_data_path}")

    # Save analysis dataset
    analysis_path = "/data/mourad/narratives/regression_data/all_news_data_llama_preds_for_regression.csv"
    analysis_df.drop('text', axis=1).to_csv(analysis_path)
    logger.info(f"Saved analysis dataset to {analysis_path}")
    
    logger.info("Batch prediction processing completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process batch predictions from language models")
    parser.add_argument(
        "--dataset",
        choices=['NOW', 'PROQUEST', 'full_proquest'],
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
    
    args = parser.parse_args()
    main(args.dataset, args.model, args.train_ds) 