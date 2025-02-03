import pandas as pd
from tqdm.auto import tqdm
import os
from pathlib import Path
from typing import List, Dict
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class NOWDataProcessor:
    def __init__(
        self,
        sentences_dir: str,
        samples_per_month: int = 10,
        start_year: int = 2012,
        end_year: int = 2023,
        random_seed: int = 0
    ):
        self.sentences_dir = Path(sentences_dir)
        self.samples_per_month = samples_per_month
        self.years = range(start_year, end_year)
        self.random_seed = random_seed
        self.months = [f"{m:02d}" for m in range(1, 13)]

    def load_monthly_data(self, year: int, month: str) -> pd.DataFrame:
        """Load and preprocess data for a specific year and month."""
        file_path = self.sentences_dir / str(year) / f'us_now_{year}_{month}.tsv'
        try:
            df = pd.read_csv(file_path, sep='\t')
            df = df[df.mention_flag == 1].drop(['mention_flag', 'unique_id'], axis=1)
            df['month'] = int(month)
            df['year'] = year
            return df.reset_index(drop=True)
        except Exception as e:
            logging.warning(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def combine_all_data(self) -> pd.DataFrame:
        """Combine data from all years and months."""
        dfs = []
        for year in tqdm(self.years, desc="Processing years"):
            for month in self.months:
                df = self.load_monthly_data(year, month)
                if not df.empty:
                    dfs.append(df)
        
        return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()

    def create_evaluation_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create evaluation splits with annotator assignments."""
        # Sample data per month
        eval_df = df.groupby(['year', 'month']).sample(
            n=self.samples_per_month,
            random_state=self.random_seed
        )
        eval_df = eval_df.sample(frac=1, random_state=self.random_seed)

        # Create annotator assignments
        agreement_data_len = 201
        eval_df_length = len(eval_df) - agreement_data_len
        
        # Initial assignments
        eval_df['assigned'] = (
            ["all"] * agreement_data_len +
            ["mh"] * int(eval_df_length * 1/3) +
            ["qz"] * int(eval_df_length * 1/3) +
            ["az"] * int(eval_df_length * 1/3)
        )

        # Create combined dataframe with agreement data
        annotator_dfs = []
        for annotator in ["mh", "qz", "az"]:
            annotator_df = pd.concat([
                eval_df[eval_df.assigned == annotator],
                eval_df[eval_df.assigned == "all"]
            ], axis=0)
            annotator_dfs.append(annotator_df)

        new_df = pd.concat(annotator_dfs, axis=0)
        
        # Update final assignments with asterisks for agreement data
        new_df['assigned'] = (
            ["mh"] * len(eval_df[eval_df.assigned == "mh"]) + ["mh*"] * agreement_data_len +
            ["qz"] * len(eval_df[eval_df.assigned == "qz"]) + ["qz*"] * agreement_data_len +
            ["az"] * len(eval_df[eval_df.assigned == "az"]) + ["az*"] * agreement_data_len
        )

        return new_df

def main():
    # Configuration
    config = {
        "sentences_dir": "/data/mourad/narratives/inflation/sentences",
        "samples_per_month": 10,
        "output_dir": "/data/mourad/narratives/inflation",
        "eval_output_dir": "output/annotation"
    }

    # Initialize processor
    processor = NOWDataProcessor(
        sentences_dir=config["sentences_dir"],
        samples_per_month=config["samples_per_month"]
    )

    # Process data
    logging.info("Loading and combining data...")
    combined_df = processor.combine_all_data()
    
    if combined_df.empty:
        logging.error("No data was loaded. Exiting.")
        return

    logging.info("Creating evaluation splits...")
    eval_df = processor.create_evaluation_splits(combined_df)

    # Save outputs
    logging.info("Saving results...")
    output_path = Path(config["output_dir"])
    eval_output_path = Path(config["eval_output_dir"])
    
    output_path.mkdir(parents=True, exist_ok=True)
    eval_output_path.mkdir(parents=True, exist_ok=True)

    combined_df.to_json(
        output_path / 'all_filtered.jsonl.gz',
        orient='records',
        lines=True,
        compression='gzip'
    )
    
    eval_df.to_csv(
        eval_output_path / f'annotate-{config["samples_per_month"]}_per_month.tsv',
        sep="\t",
        index=False
    )
    
    logging.info("Processing completed successfully!")

if __name__ == "__main__":
    main() 