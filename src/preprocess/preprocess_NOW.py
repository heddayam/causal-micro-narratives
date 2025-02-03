"""Script for preprocessing NOW corpus data."""

import logging
import os
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import spacy
import swifter
from spacy.language import Language

from src.utils import utils

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

class NOWPreprocessor:
    """Handles preprocessing of NOW corpus data."""

    def __init__(self, nlp_model: str = "en_core_web_sm"):
        """Initialize the preprocessor with spaCy model.

        Args:
            nlp_model: Name of spaCy model to use
        """
        self.nlp = self._setup_nlp(nlp_model)

    @staticmethod
    def _setup_nlp(model_name: str) -> Language:
        """Set up spaCy NLP pipeline.

        Args:
            model_name: Name of spaCy model to use

        Returns:
            Configured spaCy pipeline
        """
        try:
            nlp = spacy.load(model_name)
        except OSError:
            spacy.cli.download(model_name)
            nlp = spacy.load(model_name)
        nlp.add_pipe("sentencizer")
        nlp.select_pipes(enable=["sentencizer"])
        return nlp

    def _get_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using spaCy.

        Args:
            text: Input text to split into sentences

        Returns:
            List of sentence strings
        """
        sentences = []
        # Process text in chunks to handle large documents
        chunk_size = 1000000
        
        if len(text) > chunk_size:
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                sentences.extend(self._get_sentences(chunk))
            return sentences
        
        doc = self.nlp(text)
        return [
            sent.text.strip()
            for sent in doc.sents
            if sent.text.strip() and '@' not in sent.text
        ]

    def preprocess_file(
        self,
        filepath: str,
        sentence_only: bool = False,
        article_only: bool = False,
        pre1920: bool = False
    ) -> Optional[pd.DataFrame]:
        """Preprocess a single NOW corpus file.

        Args:
            filepath: Path to input file
            sentence_only: If True, return only processed sentences
            article_only: If True, return only raw articles
            pre1920: Flag for pre-1920 inflation check

        Returns:
            Processed DataFrame or None if processing fails
        """
        try:
            df = pd.read_table(
                filepath,
                sep="\t",
                encoding="ISO-8859-1",
                header=None,
                on_bad_lines="warn"
            )
            
            if article_only:
                return df

            df.columns = ['text']
            df['text'] = utils.filter_now(df['text'])
            
            # Extract IDs and clean text
            res = df.text.swifter.apply(utils.extract_id)
            ids, texts = zip(*res)
            df['id'] = ids
            df['text'] = texts

            # Process sentences
            df['text'] = df.text.apply(self._get_sentences)
            df = df.explode('text').reset_index(drop=True)
            df = df.dropna()

            if sentence_only:
                return df

            # Process surrounding context
            df = df.groupby('id').apply(
                lambda x: self._get_surrounding_rows(x, pre1920)
            )
            
            return df.reset_index(drop=True) if df is not None else None

        except Exception as e:
            logger.error(f"Error processing {filepath}: {str(e)}")
            return None

    def _get_surrounding_rows(
        self,
        df: pd.DataFrame,
        pre1920: bool = False
    ) -> Optional[pd.DataFrame]:
        """Get surrounding context for inflation mentions.

        Args:
            df: Input DataFrame
            pre1920: Flag for pre-1920 inflation check

        Returns:
            DataFrame with surrounding context or None
        """
        df = df.reset_index(drop=True)
        mask = df['text'].apply(lambda x: utils.check_inflation(x, pre1920=pre1920))
        
        to_keep = []
        for i, has_mention in enumerate(mask):
            if has_mention:
                context_range = range(max(0, i - 2), min(len(df), i + 3))
                tmp = df.loc[context_range].copy()
                tmp['mention_flag'] = 0
                tmp.at[i, 'mention_flag'] = 1
                tmp['unique_id'] = i
                
                grouped = (tmp.groupby('id')
                          .agg(list)
                          .explode(['text', 'mention_flag', 'unique_id'])
                          .reset_index())
                to_keep.append(grouped)
        
        return pd.concat(to_keep, axis=0).reset_index(drop=True) if to_keep else None


def process_corpus(
    base_dir: str = '/data/now/extracted',
    output_dir: str = '/data/mourad/narratives/inflation/sentences',
    years: range = range(2012, 2023),
    sentence_only: bool = False,
    article_only: bool = False
) -> None:
    """Process the entire NOW corpus.

    Args:
        base_dir: Base directory containing NOW corpus
        output_dir: Directory to save processed files
        years: Range of years to process
        sentence_only: If True, only process sentences
        article_only: If True, only process articles
    """
    preprocessor = NOWPreprocessor()
    stats = defaultdict(list)
    
    for year in years:
        for month in [f"{m:02d}" for m in range(1, 13)]:
            logger.info(f"Processing {year} - {month}")
            
            files = glob(f'{base_dir}/{year}/text/*{str(year)[-2:]}*{month}-*.txt')
            if not files:
                logger.warning(f"No files found for {year}-{month}")
                continue

            try:
                dfs = []
                for file in files:
                    df = preprocessor.preprocess_file(
                        file,
                        sentence_only=sentence_only,
                        article_only=article_only
                    )
                    if df is not None:
                        dfs.append(df)

                if not dfs:
                    continue

                combined_df = pd.concat(dfs, axis=0)
                
                if article_only:
                    stats[year].append(len(combined_df))
                elif sentence_only:
                    stats[year].append(len(combined_df))
                    logger.info(f"Year {year} has {len(combined_df)} sentences")
                else:
                    out_dir = os.path.join(output_dir, str(year))
                    os.makedirs(out_dir, exist_ok=True)
                    output_file = os.path.join(out_dir, f'us_now_{year}_{month}.tsv')
                    combined_df.to_csv(output_file, sep='\t', index=False)
                    logger.info(f"Saved processed data to {output_file}")

            except Exception as e:
                logger.error(f"Error processing {year}-{month}: {str(e)}")
                continue

    # Save statistics
    if stats:
        stats_df = pd.DataFrame(stats).reset_index()
        stats_df = pd.melt(
            stats_df,
            id_vars="index",
            var_name='year',
            value_name='count_type'
        )
        stats_df = stats_df.rename({'index': 'month'}, axis=1)
        
        filename = "NOW_sentence_counts.csv" if sentence_only else "NOW_article_counts.csv"
        output_path = f"output/{filename}"
        stats_df.to_csv(output_path, index=False)
        logger.info(f"Saved statistics to {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process NOW corpus data')
    parser.add_argument('--sentence_only', action='store_true',
                      help='Only process sentences')
    parser.add_argument('--article_only', action='store_true',
                      help='Only process articles')
    parser.add_argument('--log-level', 
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default='INFO',
                      help='Set the logging level')
    args = parser.parse_args()

    # Set log level from command line argument
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting NOW corpus processing")
    process_corpus(sentence_only=args.sentence_only, article_only=args.article_only)
    logger.info("Finished NOW corpus processing") 