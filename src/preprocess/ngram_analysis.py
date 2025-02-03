from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress specific pandas and seaborn warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

class NgramAnalyzer:
    """Class to handle n-gram analysis of text corpus with visualization capabilities."""
    
    def __init__(self, vocabulary: Dict[str, int], output_dir: str = 'output/ngrams'):
        """
        Initialize the NgramAnalyzer.
        
        Args:
            vocabulary: Dictionary mapping words to their indices
            output_dir: Directory to save output visualizations
        """
        self.vocabulary = vocabulary
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lemmatizer = WordNetLemmatizer()
        
    def vectorize_corpus(
        self, 
        df: pd.DataFrame, 
        ngram_range: Tuple[int, int] = (1, 1)
    ) -> pd.DataFrame:
        """
        Vectorize the text corpus using TF-IDF.
        
        Args:
            df: DataFrame containing the text corpus
            ngram_range: Range of n-gram sizes to consider
            
        Returns:
            DataFrame with vectorized text
        """
        try:
            corpus = df.text
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words='english',
                lowercase=True,
                vocabulary=self.vocabulary,
                use_idf=False,
                norm=None,
                binary=True
            )
            
            bow = vectorizer.fit_transform(corpus)
            return pd.DataFrame(bow.toarray(), columns=self.vocabulary.keys())
            
        except Exception as e:
            logger.error(f"Error in vectorizing corpus: {str(e)}")
            raise
    
    def prepare_data(
        self, 
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare data for visualization.
        
        Args:
            df: DataFrame with vectorized text and time information
            
        Returns:
            Tuple of processed DataFrame and x-axis labels
        """
        try:
            # Lemmatize feature names and handle grouping
            x_label = "time"
            x = df[x_label]
            feature_df = df[list(self.vocabulary.keys())]
            unique_kw = [self.lemmatizer.lemmatize(t) for t in feature_df.columns]
            feature_df.columns = unique_kw
            feature_df = feature_df.groupby(feature_df.columns, axis=1).agg(sum)
            self.features = feature_df.columns
            feature_df[x_label] = x
            indiv_df = df.groupby(['time', 'month_year']).agg(np.mean)*100
            indiv_df = indiv_df.reset_index()

            xlabels = []
            current_yr = ''
            for m in df.month_year.unique().tolist():
                year = m.split('-')[1]
                if current_yr != year:
                    xlabels.append(str(year))
                    current_yr = year
                else:
                    xlabels.append('')
            
            return indiv_df, xlabels
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
    
    def plot_word_frequencies(
        self, 
        indiv_df: pd.DataFrame, 
        xlabels: List[str]
    ) -> None:
        """
        Generate individual bar plots for each word's frequency.
        
        Args:
            indiv_df: DataFrame with processed frequency data
            xlabels: Labels for x-axis
        """
        try:
            for word in self.features:
                plt.figure(figsize=(8, 6))
                # Convert inf values to NaN before plotting
                plot_data = indiv_df.copy()
                plot_data[word] = plot_data[word].replace([np.inf, -np.inf], np.nan)
                
                sns.barplot(
                    data=plot_data,
                    x='time',
                    y=word,
                    width=1,
                    edgecolor='black'
                )
                plt.xticks(list(set(indiv_df.time)), xlabels)
                plt.xlabel("Year")
                plt.ylabel(f"% of articles that mention {word}")
                
                output_path = self.output_dir / f"{word}.png"
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f"Error in plotting word frequencies: {str(e)}")
            raise
    
    def plot_comparison(
        self, 
        df: pd.DataFrame, 
        xlabels: List[str]
    ) -> None:
        """
        Generate comparison plot of all words.
        
        Args:
            df: DataFrame with processed data
            xlabels: Labels for x-axis
        """
        try:
            df_melted = pd.melt(
                df,
                id_vars=['time'],
                value_vars=self.features,
                var_name="word",
                value_name='counts'
            )
            # Handle inf values
            df_melted['counts'] = df_melted['counts'].replace([np.inf, -np.inf], np.nan)
            df_melted = df_melted[df_melted.counts != 0]
            
            plt.figure(figsize=(8, 6))
            # Use newer seaborn API
            sns.kdeplot(
                data=df_melted[df_melted.word != 'inflation'],
                x='time',
                hue='word',
                multiple="fill",
                bw_adjust=1,
                clip=(0, len(xlabels)-1)
            )
            plt.xticks(list(set(df.time)), xlabels)
            plt.xlabel("Year")
            plt.ylabel("Words related to inflation")
            
            output_path = self.output_dir / "compare.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error in plotting comparison: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Analyze n-grams in text corpus')
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        default="/data/mourad/narratives/inflation/all_filtered.jsonl.gz",
        help='Path to input JSON lines file'
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default='output/ngrams',
        help='Directory for output files'
    )
    parser.add_argument(
        "-d",
        "--debug",
        action='store_true',
        help="Enable debug mode"
    )
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Define vocabulary with indices
    vocabulary = {
        "inflation": 0,
        "demand": 1,
        "supply": 2,
        "supplies": 3,
        "cost": 4,
        "costs": 5,
        "wage": 6,
        "wages": 7,
        "uncertainty": 8,
        "uncertainties": 9,
        "saving": 10,
        "savings": 11,
        "investment": 12,
        "investments": 13
    }

    try:
        # Initialize analyzer
        analyzer = NgramAnalyzer(vocabulary, args.output_dir)
        
        # Load and preprocess data
        logger.info("Loading data...")
        df = pd.read_json(
            args.input_file,
            orient='records',
            lines=True,
            compression='gzip'
        )
        df = df.sort_values(['year', 'month'])
        df['time'] = df.groupby(['year', 'month']).ngroup()
        
        # Vectorize corpus
        logger.info("Vectorizing corpus...")
        counts = analyzer.vectorize_corpus(df)
        counts['time'] = df.time
        counts['month_year'] = df.month.astype(str) + '-' + df.year.astype(str)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        indiv_df, xlabels = analyzer.prepare_data(counts)
        analyzer.plot_word_frequencies(indiv_df, xlabels)
        analyzer.plot_comparison(counts, xlabels)
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 