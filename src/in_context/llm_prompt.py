from pathlib import Path
from typing import Optional, List, Dict, Union
import os
import pandas as pd
from tqdm import tqdm
import httpx
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import argparse
import logging
import sys
from datetime import datetime
from src.utils import utils
import json
from datasets import Dataset
from dotenv import load_dotenv
load_dotenv()

# Configure logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Silence httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"inflation_analysis_{timestamp}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure tqdm for pandas operations
tqdm.pandas()

class OpenRouterClient:
    """Client for making requests to OpenRouter API."""
    
    def __init__(self):
        """Initialize OpenRouter client.
        """
        self.api_key = os.environ.get('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("OpenRouter client initialized")

    def generate(self, model: str, messages: List[Dict[str, str]], 
                max_tokens: int = 500, temperature: float = 0,
                **kwargs) -> Dict:
        """Generate completion using OpenRouter API.
        
        Args:
            model: Model identifier
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response dictionary
        """
        url = f"{self.base_url}/chat/completions"
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs  # Include any additional parameters
        }
        
        response = httpx.post(url, json=data, headers=self.headers)
        return response.json()

def get_completion(sentence: str, client: OpenRouterClient, 
                  prompt: str, model: str) -> Optional[str]:
    """Get completion from LLM for a given sentence.
    
    Args:
        sentence: Input text to process
        client: OpenRouter client instance
        prompt: Prompt to use
        model: Model identifier
        
    Returns:
        Extracted answer or None if no answer found
    """
    logger = logging.getLogger("get_completion")
    messages = [{"role": "user", "content": prompt + "\n\n" + sentence}]

    try:
        response = client.generate(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0,
            provider= {
                "require_parameters": True
            },
            response_format = {
                "type": "json_object"
                }
        )
        
        completion = response['choices'][0]['message']['content'].strip()
        return json.loads(completion)
        
    except Exception as e:
        logger.error(f"Error getting completion for sentence: {sentence}... Error: {str(e)}")
        return None

def process_dataset(df: pd.DataFrame, model: str, instr_path: str, fewshot_path: str) -> pd.DataFrame:
    """Process dataset with LLM completions.
    
    Args:
        df: Input DataFrame
        model: Model identifier        
    Returns:
        DataFrame with added completions
    """
    logger = logging.getLogger("process_dataset")
    logger.info(f"Processing dataset with {len(df)} rows")
    
    client = OpenRouterClient()
    prompt = Path(instr_path).read_text()
    if fewshot_path:
        fewshots = Path(fewshot_path).read_text()
        prompt += "\n\n" + fewshots

    logger.info("Starting LLM in-context prompting")
    df['completion'] = df.text.progress_apply(
        get_completion, 
        args=(client, prompt, model)
    )
    
    completion_stats = {
        'total': len(df),
        'successful': df['completion'].notna().sum()
    }
    logger.info(f"Processing complete. Stats: {completion_stats}")
    
    return df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Classify inflation narratives using in-context prompting')
    
    parser.add_argument("-rw", "--rewrite", action='store_true', 
                       help='Rewrite the output (cached results still used)')
    parser.add_argument('--model', choices=['anthropic/claude-2', 'openai/gpt-3.5-turbo', 
                                          'openai/gpt-4-turbo', "openai/gpt-4o"], 
                       default='openai/gpt-4o')
    parser.add_argument('--instr_path', default='prompts/gpt/instruction_v3.txt', help='Path to instruction prompt')
    parser.add_argument('--fewshot_path', required=False, help='Path to few-shot prompt')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--out_dir', default='/data/mourad/narratives/model_json_preds/')
    parser.add_argument('--dataset', choices=['proquest', 'now', 'now_and_proquest'], required=True)

    # TODO add support for different train / test datasets
    
    args = parser.parse_args()
    logger = logging.getLogger("main")
    logger.info(f"Starting inflation analysis with args: {vars(args)}")

    # Setup output paths
    out_base = Path(args.out_dir)
    model_name = args.model.split('/')[-1]
    instr_name = args.instr_path.split('/')[-1].split('.')[0]
    fewshot_name = "_" + args.fewshot_path.split('/')[-1].split('.')[0] if args.fewshot_path else ""
    out_path = out_base / f"{model_name}_{instr_name}{fewshot_name}_{args.dataset}"

    # Check if output exists and rewrite not requested
    if os.path.exists(out_path) and not args.rewrite:
        logger.info(f"Loading existing results from {out_path}")
        data = Dataset.from_file(out_path)
        df = data.to_pandas()
        return df
        
    # Load and filter data
    logger.info(f"Loading {args.dataset} test data")
    data = utils.load_hf_dataset(split="test", dataset=args.dataset)
    data = data.map(lambda x: {"input": utils.reconstruct_training_input(x)}, batch_size=False, load_from_cache_file=False)
    df = data.to_pandas()

    if args.debug:
        logger.debug("Debug mode: sampling 10 rows")
        df = df.sample(10)
        
    # Process data
    df = process_dataset(df, args.model, args.instr_path, args.fewshot_path)

    # Save results
    if not args.debug:
        logger.info(f"Saving results to {out_path}")
        os.makedirs(out_path, exist_ok=True)
        data = Dataset.from_pandas(df)
        data.save_to_disk(out_path)
        
    logger.info("Processing complete")

if __name__ == "__main__":
    main()

        