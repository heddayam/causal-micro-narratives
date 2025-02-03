"""
Narrative prediction module for generating structured narrative predictions from text using various LLMs.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import os
import json
from dataclasses import dataclass

# Third-party imports
import torch
from transformers import (
    pipeline,
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig, 
    TextStreamer
)
from peft import PeftModel, PeftConfig
from vllm import LLM, SamplingParams
from guidance import models, gen, select
import guidance
from datasets import Dataset
from tqdm.auto import tqdm
from tokenizers import pre_tokenizers
from dotenv import load_dotenv

# Local imports
from src.utils import utils
from phi2_sft_train import PROMPT_DICT

# Load environment variables
load_dotenv()

# Constants
CACHE_DIR = "/net/projects/chai-lab/mourad/data/models_cache"
MODEL_PATH_BASE = "/net/projects/chai-lab/mourad/narratives-data/merged_model"
CKPT_BASE = Path('/net/projects/chai-lab/mourad/narratives-data/sft_out')
OUTPUT_BASE = "/net/projects/chai-lab/mourad/narratives-data/model_json_preds"

# Dataset constants
MAX_SENTENCE_LENGTH = 400  # Maximum number of words in a sequence

FULL_DATASET_PATHS = {
    'NOW_filtered': {
        'path': "/net/projects/chai-lab/mourad/narratives-data/filtered_sentences_for_prediction/",
        'chunk_size': 60000,
        'filename': None
    },
    'PROQUEST_filtered': {
        'path': "/net/projects/chai-lab/mourad/narratives-data/filtered_sentences_for_prediction/{test_ds}",
        'chunk_size': 50000,
        'filename': "processed_data_1923-2025.jsonl.gz"
    }
}

@dataclass
class NarrativeConfig:
    """Configuration for narrative generation."""
    model: str
    gpu: str
    reuse: bool
    sample: int  # For full datasets: chunk index to process
    train_ds: str
    test_ds: str
    ckpt: Optional[str] = None
    max_tokens: int = 1000
    split: Optional[str] = None  # Can be train/test or NOW_filtered/PROQUEST_filtered
    debug: bool = False

@guidance(stateless=False)
def narrative_maker(lm: Any, sentence: str, dataset: str) -> Any:
    """Generate narrative predictions using guidance.
    
    Args:
        lm: Language model instance
        sentence: Input text to analyze
        dataset: Dataset type ('proquest', 'now', or 'now_and_proquest')
    
    Returns:
        Generated narrative classification
    """
    instr = PROMPT_DICT['prompt'].format(SENTENCE=sentence)
    stop_regex = "]"
    lm += instr
    lm += f"""\
    {{
    "foreign": {select(['true', 'false'], 'foreign')},
    "contains-narrative": {select(['true', 'false'], 'contains-narrative')},
    """
    if lm['contains-narrative'].lower() == "true":
        if dataset == 'proquest':
            lm += f"""\
            "inflation-narratives": {{
                "inflation-time": "{select(['past', 'present', 'future', 'general'],'inflation-time')}",
                "inflation-direction": "{select(['up', 'down', 'same'], 'inflation-direction')}",
                "narratives": [{gen('narratives', stop_regex=stop_regex, max_tokens=256)}]
                }},
            """
        elif dataset == 'now':
            lm += f"""\
            "inflation-narratives": {{
                "inflation-time": "{select(['past', 'present', 'future', 'general'],'inflation-time')}",
                "counter-narrative": "{select(['true', 'false'], 'counter-narrative')}",
                "narratives": [{gen('narratives', stop_regex=stop_regex, max_tokens=256)}]
            }},
        """
        elif dataset == 'now_and_proquest':
            lm += f"""\
            "inflation-narratives": {{
            "inflation-time": "{select(['past', 'present', 'future', 'general'],'inflation-time')}",
            "narratives": [{gen('narratives', stop_regex=stop_regex, max_tokens=256)}]
            }},
        """
    else:
        lm += f"""\
        "inflation-narratives": None,
        """
    lm += f"""\
    }}
    """
    return lm

class NarrativeGenerator:
    """Main class for generating narrative predictions from text."""
    
    def __init__(self, config: NarrativeConfig):
        """Initialize the narrative generator with given configuration."""
        self.config = config
        self.model = config.model
        self.debug = config.debug
        self.split = config.split
        self.sample = config.sample
        self.train_ds = config.train_ds
        self.test_ds = config.test_ds
        self.ckpt = config.ckpt

        self._setup_model()
        self._load_dataset()

    def _setup_model(self) -> None:
        """Set up the language model based on configuration."""
        if self.model not in utils.model_mapping:
            raise ValueError(f"Model {self.model} not supported")
        
        model_str = utils.model_mapping[self.model]
        
        if self.ckpt is not None:
            ckpt_path = CKPT_BASE / (self.model + f"_{self.train_ds}") / self.ckpt
            print(f"Loading model from {ckpt_path}")
            self.llm = self._load_finetuned_model(model_str, ckpt_path, self.config.reuse)
        else:
            self.llm = models.Transformers(model_str, device_map='auto')

    def _load_dataset(self) -> None:
        """Load and prepare the dataset based on configuration.
        
        Handles two types of datasets:
        1. Full datasets (NOW_filtered, PROQUEST_filtered): Large datasets processed in chunks
        2. Standard datasets: Regular train/test splits processed in full
        
        All datasets are filtered to remove sequences longer than MAX_SEQUENCE_LENGTH words.
        """
        if self.split in FULL_DATASET_PATHS:
            self._load_full_dataset_chunk()
        else:
            self._load_dataset_split()

        # Filter out long sequences for all datasets
        self._filter_long_sentences()

        if self.debug:
            self._prepare_debug_dataset()

    def _filter_long_sentences(self) -> None:
        """Filter out sequences that are too long for processing."""
        original_size = len(self.dataset)
        data = self.dataset.to_pandas()
        data['lens'] = data['text'].apply(lambda x: len(x.split()))
        data = data[data.lens < MAX_SENTENCE_LENGTH]
        self.dataset = Dataset.from_pandas(data, preserve_index=False)
        filtered_size = len(self.dataset)
        
        if filtered_size < original_size:
            print(f"Filtered out {original_size - filtered_size} sequences longer than {MAX_SENTENCE_LENGTH} words")
            print(f"Remaining dataset size: {filtered_size} instances")

    def _load_full_dataset_chunk(self) -> None:
        """Load a chunk of a full dataset for distributed processing."""
        dataset_config = FULL_DATASET_PATHS[self.split]
        path = dataset_config['path']
        if self.split == 'PROQUEST_filtered':
            path = path.format(test_ds=self.test_ds)

        self.dataset = utils.read_all_data(
            path,
            location=False,
            filename=dataset_config['filename']
        )

        # Select the appropriate chunk for this job
        chunk_size = dataset_config['chunk_size']
        start_idx = chunk_size * self.sample
        try:
            end_idx = chunk_size * (self.sample + 1)
            self.dataset = self.dataset.select(list(range(start_idx, end_idx)))
        except:
            # Handle last chunk which might be smaller
            self.dataset = self.dataset.select(list(range(start_idx, len(self.dataset))))

        print(f"Processing {self.split} chunk {self.sample}: {len(self.dataset)} instances")

    def _load_dataset_split(self) -> None:
        """Load a standard train/test split dataset."""
        if self.split not in ['train', 'test']:
            raise ValueError(f"Invalid split {self.split} for standard dataset")
            
        self.dataset = utils.load_hf_dataset(
            path=f"/net/projects/chai-lab/mourad/narratives-data/sft_data_{self.test_ds}",
            split=self.split
        )
        print(f"Loaded {self.split} split: {len(self.dataset)} instances")

    def _prepare_debug_dataset(self) -> None:
        """Prepare a smaller dataset for debugging."""
        # For debugging, we just take a small sample
        self.dataset = self.dataset.select(list(range(20)))
        print(f"Debug dataset size: {len(self.dataset)} instances")

    def _load_finetuned_model(self, model_str: str, ckpt_path: Path, reuse: bool) -> models.Transformers:
        """Load a fine-tuned model from checkpoint."""
        model_path = Path(MODEL_PATH_BASE) / self.model
        
        if not reuse:
            self._prepare_model_files(model_str, ckpt_path, model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        
        if 'llama' in model_str:
            tokenizer.byte_decoder = self._setup_byte_decoder()
            return models.Transformers(model_path, tokenizer=tokenizer, device_map='auto', trust_remote_code=True, echo=False)
        
        return models.Transformers(model_path, device_map='auto', trust_remote_code=True, echo=False)

    def _prepare_model_files(self, model_str: str, ckpt_path: Path, model_path: Path) -> None:
        """Prepare model files by loading and merging weights."""
        model = AutoModelForCausalLM.from_pretrained(
            model_str, 
            device_map="auto",
            trust_remote_code=True,
            token=os.environ.get('HUGGINGFACE_API_KEY'),
            attn_implementation='sdpa',
            cache_dir=CACHE_DIR
        )
        model = PeftModel.from_pretrained(model, ckpt_path, trust_remote_code=True)
        
        tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(model_path)

        for file in Path(ckpt_path).glob('*token*'):
            print(f"Copying {file}")
            Path(file).copy(model_path)

        if 'phi2' in model_str.lower():
            Path(ckpt_path / "vocab.json").copy(model_path)
            Path(ckpt_path / "merges.txt").copy(model_path)

    @staticmethod
    def _setup_byte_decoder() -> Dict[str, int]:
        """Set up byte decoder for LLaMA models."""
        byte_decoder = {}
        known_vals = set()

        for j in range(256):
            for k in range(256):
                for l in range(256):
                    if len(byte_decoder) >= 256:
                        break
                    b = b""
                    vals = [j,k,l]
                    if not set(vals).issubset(known_vals):
                        for d in range(3):
                            b = b + vals[d].to_bytes(1, 'little', signed=False)
                        try:
                            c = b.decode()
                            t = pre_tokenizers.ByteLevel(False,False).pre_tokenize_str(c)[0][0]
                            for m in range(3):
                                if t[m] not in byte_decoder:
                                    byte_decoder[t[m]] = vals[m]
                                    known_vals.add(vals[m])
                        except UnicodeDecodeError:
                            pass

        # Add special characters
        special_chars = {
            'À': 192, 'Á': 193, 'ð': 240, 'ñ': 241, 'ò': 242, 'ó': 243,
            'ô': 244, 'õ': 245, 'ö': 246, '÷': 247, 'ø': 248, 'ù': 249,
            'ú': 250, 'û': 251, 'ü': 252, 'ý': 253, 'þ': 254, 'ÿ': 255
        }
        byte_decoder.update(special_chars)
        
        return byte_decoder

    def generate(self) -> List[str]:
        """Generate narrative predictions for the dataset."""
        generated = []
        for instance in tqdm(self.dataset):
            try:
                record = self._process_single_instance(instance['text'])
                generated.append(json.dumps(record))
            except Exception as e:
                print(f"Error processing instance: {e}")
                breakpoint()
        return generated

    def _process_single_instance(self, sentence: str) -> Dict[str, Any]:
        """Process a single text instance and generate narrative prediction."""
        outputs = self.llm + narrative_maker(sentence, self.test_ds)
        contains = outputs['contains-narrative'].lower() == 'true'
        foreign = outputs['foreign'].lower() == 'true'
        
        narratives = None
        if contains:
            narratives = self._create_narrative_structure(outputs)
            
        return {
            'foreign': foreign,
            'contains-narrative': contains,
            "inflation-narratives": narratives
        }

    def _create_narrative_structure(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create the appropriate narrative structure based on dataset type."""
        base_structure = {
            "inflation-time": outputs['inflation-time'],
            "narratives": [outputs['narratives']]
        }
        
        if self.test_ds == 'proquest':
            base_structure["inflation-direction"] = outputs['inflation-direction'].lower()
        elif self.test_ds == 'now':
            base_structure["counter-narrative"] = outputs['counter-narrative'].lower() == 'true'
            
        return base_structure

    def save_outputs(self, generated: List[str], model_type: str) -> None:
        """Save generated predictions to disk."""
        self.dataset = self.dataset.add_column('completion', generated)
        
        ckpt_steps = f"_{self.ckpt.split('-')[1]}s" if self.ckpt and self.ckpt != '' else ""
        
        if self.sample >= 0:
            out_path = Path(OUTPUT_BASE) / self.test_ds / f"full_{self.test_ds}" / f"{self.model}_{model_type}_{ckpt_steps}_train-{self.train_ds}_sample_{self.sample}_2010-2025"
        else:
            out_path = Path(OUTPUT_BASE) / f"{self.model}_{model_type}{ckpt_steps}_train-{self.train_ds}_test-{self.test_ds}"
                
        out_path.mkdir(parents=True, exist_ok=True)
        self.dataset.save_to_disk(out_path)
        print(f"Saved dataset to disk = {out_path}")

def main(config: NarrativeConfig) -> None:
    """Main function to run narrative generation."""
    generator = NarrativeGenerator(config)
    generated = generator.generate()

    model_type = 'ft' if config.ckpt is not None else 'base'
    if config.debug and not config.split.endswith('filtered'):
        breakpoint()
    
    generator.save_outputs(generated, model_type)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate narrative predictions from text using LLMs")
    parser.add_argument('--model', choices=['phi2', 'phi2_first_run', 'llama31'], required=True)
    parser.add_argument('--gpu', type=str, choices=['a100', 'a40'], default='a100')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--split', choices=['train', 'test', 'NOW_filtered', 'PROQUEST_filtered'], 
                      required=True, help='NOW_filtered/PROQUEST_filtered are full datasets processed in chunks')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample', type=int, default=-1,
                      help='For full datasets: chunk index to process (required for NOW_filtered/PROQUEST_filtered)')
    parser.add_argument("--train_ds", choices=['now', 'proquest', 'now_and_proquest'], required=True)
    parser.add_argument("--test_ds", choices=['now', 'proquest', 'processed_data_1960-1980'], required=True)
    parser.add_argument('--reuse', action='store_true')

    args = parser.parse_args()
    
    # Validate arguments
    if args.split in FULL_DATASET_PATHS and args.sample < 0:
        raise ValueError(f"Must specify chunk index (--sample) when processing full datasets ({args.split})")
    if args.split == 'dev':
        raise ValueError("No dev split right now. Use test or train.")

    config = NarrativeConfig(**vars(args))
    main(config) 