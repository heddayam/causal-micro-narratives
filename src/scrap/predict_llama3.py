# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from transformers import BitsAndBytesConfig, TextStreamer
import torch
from peft import PeftModel, PeftConfig
from pathlib import Path

# from datasets import load_from_disk
# # from mistral_sft_train import PROMPT_DICT
# from phi2_binary_sft_train import PROMPT_DICT_BINARY
# from phi3_sft_train import PROMPT_DICT
from phi2_sft_train import PROMPT_DICT
FEWSHOT1 = Path("../prompt-templates/sft/phi2/fewshot1_now.txt").read_text()

from vllm import LLM, SamplingParams
import shutil
import json
from pathlib import Path
import argparse
import pandas as pd
import datasets
from datasets import Dataset
from guidance import models, gen, select
import guidance
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm.auto import tqdm
from tokenizers import pre_tokenizers

from src.utils import utils

from glob import glob
import os

# 4.43.3 transformers original working


cats = ['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause', 'cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect']

@guidance(stateless=False)
def narrative_maker(lm, sentence, dataset):
    instr = PROMPT_DICT['prompt'].format(SENTENCE=sentence)#.strip()
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
    def __init__(self, model, gpu, reuse, sample, train_ds, test_ds, ckpt=None, sampling_params=None, max_tokens=1000, split=None, debug=False):
        ckpt_base = Path('/net/projects/chai-lab/mourad/narratives-data/sft_out')
        
        self.model = model
        self.debug = debug
        self.split = split
        self.sample = sample
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.ckpt = ckpt

        if model == 'mistral':
            model_str = 'mistralai/Mistral-7B-Instruct-v0.2'
        elif 'phi2' in model:
            model_str = 'microsoft/phi-2'
        elif 'phi3' in model:
            model_str = 'microsoft/Phi-3-mini-4k-instruct'
        elif model == 'llama31':
            model_str = 'meta-llama/Meta-Llama-3.1-8B'
        elif 'llama31_instr' in model:
            model_str = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        elif 'gemma2_2b' == model:
            model_str = 'google/gemma-2-2b'
        else:
            raise ValueError("Model not supported")
        
        if split == 'NOW_filtered':
            self.dataset = utils.read_all_data("/net/projects/chai-lab/mourad/narratives-data/filtered_sentences_for_prediction/", location=False)
            print(self.dataset)
        elif split == 'PROQUEST_filtered':
            self.dataset = utils.read_all_data(f"/net/projects/chai-lab/mourad/narratives-data/filtered_sentences_for_prediction/{test_ds}", location=False, filename=f"processed_data_2010-2025.jsonl.gz")
            print(self.dataset)
        else:
            self.dataset = utils.load_hf_dataset(path=f"/net/projects/chai-lab/mourad/narratives-data/sft_data_{test_ds}", split=split)
        if self.debug:
            if split == 'NOW_filtered':
                per_job_sample = 60000
                try:
                    self.dataset = self.dataset.select(list(range(per_job_sample * sample, per_job_sample * (sample + 1))))
                except:
                    self.dataset = self.dataset.select(list(range(per_job_sample * sample, len(self.dataset))))
            elif split == 'PROQUEST_filtered':
                per_job_sample = 50000
                try:
                    self.dataset = self.dataset.select(list(range(per_job_sample * sample, per_job_sample * (sample + 1))))
                except:
                    self.dataset = self.dataset.select(list(range(per_job_sample * sample, len(self.dataset))))
            else:
                self.dataset = self.dataset.select(list(range(20)))
                
            data = self.dataset.to_pandas()
            data['lens'] =data['text'].apply(lambda x: len(x.split()))
            data = data[data.lens < 400]
            self.dataset = Dataset.from_pandas(data, preserve_index=False)

        if ckpt is not None:
            if model == 'mistral':
                ckpt_path = ckpt_base / (model + f"_{gpu}") / ckpt  
            else:
                ckpt_path = ckpt_base / (model + f"_{train_ds}") / ckpt

            print(f"Loading model from {ckpt_path}")
            self.llm = self.load_finetuned_model(model_str, ckpt_path, reuse)
        else:
            self.llm = models.Transformers(model_str, device_map='auto')

    def load_finetuned_model(self, model_str, ckpt_path, reuse):
        model_path = f"/net/projects/chai-lab/mourad/narratives-data/merged_model/{self.model}"
        if not reuse:
            cache_dir="/net/projects/chai-lab/mourad/data/models_cache"
            model = AutoModelForCausalLM.from_pretrained(
                        model_str, 
                        device_map="auto",
                        trust_remote_code=True,
                        token = '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',
                        attn_implementation='eager' if 'gemma-2-2b' in model_str else 'sdpa',
                        cache_dir=cache_dir)
                # model.resize_token_embeddings(32000+len(json.load(open(f"{ckpt}/added_tokens.json"))))
            model = PeftModel.from_pretrained(model, ckpt_path, trust_remote_code=True)
            
            tok = AutoTokenizer.from_pretrained(ckpt_path, use_fast=False)

            merged_model = model.merge_and_unload()

            
            merged_model.save_pretrained(model_path)


            for file in glob(f"{ckpt_path}/*token*"):
                print(file)
                shutil.copy(file, model_path)

            if 'phi2' in model_str.lower():
                shutil.copy(f"{ckpt_path}/vocab.json", model_path)
                shutil.copy(f"{ckpt_path}/merges.txt", model_path)
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)

        if 'llama' in model_str:
            byte_decoder = {}
            alphabet = pre_tokenizers.ByteLevel(False, False).alphabet()
            known_vals = set([])

            for j in range(256):
                for k in range(256):
                    for l in range(256):
                        if len(byte_decoder.keys()) < 256:
                            b = b""
                            vals = [j,k,l]
                            if not set(vals).issubset(known_vals):
                                for d in range(3):
                                    b = b + vals[d].to_bytes(1, 'little', signed=False)
                                try:
                                    c = b.decode()
                                    t = pre_tokenizers.ByteLevel(False,False).pre_tokenize_str(c)[0][0]
                                    for m in range(3):
                                        if t[m] not in byte_decoder.keys():
                                            byte_decoder[t[m]] = vals[m]
                                            known_vals.add(vals[m])
                                except UnicodeDecodeError:
                                    pass


            print(len(byte_decoder))

            byte_decoder['À'] = 192
            byte_decoder['Á'] = 193

            byte_decoder['ð'] = 240
            byte_decoder['ñ'] = 241
            byte_decoder['ò'] = 242
            byte_decoder['ó'] = 243
            byte_decoder['ô'] = 244
            byte_decoder['õ'] = 245
            byte_decoder['ö'] = 246
            byte_decoder['÷'] = 247
            byte_decoder['ø'] = 248
            byte_decoder['ù'] = 249
            byte_decoder['ú'] = 250
            byte_decoder['û'] = 251
            byte_decoder['ü'] = 252
            byte_decoder['ý'] = 253
            byte_decoder['þ'] = 254
            byte_decoder['ÿ'] = 255

            tokenizer.byte_decoder = byte_decoder
            llm = models.Transformers(model_path, tokenizer=tokenizer, device_map='auto', trust_remote_code=True, echo=False)
        else:
            llm = models.Transformers(model_path, device_map='auto', trust_remote_code=True, echo=False)
        # llm = LLM(model_path) #="microsoft/phi-2", enable_lora
        # breakpoint()
        return llm
    
    def generate(self):
        generated = []        
        for instance in tqdm(self.dataset):
            sentence = instance['text']

            try:
                outputs = self.llm + narrative_maker(sentence, self.test_ds)
                contains = outputs['contains-narrative'].lower() == 'true'
                foreign = outputs['foreign'].lower() == 'true'
                if contains:
                    if self.test_ds == 'proquest':
                        narratives = {
                            "inflation-time": outputs['inflation-time'],
                            "inflation-direction":  outputs['inflation-direction'].lower(),
                            "narratives": [
                                outputs['narratives']
                            ]
                        }
                    elif self.test_ds == 'now':
                        counter_narrative = outputs['counter-narrative'].lower() == 'true'
                        narratives = {
                            "inflation-time": outputs['inflation-time'],
                            "counter-narrative": counter_narrative,
                            "narratives": [
                                outputs['narratives']                                    
                            ]
                        }
                    elif self.test_ds == 'now_and_proquest':
                        narratives = {
                            "inflation-time": outputs['inflation-time'],
                            "narratives": [
                                outputs['narratives']                                    
                            ]
                        }
                else:
                    narratives = None
            except Exception as e:
                print(e)
                breakpoint()
                
            record = {
                'foreign': foreign,
                'contains-narrative': contains,
                "inflation-narratives": narratives
            }
            generated.append(json.dumps(record))

        return generated
    
    def save_outputs(self, generated, model_type):
        self.dataset = self.dataset.add_column('completion', generated)
        
        model = self.model
        ckpt_steps = ""
            
        if self.ckpt != '' and self.ckpt is not None:
            ckpt_steps = "_" + self.ckpt.split('-')[1] + "s"
        if self.sample >= 0:
            out_path = f"/net/projects/chai-lab/mourad/narratives-data/model_json_preds/{self.test_ds}/full_{self.test_ds}/{model}_{model_type}_{ckpt_steps}_train-{self.train_ds}_sample_{self.sample}_2010-2025"
        else:
            out_path = f"/net/projects/chai-lab/mourad/narratives-data/model_json_preds/{model}_{model_type}{ckpt_steps}_train-{self.train_ds}_test-{self.test_ds}"
                
        os.makedirs(out_path, exist_ok=True)
        self.dataset.save_to_disk(out_path)
        print("Saved dataset to disk = ", out_path)

def main(**kwargs):
    generator = NarrativeGenerator(**kwargs)
    generated = generator.generate()

    if not kwargs['debug'] or kwargs['split'].endswith('filtered'):
        if kwargs['ckpt'] is None:
            model_type = 'base'
        else:
            model_type = 'ft'
    else:
        breakpoint()
    
    generator.save_outputs(generated, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mistral', 'phi2', 'phi2_first_run', 'phi3', 'llama31_inst', 'llama31', 'gemma2_2b'], required=True)
    parser.add_argument('--gpu', type=str, choices=['a100', 'a40'], default='a100')
    parser.add_argument('--ckpt', default = None, required=False)
    parser.add_argument('--split', choices=['train', 'dev', 'test', 'NOW_filtered', 'PROQUEST_filtered'], required=True, help='NOW_filtered is for doing predictions on entire NOW dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample', type=int, default=-1)
    parser.add_argument("--train_ds", choices=['now', 'proquest', 'now_and_proquest'], required=True)
    parser.add_argument("--test_ds", choices=['now', 'proquest', 'processed_data_1960-1980'], required=True)
    parser.add_argument('--reuse', action='store_true')

    args=parser.parse_args()

    if args.split == 'dev':
        raise ValueError("No dev split right now. Use test or train.")

    main(**vars(args))


#for i, line in enumerate(generator.dataset['data']): print(line, "\n", generated[i], "\n", "\n") 