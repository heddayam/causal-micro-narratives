# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from transformers import BitsAndBytesConfig, TextStreamer
import torch
from peft import PeftModel, PeftConfig

# from datasets import load_from_disk
# # from mistral_sft_train import PROMPT_DICT
from phi2_binary_sft_train import PROMPT_DICT_BINARY
# from phi3_sft_train import PROMPT_DICT
from phi2_sft_train import PROMPT_DICT

from vllm import LLM, SamplingParams
import shutil
import json
from pathlib import Path
import argparse
import pandas as pd
import datasets
from guidance import models, gen, select
import guidance
from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm.auto import tqdm

from src.utils import utils

from glob import glob
import os


cats = ['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause', 'cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect']

# @guidance(stateless=False)
# def narrative_maker(lm, sentence):
#     instr = PROMPT_DICT['prompt'].format(SENTENCE=sentence)
#     stop_regex = "[^a-zA-Z]" #"[\"']"
#     breakpoint()
#     lm += f"""\
#     {instr}
#     {{
#     "foreign": {gen('foreign', stop_regex=stop_regex)},
#     "contains-narrative": {gen('contains-narrative', stop_regex=stop_regex)},
#     """
#     if lm['contains-narrative'].lower() == "true":
#         lm += f"""\
#         "inflation-narratives": {{
#             "inflation-time": "{gen('inflation-time', stop_regex=stop_regex)}",
#             "counter-narrative": {gen('counter-narrative', stop_regex=stop_regex)},
#             "narratives": [{gen('narratives', stop=']')}]
#             }},
#         """
#     else:
#         lm += f"""\
#         "inflation-narratives": None,
#         """
#     lm += f"""\
#     }}
#     """
#     return lm
# @guidance(stateless=False)
# def narrative_maker(lm, sentence):
#     instr = PROMPT_DICT['prompt'].format(SENTENCE=sentence)
#     stop_regex = "[^a-zA-Z]" #"[\"']"
#     lm += instr
#     lm += f"""\
#     {{
#     "foreign": {gen('foreign', stop_regex=stop_regex)},
#     "contains-narrative": {gen('contains-narrative', stop_regex=stop_regex)},
#     """
#     if lm['contains-narrative'].lower() == "true":
#         lm += f"""\
#         "inflation-narratives": {{
#             "inflation-time": "{gen('inflation-time', stop_regex=stop_regex)}",
#             "counter-narrative": {gen('counter-narrative', stop_regex=stop_regex)},
#             "narratives": [{gen('narratives', stop=']')}]
#             }},
#         """

#     else:
#         lm += f"""\
#         "inflation-narratives": None,
#         """
#     lm += f"""\
#     }}
#     """
#     return lm

@guidance(stateless=False)
def narrative_maker(lm, sentence, binary=False):
    if binary:
        instr = PROMPT_DICT_BINARY['prompt'].format(SENTENCE=sentence)
        lm += instr
        lm += select(['Yes', 'No'], 'contains-narrative')
    else:
        instr = PROMPT_DICT['prompt'].format(SENTENCE=sentence)#.strip()
        stop_regex = "[^a-zA-Z]" #"[\"']"
        lm += instr
        lm += f"""\
        {{
        "foreign": "{select(['true', 'false'], 'foreign')}",
        "contains-narrative": "{select(['true', 'false'], 'contains-narrative')}",
        """
        if lm['contains-narrative'].lower() == "true":
            lm += f"""\
            "inflation-narratives": {{
                "inflation-time": "{select(['past', 'present', 'future', 'general'],'inflation-time')}",
                "counter-narrative": "{select(['true', 'false'], 'counter-narrative')}",
                "narratives": [{{"{select(['cause', 'effect'], 'cause_effect')}": "{select(['demand', 'supply', 'wage', 'expect', 'monetary', 'fiscal', 'international', 'other-cause', 'cost', 'govt', 'purchase', 'rates', 'redistribution', 'savings', 'social', 'trade', 'cost-push', 'uncertain', 'other-effect'], 'narrative_category')}", "time": "{select(['past', 'present', 'future', 'general'], 'narrative_time')}"}}]
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
    def __init__(self, model, gpu, reuse, binary, sample, ckpt=None, sampling_params=None, max_tokens=1000, split=None, debug=False):
        ckpt_base = Path('/net/projects/chai-lab/mourad/narratives-data/sft_out')
        
        self.model = model
        self.debug = debug
        self.split = split
        self.sample = sample

        if model == 'mistral':
            model_str = 'mistralai/Mistral-7B-Instruct-v0.2'
        elif 'phi2' in model:
            model_str = 'microsoft/phi-2'
        elif 'phi3' in model:
            model_str = 'microsoft/Phi-3-mini-4k-instruct'
        else:
            raise ValueError("Model not supported")
        
        if split == 'NOW_filtered':
            self.dataset = utils.read_all_data("/net/projects/chai-lab/mourad/narratives-data/filtered_sentences_for_prediction/", location=False)
            # self.dataset = self.dataset.filter(lambda x: x['year'] < 2017)
            print(self.dataset)
        else:
            # self.dataset = utils.load_hf_dataset(path="/net/projects/chai-lab/mourad/data/scotus", structured=True, split=split)
            self.dataset = utils.load_hf_dataset(path="/net/projects/chai-lab/mourad/narratives-data/sft_data", split=split)
        if self.debug:
            if split == 'NOW_filtered':
                # pass
                per_job_sample = 71055 # / 4
                self.dataset = self.dataset.select(list(range(per_job_sample * sample, per_job_sample * (sample + 1))))
            else:
                self.dataset = self.dataset.select(list(range(5)))


        if ckpt is not None: # use fine-tuned model
            if model == 'mistral':
                ckpt_path = ckpt_base / (model + f"_{gpu}") / ckpt  
            elif binary:
                ckpt_path = ckpt_base / (model + f"_binary") / ckpt
            else:
                ckpt_path = ckpt_base / model / ckpt
            self.llm = self.load_finetuned_model(model_str, ckpt_path, reuse)
        else: # use base model
            # self.llm = LLM(model_str) 
            self.llm = models.Transformers(model_str, device_map='auto')

        # breakpoint()
        # dataset = dataset.filter(lambda x: x['opinion_len']  < 8096)
        # TODO Update
        # self.dataset = self.dataset.map(lambda x: {"prompt": PROMPT_DICT["prompt_infer"].format(opinion=x['opinion'])})

        # if sampling_params is None:
            # self.sampling_params = SamplingParams(temperature=0, stop=[], max_tokens=max_tokens)
        # else: 
            # self.sampling_params = sampling_params


    def load_finetuned_model(self, model_str, ckpt_path, reuse):
        model_path = f"/net/projects/chai-lab/mourad/narratives-data/merged_model/{self.model}"
        if not reuse:
            cache_dir="/net/projects/chai-lab/mourad/data/models_cache"
            model = AutoModelForCausalLM.from_pretrained(
                        model_str, 
                        device_map="auto",
                        trust_remote_code=True,
                        cache_dir=cache_dir)
                # model.resize_token_embeddings(32000+len(json.load(open(f"{ckpt}/added_tokens.json"))))
            model = PeftModel.from_pretrained(model, ckpt_path, trust_remote_code=True)
            
            tok = AutoTokenizer.from_pretrained(ckpt_path)
            # breakpoint()
            # llm = models.Transformers(model, tok, device_map='auto', trust_remote_code=True)
            # return llm 
            #inp = tok([], return_tensors="pt")
            # streamer = TextStreamer(tok)
            # model.generate(**inp, streamer=streamer, max_new_tokens=100)

            # breakpoint()
            merged_model = model.merge_and_unload()

            
            merged_model.save_pretrained(model_path)


            for file in glob(f"{ckpt_path}/*token*"):
                print(file)
                shutil.copy(file, model_path)

            if 'phi2' in model_str.lower():
                shutil.copy(f"{ckpt_path}/vocab.json", model_path)
                shutil.copy(f"{ckpt_path}/merges.txt", model_path)
        llm = models.Transformers(model_path, device_map='auto', trust_remote_code=True, echo=False)
        # llm = LLM(model_path) #="microsoft/phi-2", enable_lora
        return llm
    
    def generate_binary(self):
        generated = []
        for instance in tqdm(self.dataset):
            sentence = instance['text']
            id = instance['id']
            outputs = self.llm + narrative_maker(sentence, binary=True)
            contains =  outputs['contains-narrative'].lower() == 'yes'
            record = {
                'contains-narrative': outputs['contains-narrative']#contains
            }
            generated.append(json.dumps(record))
        return generated
        

    def generate(self):
        generated = []        
        for instance in tqdm(self.dataset):
            sentence = instance['text']
            id = instance['id']

            try:
                outputs = self.llm + narrative_maker(sentence)
                contains =  outputs['contains-narrative'].lower() == 'true'
                foreign = outputs['foreign'].lower() == 'true'
                if contains:
                    counter_narrative = outputs['counter-narrative'].lower() == 'true'
                    # breakpoint()
                    # narratives = "[" + outputs['narratives'].replace("\'", "\"") + "]"
                    # narratives = json.loads(narratives)
                    narratives = {
                        "inflation-time": outputs['inflation-time'],
                        "counter-narrative": counter_narrative,
                        "narratives": [{
                            outputs['cause_effect']: outputs['narrative_category'],
                            "time": outputs['narrative_time']
                        }]

                    }
                    # narratives = tuple(set(tuple(v.values()) for v in json.loads(narratives)))
                else:
                    narratives = None
            except Exception as e:
                print(e)
                breakpoint()
            record = {
                # 'id': id,
                # 'text': sentence,
                'foreign': foreign,
                'contains-narrative': contains,
                "inflation-narratives": narratives
            }
            generated.append(json.dumps(record))

        # outputs = self.llm.generate(prompts, self.sampling_params)
        # generated = []
        # for output in outputs:
        #     prompt = output.prompt
        #     generated_text = output.outputs[0].text
        #     generated.append(generated_text)
        #     # print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        return generated
    
    def save_outputs(self, generated, model_type, binary):
        self.dataset = self.dataset.add_column('completion', generated)
        
        model = self.model#.split("_")[0]

        # out_path = f"/data/mourad/narratives/model_json_preds/{model}_{model_type}"
        if binary:
            out_path = f"/net/projects/chai-lab/mourad/narratives-data/model_json_binary_preds/{model}_{model_type}"
        else:
            out_path = f"/net/projects/chai-lab/mourad/narratives-data/model_json_preds/{model}_{model_type}_{self.split}_sample_{self.sample}"
        os.makedirs(out_path, exist_ok=True)
        self.dataset.save_to_disk(out_path)

        # df = self.dataset.to_pandas()
        # df = df[['citation', 'opinion', 'syllabus', 'generated']]
        # file_out = f"/net/projects/chai-lab/mourad/narratives-data/model_predictions/{self.model}_{model_type}.json"
        # df.to_json(file_out, orient='records')
        print("Saved dataset to disk = ", out_path)


def main(**kwargs):
    generator = NarrativeGenerator(**kwargs)
    if kwargs['binary']:
        generated = generator.generate_binary()
    else:
        generated = generator.generate()

    if not kwargs['debug'] or kwargs['split'] == 'NOW_filtered':
        if kwargs['ckpt'] is None:
            model_type = 'base'
        else:
            model_type = 'ft'
    
    generator.save_outputs(generated, model_type, kwargs['binary'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mistral', 'phi2', 'phi2_first_run', 'phi3'], required=True)
    parser.add_argument('--gpu', type=str, choices=['a100', 'a40'], default='a100')
    parser.add_argument('--ckpt', default = None, required=False)
    parser.add_argument('--split', choices=['train', 'dev', 'test', 'NOW_filtered'], required=True, help='NOW_filtered is for doing predictions on entire NOW dataset')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--binary', action='store_true', help='narrative or not binary pred')
    args=parser.parse_args()

    if args.split == 'dev':
        raise ValueError("No dev split right now. Use test or train.")

    main(**vars(args))
