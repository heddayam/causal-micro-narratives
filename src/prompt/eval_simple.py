from src.utils import utils
import swifter
import pandas as pd
from pathlib import Path
import json

def evaluate_completion():
    pass

def get_completion(model, prompt):
    model_str = utils.model_mapping.get(model)
    completion = cache.generate(
                model=model_str, #gpt-3.5-turbo-1106
                max_tokens=256,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_object"
                }
            )
    completion = completion.choices[0].message.content.strip()
    
    return completion

def get_labels(inp):
    narr_labels = []
    inp = json.loads(inp)
    if inp['inflation-narratives'] is not None:
        if 'narratives' in inp['inflation-narratives']:
            for narr in inp['inflation-narratives']['narratives']:
                narr_labels.append((list(narr.keys())[0], list(narr.values())[0]))
    return narr_labels


model = 'gpt4o'

cache = utils.init_llm(model)

data = utils.load_hf_dataset(split="test", dataset='now')
data = data.map(lambda x: {"input": utils.reconstruct_training_input(x)}, batch_size=False, load_from_cache_file=False)
now = data.to_pandas()

data = utils.load_hf_dataset(split="test", dataset='proquest')
data = data.map(lambda x: {"input": utils.reconstruct_training_input(x)}, batch_size=False, load_from_cache_file=False)
proquest = data.to_pandas()

data = pd.concat([now, proquest])

data['labels'] = data.input.apply(get_labels)

breakpoint()

instr = Path(f"../prompt-templates/in-context/gpt/instruction_v3.txt").read_text()
zeroshot = Path(f"../prompt-templates/in-context/gpt/zeroshot.txt").read_text()
prompt = instr + "\n\n" #+ fewshots



data['completion'] = data.text.swifter.apply(lambda x: get_completion(model, prompt.format(SENTENCE=x)))






