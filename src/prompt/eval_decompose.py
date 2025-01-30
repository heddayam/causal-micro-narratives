from src.utils import utils
import swifter
import pandas as pd
from pathlib import Path
import json
import re

def evaluate_completion():
    pass

def extract_brackets(text):
    # Regular expression to find text within brackets
    pattern = r'\[(.*?)\]'

    # Find all matches
    matches = re.findall(pattern, text)
    return matches

def get_completion(model, prompt, max_tokens=256, fmt="json_object"):
    model_str = utils.model_mapping.get(model)
    completion = cache.generate(
                model=model_str, #gpt-3.5-turbo-1106
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": fmt
                }
            )
    completion = completion.choices[0].message.content.strip()
    
    return completion

def decompose_sentence(prompt, sentence, filter=False):
    prompt = prompt.format(SENTENCE=sentence)
    completion = get_completion(model, prompt, max_tokens=512, fmt='text')
    # sents = completion.split(",")
    # sents = [sent.strip() for sent in sents if 'inflation' in sent.lower()]
    # sents = ".".join(sents)

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

# breakpoint()

decomp = Path(f"../prompt-templates/in-context/decompose.txt").read_text()
instr = Path(f"../prompt-templates/in-context/instruction_decomposed.txt").read_text()
# zeroshot = Path(f"../prompt-templates/in-context/gpt/zeroshot.txt").read_text()
prompt = instr  #+ fewshots

data = data.sample(5)

data['decomp'] = data.text.apply(lambda x: decompose_sentence(decomp, x))


data['completion'] = data.decomp.apply(lambda x: get_completion(model, prompt.format(DECOMPOSED=x), fmt='text'))

data['completion'] = data.completion.apply(lambda x: extract_brackets(x))
breakpoint()




