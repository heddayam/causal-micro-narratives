from src.utils import utils
import json
from collections import defaultdict



def identify_representative_samples(data):
    # selected = defaultdict(list)
    selected = {}
    types_included = []
    foreign = False
    counter_narrative = False
    inflation_times = []
    narrative_times = []
    multiple_narratives = False
    for i, example in enumerate(data.shuffle(seed=42)):
        input = example['input']
        input = json.loads(input)
        type_ids = []
        if input['contains-narrative'] is False:
            type_ids.append("none")
        else:
            for narr in input['inflation-narratives']['narratives']:
                type_ids.append(list(narr.values())[0])
            
        for t in type_ids:
            if t not in selected:
                selected[t] = "<sentence>" + example['text'] +  "</sentence>" + "\n```json\n" + json.dumps(input, indent=4) + "\n```"
                if input['foreign']:
                    foreign = True
                if t != 'none':
                    if input['inflation-narratives']['counter-narrative']:
                        counter_narrative = True
                    inflation_times.append(input['inflation-narratives']['inflation-time'])
                    for narr in input['inflation-narratives']['narratives']:
                        narrative_times.append(list(narr.values())[1])
                if len(type_ids) > 1:
                    multiple_narratives = True

                break
    
    prompts = ""
    for sel in selected.values():
        prompts += sel + "\n\n"

    with open(f"../prompt-templates/in-context/gpt/fewshot.txt", 'w') as f:
        f.write(prompts)

        # if not foreign:
        #     if instance['foreign']:
        #         selected['']
        #         foreign = True
        # breakpoint()

if __name__ == "__main__":
    split = "train"
    data = utils.load_hf_dataset(split=split)
    data = data.map(lambda x: {"input": utils.reconstruct_training_input(x)}, batch_size=False, load_from_cache_file=False)
    identify_representative_samples(data)   