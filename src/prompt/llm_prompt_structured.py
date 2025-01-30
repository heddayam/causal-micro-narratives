from src.utils import utils
# from src.utils.utils import OUT_BASE
from pathlib import Path
from tqdm import tqdm
import argparse
import os
import json

tqdm.pandas()


def get_completion(cache, prompt, model, claude_or_gpt):
    model_str = utils.model_mapping.get(model)
    # print(prompt.split("\n")[-1])
    if claude_or_gpt == 'claude':
        completion = cache.generate(
            model=model_str,
            max_tokens=15,
            temperature=0,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]}]
            # prompt=prompt.format(HUMAN_PROMPT=HUMAN_PROMPT.strip(), AI_PROMPT=AI_PROMPT, SENTENCE=sentence),
        )
        completion = completion.content[0].text.strip()
        print(completion)
    elif model_str.startswith("o1"):
        completion = cache.generate(
            model=model_str, #gpt-3.5-turbo-1106
            messages=[{"role": "user", "content": prompt}],
        )
        completion = completion.choices[0].message.content.strip()
        completion = completion.replace("```json\n", "")
        completion = completion.replace("\n```", "")
        print(json.loads(completion))
    else:
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
        print(json.loads(completion))
        # breakpoint()

    # completion = completion.replace("```json\n", "")
    # completion = completion.replace("\n```", "")

    return completion

def init_and_query_llm(data, model, dataset, test_ds, seed):
    claude_or_gpt = 'claude' if model.startswith('claude') else 'gpt'
    cache = utils.init_llm(model)
    instr = Path(f"../prompt-templates/in-context/{claude_or_gpt}/instruction_v3.txt").read_text()
    # fewshots = Path(f"../prompt-templates/in-context/{claude_or_gpt}/fewshot_{dataset}_{seed}.txt").read_text()
    # fewshots = Path(f"../prompt-templates/in-context/{claude_or_gpt}/fewshot_v2.txt").read_text()
    zeroshot = Path(f"../prompt-templates/in-context/{claude_or_gpt}/zeroshot.txt").read_text()
    # instr = Path(f"../prompt-templates/in-context/{claude_or_gpt}/fewshot_cot.txt").read_text()
    # instr = Path(f"../prompt-templates/in-context/{claude_or_gpt}/question.txt").read_text()
    prompt = instr + "\n\n" #+ fewshots


    # completions = []
    def format_and_get_completion(instance):
        input = prompt + zeroshot.format(SENTENCE=instance['text'])
        # input = instr.format(SENTENCE=instance['text'])
        completion = get_completion(cache, input, model, claude_or_gpt)
        return completion

    # data = data.select(range(250))

    print(data)
    data = data.map(lambda x: {"completion": format_and_get_completion(x)}, batch_size=False, load_from_cache_file=False)
    breakpoint()
    if dataset != test_ds:
        out_path = f"/data/mourad/narratives/model_json_preds/{model}_train-{dataset}_test-{test_ds}_seed{seed}"
    else:
        out_path = f"/data/mourad/narratives/model_json_preds/{model}_{dataset}_seed{seed}"
    os.makedirs(out_path, exist_ok=True)
    print(out_path)
    data.save_to_disk(out_path)
    
    # df['completion'] = df.text.progress_apply(get_completion, args=(cache, prompt, model, claude_or_gpt))

def main(model, debug, dataset, test_ds, seed):
    data = utils.load_hf_dataset(split="test", dataset=test_ds)
    data = data.map(lambda x: {"input": utils.reconstruct_training_input(x)}, batch_size=False, load_from_cache_file=False)
    init_and_query_llm(data, model, dataset, test_ds, seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    # parser.add_argument("-y", "--year", type=int, required=False, default=None, help="Year to filter on")
    # parser.add_argument("-rw", "--rewrite", action='store_true', help='rewrite the gpt4 output? cached results will still be used')
    parser.add_argument('--model', choices=['claude', 'gpt35', 'gpt4t', "gpt4", "gpt4o", "o1-mini"], default='gpt4t')
    parser.add_argument('--debug', '-d', action='store_true')
    # parser.add_argument('--do_eval', action='store_true', help="whether to do eval on manually annotated data")
    # parser.add_argument('--eval_path', default="../data/eval/annotated/effect-done-mourad.tsv", help="path to manually annotated data for eval")
    # parser.add_argument('--inflation_type', choices=['cause', 'effect'])
    # parser.add_argument("--downsample", type=int, default=10000, help="Downsample # of sentences to analyze with LLM to this, -1 to not downsample")
    parser.add_argument('--out_dir', default='/data/mourad/narratives/proquest')
    parser.add_argument('--fewshot_ds', choices=['proquest', 'now', 'now_and_proquest'], required=True)
    parser.add_argument('--test_ds', choices=['proquest', 'now'], required=True)
    parser.add_argument('--fewshot_seed', type=int, default=1, choices=[1,2,3,4,5])
    # parser.add_argument("--prompt_path", default="")
    # Parse the arguments
    args = parser.parse_args()

    # for seed in :
    main(args.model, args.debug, args.fewshot_ds, args.test_ds, args.fewshot_seed)