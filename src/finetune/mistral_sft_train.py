#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import os

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from peft import LoftQConfig, LoraConfig, get_peft_model
from src.utils import utils
# from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from datetime import datetime
from pathlib import Path

os.environ["WANDB_PROJECT"]="inflation_narratives_sft"

IGNORE_INDEX = -100

PROMPT_DICT = {
    "prompt": Path("/net/scratch/mourad/economic-narratives/src/prompt-templates/sft/mistral-instruct/prompt1.txt").read_text(),
    "prompt_nospecial": Path("/net/scratch/mourad/economic-narratives/src/prompt-templates/sft/mistral-instruct/prompt1_nospecial.txt").read_text()
    # "prompt_infer": 
    #     "<s>[INST]Provide a summary of the following Supreme Court opinion:\n\n"
    #     " [TEXT_START]\n\n{opinion}\n\n[TEXT_END]\n\n[/INST]"
    # )
}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        breakpoint()
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    run_name: str 
    report_to: str = field(default="wandb")
    # max_steps: int = field(default=None)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """

        
    tokenizer.pad_token = tokenizer.unk_token

    if len(special_tokens_dict) != 0:
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        # num_new_tokens  = 0
        # # num_new_tokens = tokenizer.add_tokens([SUMMARY_TEMPLATE], special_tokens=True) ##This line is updated

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]



    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(instance, tokenizer):
    batch_size = len(instance['text'])
    model_inputs = {"input_ids": [], "labels": []}
    for idx in range(batch_size):
        instr = PROMPT_DICT['prompt'].format(SENTENCE=instance['text'][idx]) 
        template = instance['template'][idx].split("#")
        data = instance['data'][idx].split("#")
        input = tokenizer(instr)['input_ids']
        label = [-100] * len(input)
        for i, t in enumerate(template):
            ttoks  = tokenizer(t)['input_ids']
            input += ttoks
            label += [-100] * len(ttoks)
            if i < len(data):
                dtoks = tokenizer(data[i])['input_ids']
                input += dtoks
                label += dtoks
        input += [tokenizer.eos_token_id]
        label += [-100]
        model_inputs["input_ids"].append(input)
        model_inputs["labels"].append(label)
    return model_inputs


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         dataset = utils.load_hf_dataset(data_path)

#         # logging.warning("Formatting inputs...")
#         # prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
#         # sources = [
#         #     prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
#         #     for example in list_data_dict
#         # ]
#         # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

#         logging.warning("Formatting & Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(dataset, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.LongTensor(instance[key]).flip(dims=[0]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).flip(dims=[1])
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX).flip(dims=[1])
        # input_ids = torch.Tensor(instances[0]['input_ids'])
        # labels = torch.Tensor(instances[0]['labels'])
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # dataset = load_from_disk(data_args.data_path)
    dataset = utils.load_hf_dataset(data_args.data_path)
    processed_datasets = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["test"]
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)



def find_target_modules(model):
    # Initialize a Set to Store Unique Layers
    unique_layers = set()
    
    # Iterate Over All Named Modules in the Model
    for name, module in model.named_modules():
        # Check if the Module Type Contains 'Linear4bit'
        if "Linear" in str(type(module)):
            # Extract the Type of the Layer
            layer_type = name.split('.')[-1]
            
            # Add the Layer Type to the Set of Unique Layers
            unique_layers.add(layer_type)

    # Return the Set of Unique Layers Converted to a List
    layers = list(unique_layers)
    layers.remove("lm_head")
    layers.remove('dense')
    #['k_proj', 'dense', 'fc1', 'v_proj', 'fc2', 'q_proj']
    return layers


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        trust_remote_code=True,
        cache_dir=training_args.cache_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    special_tokens_dict = dict()
        
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model
    )
    
    lora_config = LoraConfig(
        r=64, #32
        lora_alpha=128, #64
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            # "lm_head"
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
    logging.warning("Loading data...")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    now = datetime.now()
    training_args.run_name += now.strftime('_%Y-%m-%d_%H:%M')

    # breakpoint()

    trainer = Trainer(
    # trainer = CustomTrainer(
        model=model, 
        tokenizer=tokenizer, 
        # peft_config=lora_config,
        args=training_args, 
        **data_module
        )
    # model.config.use_cache = False
    trainer.train()
    print(trainer.model)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()