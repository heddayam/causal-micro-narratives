import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import os
import torch
import transformers
from transformers import Trainer
from peft import LoraConfig, get_peft_model
from src.utils import utils
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

os.environ["WANDB_PROJECT"] = "inflation_narratives_sft"

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt": Path("prompts/prompt_defs.txt").read_text(),
}

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class DebugTrainer(Trainer):
    """Custom trainer class used for debugging purposes."""
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="microsoft/phi-2")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    run_name: str 
    report_to: str = field(default="wandb")

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    is_phi_model: bool = True
):
    """Resize tokenizer and embedding."""
    tokenizer.pad_token = tokenizer.unk_token if is_phi_model else tokenizer.eos_token
    
    if len(special_tokens_dict) > 0:
        num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

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
            ttoks = tokenizer(t, add_special_tokens=False)['input_ids']
            input += ttoks
            label += [-100] * len(ttoks)
            
            if i < len(data):
                dtoks = tokenizer(data[i], add_special_tokens=False)['input_ids']
                input += dtoks
                label += dtoks
                
        model_inputs["input_ids"].append(input)
        model_inputs["labels"].append(label)
    return model_inputs

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
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
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

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Determine if we're using a Phi model
    is_phi_model = "phi" in model_args.model_name_or_path.lower()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        token=os.environ.get('HUGGINGFACE_API_KEY'),
        cache_dir=training_args.cache_dir
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
        token=os.environ.get('HUGGINGFACE_API_KEY')
    )

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={},
        tokenizer=tokenizer,
        model=model,
        is_phi_model=is_phi_model
    )
    
    # Configure LoRA based on model type
    lora_config = LoraConfig(
        r=16 if is_phi_model else 8,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}%")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    now = datetime.now()
    training_args.run_name += now.strftime('_%Y-%m-%d_%H:%M')

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train() 