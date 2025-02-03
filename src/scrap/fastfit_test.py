from datasets import load_from_disk, DatasetDict
from fastfit import FastFitTrainer, sample_dataset, FastFit
from transformers import AutoTokenizer, pipeline
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from src.utils import utils
import json


# Load a dataset from the Hugging Face Hub
train = load_from_disk("/data/mourad/narratives/narrative_category_data/train")
test = load_from_disk("/data/mourad/narratives/narrative_category_data/test")

dataset = DatasetDict({'train': train, 'validation': test, 'test': test})

predict = False

if predict:
    model = FastFit.from_pretrained("/data/mourad/narratives/narrative_category_data/fast-fit")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large", model_input_names=["input_ids", "attention_mask"])
    
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    results = classifier(test['text'])
    preds = [res['label'] for res in results]
    breakpoint()
else:
    # dataset["validation"] = dataset["test"]

    # Down sample the train data for 5-shot training
    dataset["train"] = sample_dataset(dataset["train"], label_column="narrative", num_samples_per_label=2)
    for inst in dataset["train"]:
        out = utils.reconstruct_training_input(inst)
        out = json.loads(out)
        print(inst['text'])
        print()
        print(out)
        print("**********************)")
        breakpoint()

    trainer = FastFitTrainer(
        model_name_or_path="microsoft/deberta-large",
        label_column_name="narrative",
        text_column_name="text",
        num_train_epochs=20,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=32,
        eval_steps=20,
        evaluation_strategy="steps",
        logging_strategy="steps",
        logging_steps=10,
        max_text_length=256,
        dataloader_drop_last=False,
        num_repeats=4,
        optim="adafactor",
        clf_loss_factor=0.1,
        fp16=True,
        dataset=dataset
    )

    model = trainer.train()
    results = trainer.evaluate()
    
    print("Accuracy: {:.1f}".format(results["eval_accuracy"] * 100))
    breakpoint()
    model.save_pretrained("/data/mourad/narratives/narrative_category_data/fast-fit")
 