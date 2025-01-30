import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


abbrev_map = {
    "power": "Reduced Purchasing Power",
    "uncertain": "Uncertainty",
    "rates": "Interest Rates",
    "wealth": "Income/Wealth Redistribution",
    "cost": "Cost of Living Increases",
    "savings": "Impact on Savings/Investments",
    "international": "International Competitiveness",
    "impact": "Social and Political Impact",
    "push": "Cost-Push on Businesses",
    "indiv": "Impact on Fixed-Income Individuals",
    "n": "missing category",
    "na": "na",
    "foreign": "foreign"
}


def calculate_precision_recall_multiclass(y_true, y_pred, average='weighted'):
    """
    Calculate precision and recall for multiclass predictions.

    Parameters:
    - y_true: List of true labels (multiclass)
    - y_pred: List of predicted labels (multiclass)
    - average: Type of averaging for precision and recall ('micro', 'macro', 'weighted')

    Returns:
    - precision: Precision score
    - recall: Recall score
    """
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=average)

    return precision, recall


def clean_for_pr(labels):
    if labels == []:
        return "None"
    for label in labels:
        if label not in ['foreign', -1]: 
                return label


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["claude", "gpt35", "gpt4t", "gpt4"], required=True)
    args = parser.parse_args()

    effect_gold = pd.read_csv("data/eval/annotated/effect-done-mourad.tsv", sep="\t")
    effect = pd.read_parquet(f"/data/mourad/narratives/categories/{args.model}_effect.parquet")

    effect_gold.label = effect_gold['label'].apply(lambda x: x.split(','))

    effect_gold.label = effect_gold['label'].apply(lambda x: [abbrev_map[abbrev.strip()] if abbrev.strip() != 'na' else 'None' for abbrev in x])
    # effect.label = effect.label.replace("na", None)
    effect_gold = effect_gold.rename({"label": "label_manual"},axis=1)

    

    df = effect.merge(effect_gold, on=["id", "text"], how="left")
    df.label = df.label.fillna("None")

   

    df['correct'] = df.apply(lambda x: list(set(x.label).intersection(set(x.label_manual))), axis=1)
    # df['correct'] = df.correct.apply(lambda x: x[0] if x else None)
    df['correct_binary'] = df.correct.apply(lambda x: len(x) > 0)


    print(df.correct_binary.mean())
    # print(df.label_manual.value_counts())
    # print(df.label.value_counts())
    # breakpoint()

    df['label_manual'] = df.label_manual.apply(clean_for_pr)
    df['label']  = df.label.apply(clean_for_pr)
    df.label_manual = df.label_manual.fillna("None")
    df.label = df.label.fillna("None")



    cm = confusion_matrix(df.label_manual, df.label)
    plt.figure(figsize=(14, 10))
    # Display confusion matrix as a heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(df.label_manual), yticklabels=np.unique(df.label))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    plt.savefig(f"/data/mourad/narratives/categories/eval/{args.model}_effect_confusion_matrix.png", dpi=300)



    label_to_id = {
        'Impact on Savings/Investments': 1,
        'Impact on Fixed-Income Individuals': 2,
        'Interest Rates': 3,
        'None': 4,
        'Social and Political Impact': 5,
        'International Competitiveness': 6,
        'Uncertainty': 7,
        'Reduced Purchasing Power': 8,
        'Cost-Push on Businesses': 9,
        'Cost of Living Increases': 10,
        'Income/Wealth Redistribution': 11
    }
    df['id_label'] = df.label.map(label_to_id)
    df['id_manual'] = df.label_manual.map(label_to_id)


    

    df = df[(df.id_manual != 4) & (df.id_label != 4)]

    acc = accuracy_score(df.id_manual, df.id_label)
    pr = calculate_precision_recall_multiclass(df.id_manual, df.id_label)

   
    print(pr)
    print(acc)

    # breakpoint()