import json
from pathlib import Path
import argparse
from collections import defaultdict
import pandas as pd
from datetime import datetime
from datasets import Dataset, DatasetDict
from nltk.tokenize.treebank import TreebankWordDetokenizer
import subprocess
import re
from src.utils import utils

DATA_BASE = Path("../../data/annotations/inflation/")
OUT_BASE = Path("/data/mourad/narratives/sft_data")
# OUT_BASE = Path("/data/mourad/narratives/annotation_data")

input_schema = """
{
"foreign": #,
"contains-narrative": #,
"inflation-narratives": {
    "inflation-time": "#",
    "counter-narrative": #,
    "narratives": #
    }
}
"""

input_schema_no_narrative = """
{
"foreign": #,
"contains-narrative": #,
"inflation-narratives": null
}
"""

class LabelStudioDataInterpreter:
    def __init__(self, data_path, format):
        self.data = self.read_tsv(data_path)
        self.temps = ['past', 'present', 'future', 'general']
        if format == "standard":
            self.extract_and_structure_annotations = self.extract_and_structure_annotations_std
        elif format == "min":
            self.extract_and_structure_annotations = self.extract_and_structure_annotations_min

    def read_tsv(self, data_filename):
        data = pd.read_csv(DATA_BASE / "exported" / data_filename, sep='\t')
        data = self.clean_and_split(data)
        return data

    def read_json(self, data_filename):
        with open(DATA_BASE / "exported" / data_filename, 'r') as f:
            data = json.load(f)
        return data

    def load_consensus_test(self):
        df = pd.read_csv("../../data/annotations/inflation/exported/annotator_diffs_consensus.tsv", sep='\t')
        df = df[['id', 'sentence', 'consensus']]
        df = df.rename({'sentence': 'text'}, axis=1)
        return df
    
    def write_jsonl(self, records, out_path, data_filename):
        # out_filename = "_".join(data_filename.split("_")[-2:])
        with open(OUT_BASE / "train.jsonl", 'w') as f:
            for record in records['train']:
                f.write(json.dumps(record) + '\n')
        
        with open(OUT_BASE / "test.jsonl", 'w') as f:
            for record in records['test']:
                f.write(json.dumps(record) + '\n')

    def write_dataset(self, records):
        
        all_splits = {}
        for split, record in records.items():
            df = pd.DataFrame.from_dict(record)
            ds = Dataset.from_pandas(df, preserve_index=False)
            all_splits[split] = ds

        ds = DatasetDict(all_splits)
        # breakpoint()
        # df = pd.DataFrame.from_dict(records['test'])
        # test_ds = Dataset.from_pandas(df)

        # df = pd.DataFrame.from_dict(records['train'])
        # train_ds = Dataset.from_pandas(df)

        # ds = DatasetDict({'train': train_ds, 'test': test_ds})
        breakpoint()
        ds.save_to_disk(str(OUT_BASE))
        return ds
    
    
    def extract_choices(self, response):
        try:
            response = json.loads(response)
        except:
            pass
        if isinstance(response, str):
            response = [response]
        else:
            response = response['choices']
        return response
    
    # def keep_most_updated_annotation(self, instance):
    #     if len(instance) > 1:
    #         if '*' in instance.assigned:
    #             instance = instance[instance.assigned == 'az*']
    #         else:
    #             instance.updated_at = instance.updated_at.apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    #             instance = instance[instance.updated_at == instance.updated_at.max()]
    #     return instance

    def keep_most_updated_annotation(self, instance):
        if len(instance) > 1:
            instance.updated_at = instance.updated_at.apply(lambda x: datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
            instance = instance[instance.updated_at == instance.updated_at.max()]
        return instance

    def assigned_split(self, assigned):
        if assigned == 'az*':
            return 'test_az'
        elif assigned == 'mh*':
            return 'test_mh'
        elif assigned == 'qz*':
            return 'test_qz'
        else:
            return 'train'

    
    def clean_and_split(self, data):
        data = data.groupby(['assigned', 'text']).apply(self.keep_most_updated_annotation)
        
        # breakpoint()
        # data['split'] = data.assigned.apply(lambda x: 'test' if '*' in x else 'train')
        data['split'] = data.assigned.apply(self.assigned_split)
        tmp = data[data.split == 'test_az'].copy()
        tmp['split'] = 'test'
        data = pd.concat([data, tmp])
        # TODO fix this
        detokenizer = TreebankWordDetokenizer()
        data['text'] = data.text.apply(lambda x: utils.detokenize(x, detokenizer))
        return data
    
    def make_dataset(self):
        records = self.extract_and_structure_annotations()
        inputs = {'train': [], 'test_az': [], 'test_mh': [], 'test_qz': [], 'test': []}
        for split, record in records.items():
            for r in record:
                instance = {"id": r['id'], "text": r['text'], "assigned": r['assigned']}
                foreign = r['foreign']
                contains_narratives = bool(r['selections'])
                if contains_narratives:
                    temporality = r['temporality']
                    counter_narrative = r['counter_narrative']
                    causes = [{"cause": s[0], "time": s[1]} for s in r['selections']['cause']] if 'cause' in r['selections'] else []
                    effects = [{"effect": s[0], "time": s[1]} for s in r['selections']['effect']] if 'effect' in r['selections'] else []
                    narratives = json.dumps(causes + effects)
                    instance['template'] = input_schema.strip()
                    try:
                        instance['data'] = "#".join([json.dumps(foreign), json.dumps(contains_narratives), temporality, json.dumps(counter_narrative), narratives])
                    except:
                        breakpoint()
                    #input = input_schema.format(FOREIGN=foreign, CONTAINTS_NARRATIVE=contains_narratives, TEMPORALITY=temporality, COUNTER_NARRATIVE=counter_narrative, NARRATIVES=narratives)
                else:
                    instance['template'] = input_schema_no_narrative.strip()
                    instance['data'] = "#".join([json.dumps(foreign), json.dumps(contains_narratives)])
                    #input = input_schema_no_narrative.format(FOREIGN=foreign, CONTAINTS_NARRATIVE=contains_narratives)    
                inputs[split].append(instance)
        return inputs

    def extract_and_structure_annotations_std(self):
        records = {'train': [], 'test_az': [], 'test_mh': [], 'test_qz': [], 'test': []}
        for name, instance in self.data.iterrows():
            try:
                annotation_id = instance['annotation_id']
                assigned = instance['assigned']
                text = instance['text']
                id = instance['id']
                counter_narrative = instance['counter-narrative'] == 'counter_narrative'
                categories = self.extract_choices(instance['narrative-type'])

                if "foreign" in categories:
                    foreign = True
                    categories.remove('foreign')
                else:
                    foreign = False
                # breakpoint()
                selections = {}
                if 'none' not in categories:
                    temporality = instance['temporality']
                    for cat in categories:
                        selections[cat] = []
                        types = self.extract_choices(instance[cat])
                        
                        for t in types:
                            if t not in self.temps:
                                selections[cat].append((t, instance[f"{t}-time"]))
                    print(selections)
                else:
                    temporality = None
                if instance.split == 'test':
                    df = self.load_consensus_test()
                    consensus = df[df.text == instance.text]
                    if len(consensus) > 0:
                        labels = consensus.consensus.item().split(",")
                        temp = [col for col in instance.dropna().index if '-time' in col]
                        if len(temp) > 0:
                            temp = instance[temp[0]]
                        else:
                            temp = 'none'
                        if 'none' not in labels:
                            for l in labels:
                                l = l.strip()
                                if l in utils.causes:
                                    if 'cause' not in selections:
                                        selections['cause'] = []
                                    selections['cause'].append((l, temp))
                                elif l in utils.effects:
                                    if 'effect' not in selections:
                                        selections['effect'] = []
                                    selections['effect'].append((l, temp))
                                else:
                                    print("error, wrong label in consensus - ", l)
                                if temporality is None:
                                    temporality = 'general'
                        else:
                            temporality = None
                            counter_narrative = False
                            selections = {}
                records[instance.split].append(
                    {
                        'annotation_id': annotation_id,
                        'id': id,
                        'assigned': assigned,
                        'text': text,
                        'foreign': foreign,
                        'counter_narrative': counter_narrative,
                        'temporality': temporality,
                        'selections': selections
                    }
                )
            except Exception as e:
                print(e)
                breakpoint()
                print(instance['id'], instance['assigned'])
        return records


    # def extract_and_structure_annotations_min(self):
    #     records = []
    #     for instance in self.data:
    #         try:
    #             id = instance['id']
    #             if 'counter-narrative' in instance and instance['counter-narrative'] == 'counter_narrative':
    #                 counter_narrative = True
    #             else:
    #                 counter_narrative = False
    #             categories = self.extract_choices(instance['narrative-type'])
    #             foreign = 'foreign' in categories
    #             selections = {}
    #             if 'none' not in categories:
    #                 temporality = instance['temporality']

    #                 for cat in categories:
    #                     if cat == 'foreign':
    #                         foreign = True
    #                     else:
    #                         selections[cat] = []
    #                         types = self.extract_choices(instance[cat])
    #                         for t in types:
    #                             if t not in self.temps:
    #                                 selections[cat].append((t, instance[f"{t}-time"]))
    #             records.append(
    #                 {
    #                     'id': id,
    #                     'foreign': foreign,
    #                     'counter_narrative': counter_narrative,
    #                     'temporality': temporality,
    #                     'selections': selections
    #                 }
    #             )
    #         except:
    #             print(instance['id'], instance['assigned'])
    #     return records



if __name__ == "__main__":
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument("--labelstudio_export", "-ls", type=str, default='export_tsv_proquest_0721.tsv', help="Filename of the LabelStudio export")
    parse_args.add_argument("--export_format", "-f", type=str, default="standard", choices=["min", "standard"])
    args = parse_args.parse_args()

    # data_filename = "labelstudio_export_02_05.json"
    interpreter = LabelStudioDataInterpreter(args.labelstudio_export, args.export_format)
    records = interpreter.make_dataset()

    breakpoint()
    ds = interpreter.write_dataset(records)
    # utils.scp_file(OUT_BASE, remote_path='/net/projects/chai-lab/mourad/narratives-data/', remote_host='dsi')
