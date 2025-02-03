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
OUT_BASE = Path("/data/mourad/narratives/sft_data_proquest_basic")
# OUT_BASE = Path("/data/mourad/narratives/annotation_data")

input_schema = """
{
"foreign": #,
"contains-narrative": #,
"inflation-narratives": {
    "inflation-time": "#",
    "inflation-direction": #,
    "narratives": #
    }
}
"""

input_schema_basic = """
{
"causes": #,
"effects": #
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
            df = df.sample(frac=1)
            ds = Dataset.from_pandas(df, preserve_index=False)
            all_splits[split] = ds
        
        ds = DatasetDict(all_splits)

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
    

    def keep_most_updated_annotation(self, instance):
        if len(instance) > 1:
            breakpoint()
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
        # data = data.groupby(['assigned', 'text']).apply(self.keep_most_updated_annotation)
        # 1895 1894
        # can tell if in test set because duplicated
        tmp = data.groupby(['assigned', 'text']).agg(list).reset_index()
        testset_texts = tmp[tmp.text.duplicated()].drop_duplicates(subset=['text']).text.tolist()
        
        
        data['assigned'] = data.apply(lambda x: x.assigned + "*" if x.text in testset_texts else x.assigned, axis=1)
        
        # data['split'] = data.assigned.apply(lambda x: 'test' if '*' in x else 'train')
        data['split'] = data.assigned.apply(self.assigned_split)
        tmp = data[data.split == 'test_az'].copy()
        tmp['split'] = 'test'
        data = pd.concat([data, tmp])
        # TODO fix this
        detokenizer = TreebankWordDetokenizer()
        data['text'] = data.text.apply(lambda x: utils.detokenize(x, detokenizer))
        
        data['lens']  = data.text.apply(lambda x: len(x.split()))
        data = data[data.lens <= utils.PROQUEST_MAX_LEN]
        return data
    
    def make_dataset(self):
        records = self.extract_and_structure_annotations()
        inputs = {'train': [], 'test_az': [], 'test_mh': [], 'test_qz': [], 'test': []}
        # breakpoint()
        for split, record in records.items():
            record_df = pd.DataFrame.from_records(record)
            grpd = record_df.groupby(['assigned', 'text'])
            # for r in record:
            for name, r in grpd:
                instance = {"id": r['id'].iloc[0], "text": r['text'].iloc[0], "assigned": r['assigned'].iloc[0]}
                foreign = bool(r['foreign'].iloc[0])
                contains_narratives = bool(r['selections'].iloc[0])
                
                causes, effects = [], []
                for idx, row in r.iterrows():
                    if 'cause' in row['selections']:
                        causes += [s[0] for s in row['selections']['cause']]
                    if 'effect' in row['selections']:
                        effects += [s[0] for s in row['selections']['effect']]
                    
                if contains_narratives:
                    temporality = r['temporality']
                    inflation_direction = r['inflation_direction']
                    causes = [{"cause": s[0], "time": s[1]} for s in r['selections']['cause']] if 'cause' in r['selections'] else []
                    effects = [{"effect": s[0], "time": s[1]} for s in r['selections']['effect']] if 'effect' in r['selections'] else []
                    narratives = json.dumps(causes + effects)
                    instance['template'] = input_schema.strip()
                    try:
                        instance['data'] = "#".join([json.dumps(foreign), json.dumps(contains_narratives), temporality, json.dumps(inflation_direction), narratives])
                    except Exception as e:
                        print(e)
                        breakpoint()

                else:
                    instance['template'] = input_schema_no_narrative.strip()
                    instance['data'] = "#".join([json.dumps(foreign), json.dumps(contains_narratives)])
                    #input = input_schema_no_narrative.format(FOREIGN=foreign, CONTAINTS_NARRATIVE=contains_narratives)    
                inputs[split].append(instance)
           
        for i in range(len(inputs['test_az'])):
            az = inputs['test_az'][i]
            mh = inputs['test_mh'][i]
            qz = inputs['test_qz'][i]
            breakpoint()
        return inputs

    def extract_and_structure_annotations_std(self):
        records = {'train': [], 'test_az': [], 'test_mh': [], 'test_qz': [], 'test': []}
        for name, instance in self.data.iterrows():
            try:
                annotation_id = instance['annotation_id']
                assigned = instance['assigned']
                text = instance['text']
                id = instance['id']
                
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

                    inflation_direction = instance['inflation-direction'].split('-')[-1]
                    if inflation_direction == 'same':
                        inflation_direction = 'na'
                    for cat in categories:
                        selections[cat] = []
                        types = self.extract_choices(instance[cat])
                        
                        for t in types:
                            if t not in self.temps:
                                selections[cat].append((t, instance[f"{t}-time"]))
                    print(selections)
                else:
                    temporality = None
                    inflation_direction = None

                records[instance.split].append(
                    {
                        'annotation_id': annotation_id,
                        'id': id,
                        'assigned': assigned,
                        'text': text,
                        'foreign': foreign,
                        'inflation_direction': inflation_direction,
                        'temporality': temporality,
                        'selections': selections
                    }
                )
            except Exception as e:
                print(e)
                breakpoint()
                print(instance['id'], instance['assigned'])
        return records


if __name__ == "__main__":
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument("--labelstudio_export", "-ls", type=str, default='export_tsv_proquest_0721.tsv', help="Filename of the LabelStudio export")
    parse_args.add_argument("--export_format", "-f", type=str, default="standard", choices=["min", "standard"])
    args = parse_args.parse_args()

    interpreter = LabelStudioDataInterpreter(args.labelstudio_export, args.export_format)
    records = interpreter.make_dataset()

    breakpoint()
    ds = interpreter.write_dataset(records)
