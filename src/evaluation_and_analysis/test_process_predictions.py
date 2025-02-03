"""
Test the flatten_json_to_df function in process_predictions_and_evaluate.py

To run:
python -m unittest src/evaluation_and_analysis/test_process_predictions.py -v
"""

import unittest
import pandas as pd
import json
from process_predictions_and_evaluate import flatten_json_to_df

class TestFlattenJsonToDf(unittest.TestCase):
    def test_basic_narrative(self):
        """Test basic case with a single narrative"""
        input_json = {
            "foreign": False,
            "contains-narrative": True,
            "inflation-narratives": {
                "inflation-time": "present",
                "inflation-direction": "up",
                "narratives": [
                    {"cause": "monetary", "time": "present"}
                ]
            }
        }
        
        metadata_df, narrative_df = flatten_json_to_df(input_json)
        
        # Check metadata
        self.assertEqual(metadata_df['foreign'].iloc[0], False)
        self.assertEqual(metadata_df['contains-narrative'].iloc[0], True)
        self.assertEqual(metadata_df['inflation-time'].iloc[0], 'present')
        self.assertEqual(metadata_df['inflation-direction'].iloc[0], 'up')
        
        # Check narrative
        self.assertEqual(narrative_df['category'].iloc[0], 'monetary')
        self.assertEqual(narrative_df['type'].iloc[0], 'cause')
        self.assertEqual(narrative_df['time'].iloc[0], 'present')

    def test_multiple_narratives(self):
        """Test case with multiple narratives"""
        input_json = {
            "foreign": False,
            "contains-narrative": True,
            "inflation-narratives": {
                "inflation-time": "present",
                "inflation-direction": "up",
                "narratives": [
                    {"cause": "monetary", "time": "present"},
                    {"effect": "wages", "time": "future"}
                ]
            }
        }
        
        metadata_df, narrative_df = flatten_json_to_df(input_json)
        
        self.assertEqual(len(narrative_df), 2)
        self.assertEqual(narrative_df['category'].iloc[0], 'monetary')
        self.assertEqual(narrative_df['category'].iloc[1], 'wages')
        self.assertEqual(narrative_df['type'].iloc[0], 'cause')
        self.assertEqual(narrative_df['type'].iloc[1], 'effect')

    def test_no_narrative(self):
        """Test case when there's no narrative"""
        input_json = {
            "foreign": False,
            "contains-narrative": False,
            "inflation-narratives": {
                "inflation-time": "",
                "inflation-direction": "",
                "narratives": []
            }
        }
        
        metadata_df, narrative_df = flatten_json_to_df(input_json)
        
        self.assertEqual(len(narrative_df), 1)
        self.assertEqual(narrative_df['category'].iloc[0], 'none')
        self.assertEqual(narrative_df['type'].iloc[0], '')
        self.assertEqual(narrative_df['time'].iloc[0], '')

    def test_string_input(self):
        """Test handling of string JSON input"""
        input_str = '''{"foreign": false, "contains-narrative": true, 
                       "inflation-narratives": {
                           "inflation-time": "present",
                           "inflation-direction": "up",
                           "narratives": [{"cause": "monetary", "time": "present"}]
                       }}'''
        
        metadata_df, narrative_df = flatten_json_to_df(input_str)
        
        self.assertEqual(metadata_df['foreign'].iloc[0], False)
        self.assertEqual(narrative_df['category'].iloc[0], 'monetary')

    def test_escaped_quotes(self):
        """Test handling of escaped quotes in JSON string"""
        input_str = '''{"foreign": false, "contains-narrative": true, 
                       "inflation-narratives": {
                           "inflation-time": "present",
                           "inflation-direction": "up",
                           "narratives": [{\\"cause\\": \\"monetary\\", \\"time\\": \\"present\\"}]
                       }}'''
        
        metadata_df, narrative_df = flatten_json_to_df(input_str)
        
        self.assertEqual(metadata_df['foreign'].iloc[0], False)
        self.assertEqual(narrative_df['category'].iloc[0], 'monetary')

    def test_narrative_as_string(self):
        """Test handling of narratives provided as string instead of object"""
        input_json = {
            "foreign": False,
            "contains-narrative": True,
            "inflation-narratives": {
                "inflation-time": "present",
                "inflation-direction": "up",
                "narratives": ['{"cause": "monetary", "time": "present"}']
            }
        }
        
        metadata_df, narrative_df = flatten_json_to_df(input_json)
        
        self.assertEqual(narrative_df['category'].iloc[0], 'monetary')
        self.assertEqual(narrative_df['time'].iloc[0], 'present')

    def test_none_narratives(self):
        """Test handling of None narratives"""
        input_json = {
            "foreign": False,
            "contains-narrative": False,
            "inflation-narratives": None
        }
        
        metadata_df, narrative_df = flatten_json_to_df(input_json)
        
        self.assertEqual(len(narrative_df), 1)
        self.assertEqual(narrative_df['category'].iloc[0], 'none')

if __name__ == '__main__':
    unittest.main() 