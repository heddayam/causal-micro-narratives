"""
Script to combine and process NOW and Proquest human annotation data.
"""

from typing import Dict, Any
from datasets import Dataset, concatenate_datasets, interleave_datasets
from src.utils import utils


def remove_distinct_fields(instance: Dict[str, Any], version: str) -> Dict[str, Any]:
    """
    Remove dataset-specific fields from the template and data strings.
    
    Args:
        instance: Dictionary containing 'template' and 'data' fields
        version: Dataset version ('now' or 'proquest')
    
    Returns:
        Modified instance with standardized fields
    """
    template = instance['template'].split('#')
    try:
        # Different datasets have different field markers we need to remove
        if version == 'now':
            idx = template.index('",\n    "counter-narrative": ')
        else:
            idx = template.index('",\n    "inflation-direction": ')
    except ValueError:
        # Return unchanged if marker not found
        return instance
    
    # Reconstruct template and data without the specific field
    template[idx+1] = '"' + template[idx+1] 
    instance['template'] = '#'.join(template[:idx] + template[idx+1:])    
    data = instance['data'].split('#')
    instance['data'] = "#".join(data[:idx] + data[idx+1:])
    return instance


def main():
    """Main function to load, process, and combine datasets."""
    # Load both datasets
    now_dataset = utils.load_hf_dataset(dataset='now')
    proquest_dataset = utils.load_hf_dataset(dataset='proquest')

    # Process each split separately
    splits = ['train', 'test', 'test_az', 'test_qz', 'test_mh']
    
    # Initialize combined dataset with NOW data
    combined_ds = now_dataset
    
    for split in splits:
        # Add source column to identify dataset origin
        now_dataset[split] = now_dataset[split].add_column('source', ['now'] * len(now_dataset[split]))
        proquest_dataset[split] = proquest_dataset[split].add_column('source', ['proquest'] * len(proquest_dataset[split]))
        
        # Remove dataset-specific fields
        now_dataset[split] = now_dataset[split].map(remove_distinct_fields, fn_kwargs={'version': 'now'})
        proquest_dataset[split] = proquest_dataset[split].map(remove_distinct_fields, fn_kwargs={'version': 'proquest'})
        
        # Combine datasets and shuffle
        combined_ds[split] = concatenate_datasets([now_dataset[split], proquest_dataset[split]])
        combined_ds[split] = combined_ds[split].shuffle(seed=0)

    # Save combined dataset
    combined_ds.save_to_disk("/data/mourad/narratives/sft_data_now_and_proquest")


if __name__ == "__main__":
    main()