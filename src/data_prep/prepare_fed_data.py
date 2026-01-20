"""
Prepare Federal Reserve data for the narrative inference pipeline.

Converts the Fed inflation sentences CSV to a pipeline-compatible JSONL.gz format
that can be processed by predict_json.py.

Usage:
    python -m src.data_prep.prepare_fed_data
"""

import csv
import gzip
import json
from pathlib import Path

# Constants
INPUT_PATH = "/data/mourad/narratives/fed_data/fed_inflation_sentences_20260107_063031.csv"
OUTPUT_PATH = "/data/mourad/narratives/fed_data/fed_processed.jsonl.gz"


def prepare_fed_data(input_path: str = INPUT_PATH, output_path: str = OUTPUT_PATH) -> int:
    """
    Convert Fed CSV to pipeline-compatible JSONL.gz format.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output JSONL.gz file

    Returns:
        Number of records written
    """
    print(f"Reading Fed data from {input_path}")

    records_written = 0
    records_skipped = 0

    with open(input_path, 'r', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)

        with gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
            for row in reader:
                # Parse date_event (format: YYYY-MM-DD)
                date_event = row.get('date_event', '')
                if not date_event or len(date_event) < 7:
                    records_skipped += 1
                    continue

                try:
                    year = int(date_event[:4])
                    month = int(date_event[5:7])
                    year_month = f"{year:04d}-{month:02d}"
                except (ValueError, IndexError):
                    records_skipped += 1
                    continue

                # Skip empty sentences
                sentence = row.get('sentence', '').strip()
                if not sentence:
                    records_skipped += 1
                    continue

                # Create file_id from filename
                filename = row.get('filename', '')
                file_id = filename.replace('.pdf', '') if filename else ''

                # Build output record
                record = {
                    'text': sentence,
                    'file_id': file_id,
                    'year_month': year_month,
                    'year': year,
                    'month': month,
                    'doc_type': row.get('type', ''),
                    'speaker': row.get('speaker', ''),
                    'speaker_title': row.get('speaker_title', ''),
                    'venue': row.get('venue', ''),
                    'date_released': row.get('date_released', ''),
                }

                f_out.write(json.dumps(record) + '\n')
                records_written += 1

    print(f"Wrote {records_written:,} records to {output_path}")
    if records_skipped > 0:
        print(f"Skipped {records_skipped:,} records (missing date or sentence)")

    return records_written


if __name__ == "__main__":
    prepare_fed_data()
