"""
Prepare Federal Reserve data for the narrative inference pipeline.

Converts the Fed inflation sentences CSV to a pipeline-compatible JSONL.gz format
that can be processed by predict_json.py.

Usage:
    python -m src.data_prep.prepare_fed_data
    python -m src.data_prep.prepare_fed_data --input /path/to/fed_inflation_sentences_TIMESTAMP.csv
"""

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

# Constants
DEFAULT_INPUT_DIR = "/data/mourad/narratives/fed_data"


def prepare_fed_data(input_path: str, output_path: str = None) -> int:
    """
    Convert Fed CSV to pipeline-compatible JSONL.gz format.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output JSONL.gz file (auto-generated if not provided)

    Returns:
        Number of records written
    """
    input_path = Path(input_path)

    # Auto-generate output path preserving timestamp from input filename
    if output_path is None:
        # Extract timestamp from input filename (e.g., fed_inflation_sentences_20260120_215833.csv)
        match = re.search(r'_(\d{8}_\d{6})\.csv$', input_path.name)
        if match:
            timestamp = match.group(1)
            output_filename = f"fed_processed_{timestamp}.jsonl.gz"
        else:
            output_filename = "fed_processed.jsonl.gz"
        output_path = input_path.parent / output_filename

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
                    'index': row.get('index', ''),  # Preserve original index
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
    parser = argparse.ArgumentParser(description="Prepare Fed data for inference pipeline")
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input CSV file (e.g., fed_inflation_sentences_20260120_215833.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output JSONL.gz file (auto-generated if not provided)"
    )
    args = parser.parse_args()
    prepare_fed_data(args.input, args.output)
