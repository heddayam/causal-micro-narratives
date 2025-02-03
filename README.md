# Economic Narratives CLS

A Python project for analyzing economic narratives using various language models and classification techniques.

## Project Structure

```
.
├── src/               # Source code
│   ├── analysis/     # Analysis modules
│   ├── annotation/   # Annotation related code
│   ├── evaluation/   # Model evaluation code
│   ├── finetune/     # Model fine-tuning code
│   ├── preprocess/   # Data preprocessing
│   ├── prompt/       # Prompting utilities
│   └── utils/        # Shared utilities
├── scripts/          # Shell scripts for various operations
├── data/            # Data directory
└── tests/           # Test suite (to be added)
```

## Installation

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

## Setup
1. Copy `.env.example` to `.env`
2. Add your API keys to `.env`
3. Never commit `.env` to version control

## Logical Pipeline

1. Preprocess data
    - preprocess
        - ngram analysis
2. Fine-tune model
3. Evaluate model



## License

[Add your license information here] 