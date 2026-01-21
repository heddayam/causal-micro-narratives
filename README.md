# Measuring Economic Narratives in Large-Scale News Corpora

A Python project for extracting and classifying economic narratives related to inflation using LLMs (SFT and in-context). 

Paper: [Causal-Micro Narratives](https://aclanthology.org/2024.wnu-1.12/)

## Project Structure

```
.
├── src/                           # Source code
│   ├── annotation/               # Tools and utilities for data annotation
│   ├── evaluation_and_analysis/  # Model evaluation and result analysis
│   ├── finetuning/              # Model fine-tuning and training code
│   ├── in_context/              # In-context learning and prompting
│   ├── preprocess/              # Data preprocessing and cleaning
│   ├── utils/                   # Shared utilities and helper functions
│   └── scrap/                   # Experimental code and analysis scripts
├── scripts/                      # Shell scripts for automation
├── data/                        # Data directory (not tracked in git)
├── pyproject.toml              # Poetry dependency management
└── poetry.lock                 # Lock file for dependencies
```

## Directory Descriptions

- **annotation/**: Data and code for processing annotations from LabelStudio. Also contains the HTML files for the annotation interface.
- **evaluation_and_analysis/**: Scripts for evaluating model performance, generating metrics, and analyzing results.
- **finetuning/**: Code for fine-tuning Phi-2 and Llama-3.1-8b using labeled data, including training configurations and scripts.
- **in_context/**: Code for zero and few-shot prompting of LLMs using API access. Uses OpenRouter.
- **preprocess/**: Data preprocessing pipelines, including cleaning, formatting, and other prep for modeling.
- **utils/**: Common utilities used across the project, such as data loading, caching, and helper functions.
- **scrap/**: Experimental/old analysis scripts and prototypes for testing new approaches.

## Installation

```bash
# Install Poetry (Python dependency management tool)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

## Setup
1. Copy `.env.example` to `.env`
2. Add your API keys and configurations to `.env`
3. Never commit `.env` to version control

## Development Pipeline

1. **Data Preprocessing**
   - Clean and format raw text data
   - Perform initial analysis (e.g., ngram analysis)
   - Prepare datasets for annotation and modeling

2. **Model Development**
   - Fine-tune models on annotated data
   - Implement in-context learning approaches
   - Develop and test prompting strategies

3. **Evaluation and Analysis**
   - Evaluate model performance
   - Generate visualizations and metrics
   - Analyze economic narrative patterns

## Batch Prediction Pipeline

For large-scale inference on the full ProQuest dataset (2010-2025):

1. **Run batch predictions** (on compute cluster with GPUs):
   ```bash
   ./scripts/run_batched_predict_2010_2025_updated.sh
   ```
   This submits SLURM jobs that save predictions to `/net/projects2/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/`

2. **Transfer predictions to local server** (if running aggregation on a different machine):
   ```bash
   scp -r mourad@fe01.ds.uchicago.edu:/net/projects2/chai-lab/mourad/narratives-data/model_json_preds/proquest/full_proquest/llama31_ft__600s_train-now_and_proquest_*_2010-2025_updated /home/mourad/causal-micro-narratives/tmp_batched/
   ```

3. **Aggregate predictions** to get shares of narratives per time/location:
   ```bash
   conda activate mourad-econ-py310
   python -m src.finetuning.process_batched_predictions \
     --dataset full_proquest \
     --model llama31_ft__600s \
     --train_ds now_and_proquest \
     --updated
   ```
   Output saved to `/data/mourad/narratives/regression_data/`

## License

