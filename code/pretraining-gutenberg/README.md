## Pre-training with Gutenberg books

**Setup**

- Ensure you have `uv` installed (https://docs.astral.sh/uv/getting-started/installation/)
- Run `uv sync` to update dependencies


**Download books & train**

Run all commands from the `pretraining-gutenberg` directory

`uv run get_data.py` to download the first 100 books from Gutenberg

`uv run prepare_dataset.py` to filter out non-English text and merge into one document

`uv run pretraining_simple.py --n_epochs 1 --batch_size 4 --debug True`