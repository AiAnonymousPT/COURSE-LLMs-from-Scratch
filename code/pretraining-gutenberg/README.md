## Pre-training with Gutenberg books

**Setup**

This is made to run on Lightening AI in the default conda environment. 

Run all commands from the `pretraining-gutenberg` directory.

Steps
- If you are running on a local machine be sure to activate a virtual environment first `source .venv/bin/activate`
- Install required dependencies `pip install -r requirements.txt`


**Download books & train**



`python get_data.py` to download the first 100 books from Gutenberg

`python prepare_dataset.py` to filter out non-English text and merge into one document

`python pretraining_simple.py --n_epochs 1 --batch_size 4` - include `--debug True` to start a test run on a very small model