# Music & Mental Health Survey Results

## Python Environment Setup

- Python 3.10 or higher is recommended
- Required libraries:
    - pandas
    - kagglehub
    - matplotlib

**Create an environment**
```bash
python3 -m venv .venv
```

**Activate environment**
```bash
# MacOS or Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

**Install packages**
```bash
pip install -r requirements.txt
```

**Run project**
```bash
# MacOS or Linux
python3 main.py

# Windows
python main.py
```

## Data Source

The dataset used in this project is sourced from Kaggle. You can find the
dataset [here](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results).

## Project Structure

- main.py: The main script to run the analysis.
- FileLoader.py: Module for loading the dataset and preprocessing.
