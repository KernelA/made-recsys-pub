# Item2Item, ALS, IALS

## Solution

### Colab preview

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KernelA/made-recsys-pub/blob/master/main.ipynb)

### Local file

[main.ipynb](./main.ipynb)

## Requirements

1. Python 3.10 or higher.
2. faiss-cpu
3. NVIDIA CUDA 11.x

## How to run

Install dependencies:
```
pip install -r. /requirements.txt
```

For development:
```
pip install -r ./requirements.txt -r ./requirements.dev.txt
```

Run:
```
dvc repro -R .
```

Open `main.ipynb` and execute all cells.

