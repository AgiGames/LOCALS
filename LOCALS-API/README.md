# LOCALS

Rapid experimentation framework for the LOCALS detector.

## Installation

Install PyTorch according to your CUDA version.

Then:

```bash
pip install locals-api
```

## Example

```python
from locals import run

run(
    seed=1111,
    train_split=0.8,
    test_split=0.1,
    images_dir="dataset/images",
    labels_dir="dataset/labels",
    num_epochs=50,
)
```