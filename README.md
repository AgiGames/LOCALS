# LOCALS - Long Object Correlation-based Analysis and Localization System
Contains dataset, models and implementation source code of LOCALS.

## How to run code:
- Step 1: Make sure you have a conda pytorch gpu environment (look up setup videos on youtube). 
- Step 2: Run ```pip install locals-api```
- Step 3: Download JupyterLab on anaconda under the pytorch gpu environment you just created (assuming anaconda exists since you will have used conda to make pytorch gpu environment).
- Step 4: Download source code, necessary models and datasets needed for running (linked in *models* and *datasets* section).
- Step 5: Run code by launching JupyterLab and pressing the run button.

  ## Models:
  - [LOCALS](https://www.kaggle.com/models/agilesharumugam/locals/)
  
  ## Datasets:
  - [Traced Pencils](https://www.kaggle.com/datasets/agilesharumugam/traced-pencils)
  - [Rivers](https://www.kaggle.com/datasets/ankit1743/skyview-an-aerial-landscape-dataset)

  ## Paper Model and Dataset:
  - [LOCALS](https://zenodo.org/records/20646155)

## LOCALS Results:

- **Training Loss Curve**
  
  ![Training Loss Curve](https://github.com/AgiGames/LOCALS/blob/main/figures/smoothed_training_loss.png "Training Loss Curve")

- **Test Results**
  
  ![Test Results](https://github.com/AgiGames/LOCALS/blob/main/figures/visualised_predictions.png "Test Results")

- **Recall, Precision, F1-Score**

  ![Recall, Precision, F1-Score](https://github.com/AgiGames/LOCALS/blob/main/figures/recall-precision-f1score.png "Recall, Precision, F1-Score")

## Evaluated Metrics on Random Seeds

| Recall          | Precision       | F1 Score        | mAP             | mCS             |
|-----------------|-----------------|-----------------|-----------------|-----------------|
| 0.9488 ± 0.0093 | 0.9441 ± 0.0293 | 0.9461 ± 0.0130 | 0.9354 ± 0.0125 | 0.9313 ± 0.0392 |

---
> Personal use only, citation required for public use.
