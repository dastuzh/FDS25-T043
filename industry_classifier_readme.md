# Industry Classification Notebook

## Overview

This Jupyter Notebook implements a **hybrid industry classification model** that combines:

1. **Text embeddings** of company business descriptions (processed using K-Nearest Neighbors with PCA).
2. **Financial features** (processed using HistGradientBoostingClassifier).

The hybrid approach fuses predictions from the text and financial branches using an **alpha-weighted probability combination** and computes a **confidence score** for each prediction.  

The notebook contains two main sections:

- **Cell 1:** Defines the utility functions and the `IndustryClassifier` class for easy fitting and prediction.  
- **Cell 2:** Implements a full pipeline with manual steps, including preprocessing, KNN, HGB, alpha fusion, and confidence computation, along with evaluation.

---

## Table of Contents

1. [Data](#data)
2. [Cell 1: IndustryClassifier Class](#cell-1-industryclassifier-class)
3. [Cell 2: Manual Pipeline](#cell-2-manual-pipeline)
4. [Usage Example](#usage-example)

---

## Data

The notebook expects two pickle files:

1. **Text embeddings:** `data/df_train.pkl`  
   Columns:  
   - `id` (unique company ID)  
   - `business_description_embedding` (list or string representation of float vectors)  
   - `industry` (string, target label)

2. **Financial data:** `data/df_financials_train.pkl`  
   Columns:  
   - `id` (unique company ID)  
   - `net_profit_margin`, `asset_turnover`  
   - `country_code`

> Both datasets are merged on `id` inside the notebook.

---

## Cell 1: IndustryClassifier Class

### Overview

The `IndustryClassifier` is a reusable class that encapsulates:

- Embedding expansion and PCA
- Financial feature preprocessing
- KNN for text embeddings (with calibrated probabilities)
- HistGradientBoostingClassifier for financial features (with calibrated probabilities)
- Alpha-weighted probability fusion
- Hybrid confidence score based on agreement and entropy

---

### Class Interface

```python
clf = IndustryClassifier(random_state=42)
```

- `random_state`: Optional; for reproducibility.
- Fixed hyperparameters internally:
  - `n_neighbors=10`
  - `n_components=75`
  - `alpha=0.65`

---

### Methods

#### `fit(path_text, path_fin)`

Trains the hybrid model.

- `path_text` (str): Path to text embeddings pickle file.
- `path_fin` (str): Path to financial data pickle file.

#### `predict(df_input, labels_decoded=False)`

Predicts industries and confidence.

- `df_input` (DataFrame): Must contain `business_description_embedding` and financial columns.  
- `labels_decoded` (bool, default=False):  
  - `False` → returns encoded labels and confidence  
  - `True` → returns decoded industry names and confidence

**Returns:**

- `labels_decoded=False` → `(y_pred_encoded, confidence)`  
- `labels_decoded=True` → `(y_pred_decoded, confidence)`

---

### Example Usage

```python
clf = IndustryClassifier()
clf.fit()

# Encoded predictions + confidence
y_pred_encoded, y_conf = clf.predict(df_test)
print(y_pred_encoded[:10])
print(y_conf[:10])

# Decoded labels + confidence
y_pred_decoded, y_conf = clf.predict(df_test, labels_decoded=True)
print(y_pred_decoded[:10])
print(y_conf[:10])
```

---

## Cell 2: Manual Pipeline

This cell contains a step-by-step implementation of the hybrid model:

1. **Data preparation:** Merge text embeddings and financial data, clean numeric columns, and create interaction features.
2. **Train/test split:** Stratified split based on `industry`.
3. **Embeddings expansion and PCA**
4. **Oversampling:** Using `RandomOverSampler` for financial features
5. **Text model:** KNN classifier with calibrated probabilities
6. **Financial model:** HistGradientBoostingClassifier with calibrated probabilities
7. **Alpha-weighted fusion:** Combine text and financial model predictions
8. **Hybrid confidence score:** Combines agreement and entropy
9. **Evaluation:** Computes Confidence-Weighted F1 (macro and weighted)

---

### Example Evaluation

```python
cw_f1_macro = multiclass_confidence_weighted_f1(y_test, y_pred_final, confidence_final, average='macro')
cw_f1_weighted = multiclass_confidence_weighted_f1(y_test, y_pred_final, confidence_final, average='weighted')

print("Confidence-weighted F1 (macro):", round(cw_f1_macro, 4))
print("Confidence-weighted F1 (weighted):", round(cw_f1_weighted, 4))
```

---

## Notes

- `n_neighbors`, `n_components`, and `alpha` are **fixed** in the class but can be manually changed in Cell 2 if desired.
- The `IndustryClassifier` class simplifies repeated use and deployment.
- Confidence score is normalized between 0 and 1 and reflects both **agreement** and **entropy** of predictions.
- `labels_decoded=True` allows you to get industry names instead of numeric labels.

---

This README ensures that anyone opening the notebook knows:

- How to load the data  
- How to train and predict using both the **class** and the **manual pipeline**  
- How confidence scores are computed

