# Founder Retention (Binary) & Multidimensional Personality Cluster Prediction (Multi-Class)  
Machine Learning Project (SVM + Neural Network)

---

## 📌 Overview  
This project contains **two related ML tasks**:

### **1. Binary Classification — Founder Retention**
Predict whether a startup founder:  
- **Stayed** (1)  
- **Left** (0)

### **2. Multi-Class Classification — Multidimensional Personality Cluster Prediction**
Predict a **5-class personality cluster label** (e.g. `Cluster_A` … `Cluster_E`)  
based on behavioral, lifestyle, and psychometric features.

For both tasks, we use:

- **Multi-Layer Perceptron (Neural Network)**
- **Support Vector Machine (SVM)**

➡️ In **both** binary and multi-class problems, the **Neural Network consistently outperformed SVM**.

---

# 🔧 1. Data Preprocessing Pipeline (Binary — Founder Retention)

All preprocessing is applied to both train and test (except label).  
Output CSVs for the binary task:

- `train_preprocessed_feature_engg.csv`
- `test_preprocessed_feature_engg.csv`

> The multi-class personality task uses an analogous preprocessing strategy  
> (scaling, encoding, and feature engineering on its own train/test files).

---

## 1.1 Feature Engineering (Founder Retention)

### ✔ Ordinal Encoding for Rating Columns
Converted rating-like textual fields into numeric levels (1–4 or 1–5):

- `work_life_balance_rating`
- `venture_satisfaction`
- `startup_performance_rating`
- `startup_reputation`
- `founder_visibility`
- `team_size_category`
- `startup_stage`
- `education_background`

Also created:
```text
<column>_ord_missing → binary flag for missing ordinal values
```

---

### ✔ Founder Age Binning
```text
0–25 → 0  
26–35 → 1  
36–45 → 2  
46–55 → 3  
56+  → 4
```

---

### ✔ Dependents & Family Features
- `has_dependents`
- `num_dependents_missing`
- `is_married`
- `is_single`

---

### ✔ Tenure & Startup Age
- `tenure_ratio`
- `tenure_gap`
- `years_since_founding_missing`

---

### ✔ Revenue Features
- `monthly_revenue_log`  
- `revenue_missing`
- `revenue_per_year_with_founder`
- `revenue_per_funding_round`

---

### ✔ Yes/No → Binary Conversion
Added `_bin` columns for:

- `working_overtime`
- `remote_operations`
- `leadership_scope`
- `innovation_support`

---

### ✔ Interaction Features
- `remote_x_distance`
- `support_count`
- `satisfaction_minus_perf`
- `reputation_x_visibility`

---

## 1.2 Additional Preprocessing

### ✔ Global Missingness Indicators
```text
<column>_was_missing → 1 if value was missing
```

---

### ✔ Rare Category Grouping
For all categorical columns:  
Categories representing **<1%** of the data are replaced with `"Other"`.

---

### ✔ Outlier Clipping (IQR method)
For each numeric feature:
```text
clip to [Q1 – 3·IQR,  Q3 + 3·IQR]
```

---

### ✔ Log Transforms
Applied to skewed fields when present:
- `distance_from_investor_hub_log`
- `years_since_founding_log`

---

## 1.3 ColumnTransformer

Numeric Pipeline:  
- Median imputation  
- Standard scaling  

Categorical Pipeline:  
- Most frequent imputation  
- OneHotEncoder  

The final design matrix is saved to the preprocessed CSVs listed above.

---

# 🤖 2. Models — Binary Founder Retention

After preprocessing, identical datasets are used for both SVM and NN.

---

## 🔷 2.1 Support Vector Machine (Binary)

### Model:
```python
SVC(kernel="rbf", C=1.0, gamma="scale")
```

### Pipeline:
1. Load preprocessed train/test
2. Stratified 80/20 split
3. Train SVM classifier
4. Validate on validation split
5. Retrain on full training data
6. Predict on test
7. Convert:
   - 0 → Left  
   - 1 → Stayed  
8. Save submission

### Performance:
- **Reasonable**, but **significantly lower** than the Neural Network model.

---

## 🔷 2.2 Neural Network (Binary)

### Architecture:
```text
Input
 → Linear(256) → ReLU → Dropout(0.3)
 → Linear(128) → ReLU → Dropout(0.3)
 → Linear(1)   → BCEWithLogitsLoss
```

### Training Details:
- Loss: `BCEWithLogitsLoss` with **pos_weight** to handle class imbalance  
- Optimizer: Adam (lr = 1e-3)  
- Scheduler: ReduceLROnPlateau  
- Early stopping (patience = 8)  
- Stratified train/validation split  

### Output:
```text
sigmoid(logits) >= 0.5 → {0,1}
```

### Result:
👉 **Neural Network achieved higher validation accuracy and better generalization**  
than SVM for the founder retention task.

---

# 🎯 3. Multidimensional Personality Cluster Prediction (Multi-Class)

A separate dataset is used for this task, with:

- Feature columns representing **behavioral, lifestyle, and psychometric scores**
- Target column: `personality_cluster` with **5 classes**  
  (e.g. `Cluster_A`, `Cluster_B`, `Cluster_C`, `Cluster_D`, `Cluster_E`)

Preprocessing is analogous:
- Basic feature engineering for numerical / rating-style inputs  
- Scaling of numeric features  
- Encoding of categorical attributes  
- Train/validation split for model evaluation  

---

## 🔷 3.1 SVM (Multi-Class)

Using the same RBF kernel SVM in multi-class mode:

```python
SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovr")
```

- Trained on preprocessed personality features  
- Predicts a label in `{Cluster_A, …, Cluster_E}`  

---

## 🔷 3.2 Neural Network (Multi-Class)

### Modified Architecture:
```text
Input
 → Linear(256) → ReLU → Dropout
 → Linear(128) → ReLU → Dropout
 → Linear(5)   → CrossEntropyLoss
```

### Training:
- Loss: `CrossEntropyLoss`  
- Optimizer: Adam  
- Early stopping strategy similar to binary case  

### Prediction:
```text
argmax(logits, dim=1) → class index → mapped to Cluster_A … Cluster_E
```

### Result:
👉 **Neural Network outperformed SVM on the personality cluster prediction task**  
with higher overall accuracy and better class-wise performance.

---

# 🏁 Summary

| Task                                      | Model           | Performance      |
|-------------------------------------------|-----------------|------------------|
| Founder Retention (Binary)               | **Neural Net**  | ⭐ Best           |
| Founder Retention (Binary)               | SVM             | Good but weaker  |
| Personality Cluster (5-Class, Multi-Class)| **Neural Net**  | ⭐ Best           |
| Personality Cluster (5-Class, Multi-Class)| SVM             | Lower accuracy   |

---

# 📦 Files (Binary Task)

| File                                 | Description                               |
|--------------------------------------|-------------------------------------------|
| `train_preprocessed_feature_engg.csv`| Full processed training features          |
| `test_preprocessed_feature_engg.csv` | Full processed test features              |
| `predictions_gpu_nn.csv`            | Binary NN predictions (retention)         |
| `predictions.csv`                   | Binary SVM predictions (retention)        |

> For the multi-class personality dataset, analogous preprocessed train/test  
> and predictions files are generated (with dataset-specific filenames).

---

# 🚀 Final Note  

The pipelines are fully modular and can be extended with:

- Gradient boosting models (XGBoost, LightGBM, CatBoost)  
- Ensembling (stacking / blending NN + tree models)  
- SHAP-based interpretability  
- Hyperparameter tuning (GridSearch / Optuna / Bayesian optimization)

Feel free to plug in additional models on top of the preprocessed data.
