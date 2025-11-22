import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =========================
# 1. Load preprocessed train & test
# =========================
train_df = pd.read_csv("train_preprocessed_feature_engg.csv")
test_df_proc = pd.read_csv("test_preprocessed_feature_engg.csv")

# Target is now the 5-class label
y = train_df["personality_cluster"]                 # strings like 'Cluster_A' ... 'Cluster_E'

# Drop target + ID from features
X = train_df.drop(columns=["personality_cluster", "participant_id"])
X_test = test_df_proc.drop(columns=["participant_id"])

print("Train shape:", X.shape, "Test shape:", X_test.shape)

# =========================
# 2. Train/validation split to check performance
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. Define and train SVM (multiclass via one-vs-one)
# =========================
svm_clf = SVC(
    kernel="rbf",
    C=3.0,           # you can tune this (1.0, 3.0, 5.0 ...)
    gamma="scale",
    probability=False,
    random_state=42
)

svm_clf.fit(X_train, y_train)

# =========================
# 4. Validate
# =========================
y_valid_pred = svm_clf.predict(X_valid)
val_acc = accuracy_score(y_valid, y_valid_pred)
print(f"Validation accuracy: {val_acc:.4f}")

# =========================
# 5. Train on FULL train data (for final model)
# =========================
svm_clf.fit(X, y)

# =========================
# 6. Predict on preprocessed test
# =========================
test_preds = svm_clf.predict(X_test)   # directly returns cluster labels

# =========================
# 7. Build predictions.csv with participant_id from ORIGINAL test.csv
# =========================
test_raw = pd.read_csv("test.csv")  # to get participant_id

predictions_df = pd.DataFrame({
    "participant_id": test_raw["participant_id"],
    "personality_cluster": test_preds
})

# Ensure correct column order
predictions_df = predictions_df[["participant_id", "personality_cluster"]]

# =========================
# 8. Save to CSV
# =========================
predictions_path = "svm_predictions_multiclass.csv"
predictions_df.to_csv(predictions_path, index=False)

print(f"Saved predictions to {predictions_path}")
print(predictions_df.head())
