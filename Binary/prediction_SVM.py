import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# =========================
# 1. Load preprocessed train & test
# =========================
train_df = pd.read_csv("train_preprocessed_feature_engg.csv")
test_df_proc = pd.read_csv("test_preprocessed_feature_engg.csv")

# Target is in train_preprocessed as 0/1
y = train_df["retention_status"]
X = train_df.drop(columns=["retention_status"])

print("Train shape:", X.shape, "Test shape:", test_df_proc.shape)

# =========================
# 2. Train/validation split to check performance
# =========================
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3. Define and train SVM
# =========================
svm_clf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=False,  # change to True if you later want predict_proba
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
# 5. Train on FULL train data (optional but recommended for final model)
# =========================
svm_clf.fit(X, y)

# =========================
# 6. Predict on preprocessed test
# =========================
test_preds_binary = svm_clf.predict(test_df_proc)  # 0/1

# Map 0/1 back to "Left"/"Stayed"
id_to_label = {0: "Left", 1: "Stayed"}
test_preds_label = pd.Series(test_preds_binary).map(id_to_label)

# =========================
# 7. Build predictions.csv with founder_id from ORIGINAL test.csv
# =========================
test_raw = pd.read_csv("test.csv")  # to get founder_id

predictions_df = pd.DataFrame({
    "founder_id": test_raw["founder_id"],
    "retention_status": test_preds_label
})

# Ensure same column order as sample_submission
predictions_df = predictions_df[["founder_id", "retention_status"]]

# =========================
# 8. Save to CSV
# =========================
predictions_path = "predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

print(f"Saved predictions to {predictions_path}")
print(predictions_df.head())
