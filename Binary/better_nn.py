import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# =========================
# 1. Config & device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 128
num_epochs = 80              # more epochs, but early stopping will cut it
learning_rate = 1e-3
val_ratio = 0.2              # 20% for validation
patience = 8                 # early stopping patience

torch.manual_seed(42)
np.random.seed(42)

# =========================
# 2. Load preprocessed data
# =========================
train_df = pd.read_csv("train_preprocessed_topk.csv")
test_df_proc = pd.read_csv("test_preprocessed_topk.csv")
test_raw = pd.read_csv("test.csv")  # for founder_id later

# Split features/labels
y = train_df["retention_status"].values.astype(np.float32)  # 0/1
X = train_df.drop(columns=["retention_status"]).values.astype(np.float32)
X_test = test_df_proc.values.astype(np.float32)

input_dim = X.shape[1]
print("Input feature dimension:", input_dim)

# =========================
# 3. Stratified train/val split (better than random_split)
# =========================
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X,
    y,
    test_size=val_ratio,
    random_state=42,
    stratify=y
)

print("Train size:", X_train_np.shape[0], "Val size:", X_val_np.shape[0])

# =========================
# 4. Custom Dataset
# =========================
class TabularDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]

train_dataset = TabularDataset(X_train_np, y_train_np)
val_dataset   = TabularDataset(X_val_np, y_val_np)
full_dataset  = TabularDataset(X, y)          # for later full-data training if you want

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# =========================
# 5. Handle class imbalance with pos_weight
# =========================
num_pos = (y_train_np == 1).sum()
num_neg = (y_train_np == 0).sum()
print("Train class counts:", num_neg, "negatives,", num_pos, "positives")

# pos_weight = (#neg / #pos) so that minority class gets higher loss weight
pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)
print("pos_weight for BCEWithLogitsLoss:", pos_weight.item())

# =========================
# 6. Define the model
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # single logit for binary classification
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (batch,) logits

model = MLP(input_dim).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    verbose=True
)

# =========================
# 7. Training + early stopping
# =========================
def evaluate(loader, model):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

best_val_loss = float("inf")
best_state = None
epochs_no_improve = 0

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    total_train = 0
    total_correct_train = 0

    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * yb.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct_train += (preds == yb).sum().item()
        total_train += yb.size(0)

    train_loss = running_loss / total_train
    train_acc = total_correct_train / total_train

    val_loss, val_acc = evaluate(val_loader, model)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping tracking
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best validation model
if best_state is not None:
    model.load_state_dict(best_state)

final_model = model  # use this for test

# =========================
# 8. Predict on test data
# =========================
test_dataset = TabularDataset(X_test, y=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

final_model.eval()
all_preds = []

with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        logits = final_model(xb)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long().cpu().numpy()
        all_preds.append(preds)

all_preds = np.concatenate(all_preds, axis=0)  # shape (n_test,)
print("Test predictions shape:", all_preds.shape)

# Map 0/1 -> "Left"/"Stayed"
id_to_label = {0: "Left", 1: "Stayed"}
pred_labels = pd.Series(all_preds).map(id_to_label)

# =========================
# 9. Create predictions file (same format as sample_submission)
# =========================
predictions_df = pd.DataFrame({
    "founder_id": test_raw["founder_id"],
    "retention_status": pred_labels
})

predictions_df = predictions_df[["founder_id", "retention_status"]]
output_path = "predictions_topk.csv"
predictions_df.to_csv(output_path, index=False)

print(f"Saved GPU NN predictions to {output_path}")
print(predictions_df.head())
