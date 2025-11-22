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
num_epochs = 80              # early stopping will cut it
learning_rate = 1e-3
val_ratio = 0.2              # 20% for validation
patience = 8                 # early stopping patience

torch.manual_seed(42)
np.random.seed(42)

# =========================
# 2. Load preprocessed data
# =========================
train_df = pd.read_csv("train_preprocessed_feature_engg.csv")
test_df_proc = pd.read_csv("test_preprocessed_feature_engg.csv")
test_raw = pd.read_csv("test.csv")  # for participant_id later

# ----- Encode target (5 classes) -----
class_labels = sorted(train_df["personality_cluster"].unique())
label_to_idx = {lab: i for i, lab in enumerate(class_labels)}
idx_to_label = {i: lab for lab, i in label_to_idx.items()}

y = train_df["personality_cluster"].map(label_to_idx).values.astype(np.int64)

# Drop target + ID from train, ID from test
X = train_df.drop(columns=["personality_cluster", "participant_id"]).values.astype(np.float32)
X_test = test_df_proc.drop(columns=["participant_id"]).values.astype(np.float32)

num_classes = len(class_labels)
input_dim = X.shape[1]
print("Input feature dimension:", input_dim)
print("Num classes:", num_classes, class_labels)

# =========================
# 3. Stratified train/val split
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=True)

# =========================
# 5. Define multi-class model
# =========================
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # logits for each class
        )

    def forward(self, x):
        return self.net(x)  # (batch, num_classes)

model = MLP(input_dim, num_classes).to(device)

# CrossEntropyLoss expects logits + int64 labels (0..C-1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=3,
    verbose=True
)

# =========================
# 6. Evaluation
# =========================
def evaluate(loader, model):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device, dtype=torch.long)

            logits = model(xb)                    # (batch, num_classes)
            loss = criterion(logits, yb)          # scalar

            preds = logits.argmax(dim=1)          # (batch,)
            total_correct += (preds == yb).sum().item()
            total_loss += loss.item() * yb.size(0)
            total_samples += yb.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy

# =========================
# 7. Training + early stopping
# =========================
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
        yb = yb.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * yb.size(0)
        preds = logits.argmax(dim=1)
        total_correct_train += (preds == yb).sum().item()
        total_train += yb.size(0)

    train_loss = running_loss / total_train
    train_acc = total_correct_train / total_train

    val_loss, val_acc = evaluate(val_loader, model)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss - 1e-4:
        best_val_loss = val_loss
        best_state = model.state_dict()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Load best model
if best_state is not None:
    model.load_state_dict(best_state)

final_model = model

# =========================
# 8. Predict on test data
# =========================
test_dataset = TabularDataset(X_test, y=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=0, pin_memory=True)

final_model.eval()
all_preds = []

with torch.no_grad():
    for xb in test_loader:
        xb = xb.to(device)
        logits = final_model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)

all_preds = np.concatenate(all_preds, axis=0)  # shape (n_test,)
print("Test predictions shape:", all_preds.shape)

# Map indices -> cluster labels
pred_labels = pd.Series(all_preds).map(idx_to_label)

# =========================
# 9. Create predictions file
# =========================
predictions_df = pd.DataFrame({
    "participant_id": test_raw["participant_id"],
    "personality_cluster": pred_labels
})

predictions_df = predictions_df[["participant_id", "personality_cluster"]]
output_path = "predictions_gpu_nn_multiclass.csv"
predictions_df.to_csv(output_path, index=False)

print(f"Saved predictions to {output_path}")
print(predictions_df.head())
