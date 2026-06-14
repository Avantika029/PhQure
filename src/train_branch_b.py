import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings("ignore")

BASE     = os.path.join(os.path.expanduser("~"), "Desktop",
                        "Dissertation", "PhQure")
QR_PHISH = os.path.join(BASE, "data", "qr_images", "phishing")
QR_LEGIT = os.path.join(BASE, "data", "qr_images", "legitimate")
MODELS   = os.path.join(BASE, "models")
DATA     = os.path.join(BASE, "data")
os.makedirs(MODELS, exist_ok=True)

DEVICE     = torch.device("cpu")
BATCH_SIZE = 16
EPOCHS     = 15
LR         = 0.001
N_SAMPLES  = 15000   # per class

print("=" * 60)
print("  PhQure — Branch B (EfficientNet-B0 on QR Images)")
print("=" * 60)
print(f"  Device     : CPU")
print(f"  Samples    : {N_SAMPLES*2:,} ({N_SAMPLES:,} per class)")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")

# ── Dataset ────────────────────────────────────────
class QRDataset(Dataset):
    def __init__(self, phish_dir, legit_dir, n_samples, transform):
        self.transform = transform
        self.samples   = []

        phish_files = sorted(os.listdir(phish_dir))[:n_samples]
        legit_files = sorted(os.listdir(legit_dir))[:n_samples]

        for f in phish_files:
            self.samples.append((os.path.join(phish_dir, f), 1))
        for f in legit_files:
            self.samples.append((os.path.join(legit_dir, f), 0))

        np.random.seed(42)
        np.random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), torch.tensor(label, dtype=torch.long)

# ── Transforms ─────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Load data ──────────────────────────────────────
full_dataset = QRDataset(QR_PHISH, QR_LEGIT, N_SAMPLES, val_tf)
total        = len(full_dataset)
train_size   = int(0.8 * total)
val_size     = total - train_size

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42))

train_dataset.dataset = QRDataset(QR_PHISH, QR_LEGIT, N_SAMPLES, train_tf)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print(f"\n  Training samples   : {train_size:,}")
print(f"  Validation samples : {val_size:,}")

# ── Model — EfficientNet-B0 ────────────────────────
print("\n  Loading EfficientNet-B0 (pretrained on ImageNet)...")
model = models.efficientnet_b0(weights="IMAGENET1K_V1")

# Replace classifier head for binary classification
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(model.classifier[1].in_features, 2)
)
model = model.to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")

# ── Loss and optimiser ─────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── Training loop ──────────────────────────────────
print(f"\n{'─'*60}")
print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} {'Val Acc':<12} {'Val F1'}")
print(f"{'─'*60}")

history = {"train_loss": [], "train_acc": [],
           "val_acc":    [], "val_f1":    []}
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):

    # ── Train ──
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * imgs.size(0)
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += imgs.size(0)

    scheduler.step()
    avg_loss  = train_loss / train_total
    train_acc = train_correct / train_total

    # ── Validate ──
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1  = f1_score(all_labels, all_preds)

    history["train_loss"].append(avg_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)

    print(f"  {epoch:<8} {avg_loss:<14.4f} {train_acc*100:<12.2f} "
          f"{val_acc*100:<12.2f} {val_f1*100:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),
                   os.path.join(MODELS, "efficientnet_branchB.pth"))

# ── Final evaluation ───────────────────────────────
print(f"\n  Loading best model for final evaluation...")
model.load_state_dict(torch.load(os.path.join(MODELS,
                      "efficientnet_branchB.pth")))
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs   = imgs.to(DEVICE)
        outputs = model(imgs)
        probs   = torch.softmax(outputs, dim=1)[:, 1]
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"\n{'='*60}")
print(f"  BRANCH B — FINAL RESULTS")
print(f"{'='*60}")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  F1 Score : {f1*100:.2f}%")
print(f"  AUC      : {auc:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['Legit','Phishing'])}")

# ── Confusion matrix ───────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Legit","Phishing"],
            yticklabels=["Legit","Phishing"])
plt.title("Branch B — Confusion Matrix (EfficientNet-B0)", fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(DATA, "branchB_confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Training curves ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Branch B — EfficientNet-B0 Training History",
             fontweight="bold", fontsize=13)

axes[0].plot(history["train_loss"], "b-o", label="Train Loss")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([a*100 for a in history["train_acc"]],
             "b-o", label="Train Acc")
axes[1].plot([a*100 for a in history["val_acc"]],
             "r-o", label="Val Acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA, "branchB_training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  Outputs saved:")
print(f"  - branchB_confusion_matrix.png")
print(f"  - branchB_training_curves.png")
print(f"  - models/efficientnet_branchB.pth")
print(f"\n  Target to beat : AUC 0.9133 (Trad & Chehab, 2025)")
print(f"  Your AUC       : {auc:.4f}")
if auc > 0.9133:
    print(f"  Result         : BENCHMARK BEATEN ✓")
else:
    print(f"  Result         : More epochs needed — run again with EPOCHS=20")
print(f"{'='*60}")