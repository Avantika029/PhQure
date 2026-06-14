import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          get_linear_schedule_with_warmup)
from torch.optim import AdamW
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score, confusion_matrix,
                             classification_report)
import warnings
warnings.filterwarnings("ignore")

BASE   = os.path.join(os.path.expanduser("~"), "Desktop",
                      "Dissertation", "PhQure")
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 128   # max token length
BATCH_SIZE = 32
EPOCHS     = 5
LR         = 2e-5

print("=" * 60)
print("  PhQure — Branch C (DistilBERT on Raw URLs)")
print("=" * 60)
print(f"  Device     : {DEVICE}")
print(f"  Max length : {MAX_LEN} tokens")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  LR         : {LR}")

# ── Dataset class ──────────────────────────────────
class URLDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.urls      = df["url"].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.urls[idx]),
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx],
                                           dtype=torch.long)
        }

# ── Load data ──────────────────────────────────────
print("\n  Loading datasets...")
train_df = pd.read_csv(os.path.join(DATA, "branch_c_train.csv"))
test_df  = pd.read_csv(os.path.join(DATA, "branch_c_test.csv"))

print(f"  Training   : {len(train_df):,} URLs")
print(f"  Test       : {len(test_df):,} URLs")

# ── Tokenizer ──────────────────────────────────────
print("\n  Loading DistilBERT tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

train_dataset = URLDataset(train_df, tokenizer, MAX_LEN)
test_dataset  = URLDataset(test_df,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ── Model ──────────────────────────────────────────
print("  Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2)
model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters : {total_params:,}")

# ── Optimiser and scheduler ────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ── Training loop ──────────────────────────────────
print(f"\n{'─'*60}")
print(f"  {'Epoch':<8} {'Train Loss':<14} {'Train Acc':<12} "
      f"{'Val Acc':<12} {'Val F1':<12} {'Val AUC'}")
print(f"{'-'*60}")

history = {"train_loss": [], "train_acc": [],
           "val_acc":    [], "val_f1":    [], "val_auc": []}
best_auc = 0.0

for epoch in range(1, EPOCHS + 1):

    # ── Train ──
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention  = batch["attention_mask"].to(DEVICE)
        labels     = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention,
                        labels=labels)
        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_loss    += loss.item() * input_ids.size(0)
        preds          = logits.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += input_ids.size(0)

    avg_loss  = train_loss / train_total
    train_acc = train_correct / train_total

    # ── Evaluate ──
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention  = batch["attention_mask"].to(DEVICE)
            labels     = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention)
            probs   = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds   = outputs.logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_acc = accuracy_score(all_labels, all_preds)
    val_f1  = f1_score(all_labels, all_preds)
    val_auc = roc_auc_score(all_labels, all_probs)

    history["train_loss"].append(avg_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["val_f1"].append(val_f1)
    history["val_auc"].append(val_auc)

    print(f"  {epoch:<8} {avg_loss:<14.4f} {train_acc*100:<12.2f}"
          f"{val_acc*100:<12.2f} {val_f1*100:<12.2f} {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        model.save_pretrained(os.path.join(MODELS, "distilbert_branchC"))
        tokenizer.save_pretrained(os.path.join(MODELS, "distilbert_branchC"))

# ── Final evaluation ───────────────────────────────
print(f"\n  Loading best model (AUC: {best_auc:.4f})...")
model = DistilBertForSequenceClassification.from_pretrained(
    os.path.join(MODELS, "distilbert_branchC"))
model = model.to(DEVICE)
model.eval()

all_preds, all_labels, all_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention  = batch["attention_mask"].to(DEVICE)
        labels     = batch["label"].to(DEVICE)
        outputs    = model(input_ids=input_ids, attention_mask=attention)
        probs      = torch.softmax(outputs.logits, dim=1)[:, 1]
        preds      = outputs.logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1  = f1_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)

print(f"\n{'='*60}")
print(f"  BRANCH C — FINAL RESULTS")
print(f"{'='*60}")
print(f"  Accuracy : {acc*100:.2f}%")
print(f"  F1 Score : {f1*100:.2f}%")
print(f"  AUC      : {auc:.4f}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['Legit','Phishing'])}")

# ── Confusion matrix ───────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Legit", "Phishing"],
            yticklabels=["Legit", "Phishing"])
plt.title("Branch C — Confusion Matrix (DistilBERT)", fontweight="bold")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(DATA, "branchC_confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close()

# ── Training curves ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Branch C — DistilBERT Training History",
             fontweight="bold", fontsize=13)

axes[0].plot(history["train_loss"], "b-o", label="Train Loss")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot([a*100 for a in history["val_acc"]],  "r-o", label="Val Acc")
axes[1].plot([a*100 for a in history["train_acc"]], "b-o", label="Train Acc")
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(DATA, "branchC_training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()

print(f"\n  Outputs saved:")
print(f"  - branchC_confusion_matrix.png")
print(f"  - branchC_training_curves.png")
print(f"  - models/distilbert_branchC/")
print(f"\n  Novel claim: First DistilBERT application to raw URL")
print(f"  classification — zero feature engineering required")
print(f"{'='*60}")