"""
RoBERTa-based Malicious Content Classifier
===========================================
Fine-tunes roberta-base for binary classification of phishing/smishing vs benign.

Author: Applied NLP / Security Team
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, RobertaForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from transformers import DataCollatorWithPadding
import warnings
warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
INPUT_FILE = "master_corpus_v2.csv"
MODEL_NAME = "roberta-base"
OUTPUT_DIR = "./roberta_malicious_classifier"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATASET CLASS
# =========================
class MaliciousDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# =========================
# WEIGHTED TRAINER
# =========================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# =========================
# METRICS COMPUTATION
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)
    
    return {
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs)
    }


# =========================
# MAIN PIPELINE
# =========================
def main():
    print("=" * 70)
    print("ROBERTA MALICIOUS CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Step 1: Load data
    print("\n[1/6] Loading dataset...")
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["clean_text"]).reset_index(drop=True)
    df["y"] = df["attack_label"].isin(["smishing", "phishing"]).astype(int)
    print(f"      Total: {len(df):,} | Malicious: {df['y'].sum():,}")

    X = df["clean_text"].values
    y = df["y"].values

    # Step 2: Stratified split (same as baseline)
    print("\n[2/6] Splitting data (stratified 80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Step 3: Compute class weights
    print("\n[3/6] Computing class weights...")
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    print(f"      Weights: benign={class_weights[0]:.4f}, malicious={class_weights[1]:.4f}")

    # Step 4: Tokenizer and datasets
    print("\n[4/6] Preparing tokenizer and datasets...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    train_dataset = MaliciousDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    test_dataset = MaliciousDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    print(f"      Tokenizer: {MODEL_NAME} | Max length: {MAX_LENGTH}")

    # Step 5: Model and training
    print("\n[5/6] Loading model and training...")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=500,
        save_total_limit=2,
        seed=RANDOM_STATE,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    print("      Training complete.")

    # Step 6: Final evaluation
    print("\n[6/6] Final Evaluation")
    print("=" * 70)

    # Get predictions
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=1)

    # Metrics
    precision = precision_score(y_test, preds, pos_label=1)
    recall = recall_score(y_test, preds, pos_label=1)
    f1 = f1_score(y_test, preds, pos_label=1)
    roc_auc = roc_auc_score(y_test, probs)

    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Benign  Malicious")
    print(f"  Actual Benign   {tn:>7}    {fp:>7}")
    print(f"  Actual Malicious{fn:>7}    {tp:>7}")

    print("\nMalicious Class Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    print(f"\nFalse Negatives (missed attacks): {fn}")
    print(f"False Positives (benign flagged): {fp}")

    print("\nFull Classification Report:")
    print(classification_report(y_test, preds, target_names=["Benign", "Malicious"]))

    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Model saved.")

    # Save predictions for analysis
    results_df = pd.DataFrame({
        "true_label": y_test,
        "pred_label": preds,
        "prob_malicious": probs
    })
    results_df.to_csv(f"{OUTPUT_DIR}/test_predictions.csv", index=False)
    print(f"Predictions saved to {OUTPUT_DIR}/test_predictions.csv")

    print("\n" + "=" * 70)
    print("ROBERTA TRAINING COMPLETE")
    print("=" * 70)

    # Comparison summary
    print("\n--- BASELINE COMPARISON ---")
    print("TF-IDF + LR:  Precision=0.68, Recall=0.97, F1=0.80, ROC-AUC=0.999")
    print(f"RoBERTa:      Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, ROC-AUC={roc_auc:.3f}")


if __name__ == "__main__":
    main()

