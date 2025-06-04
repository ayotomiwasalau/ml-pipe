#!/usr/bin/env python
import argparse
import logging
import os
import pandas as pd
import wandb
import mlflow
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def tokenize_function(tokenizer):
    def _tokenize(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")
    return _tokenize


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


def go(args):
    run = wandb.init(job_type="train_llm")

    logger.info("Downloading and reading train artifact")
    train_data_path = run.use_artifact(args.train_data).file()
    df = pd.read_csv(train_data_path, low_memory=False)

    X = df["text_feature"]
    y = df["genre"]

    logger.info("Splitting train/val")
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_size,
        stratify=df[args.stratify] if args.stratify != "null" else None,
        random_state=args.random_seed,
    )

    labels = sorted(y.unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_ds = Dataset.from_dict({"text": list(X_train), "label": [label2id[l] for l in y_train]})
    val_ds = Dataset.from_dict({"text": list(X_val), "label": [label2id[l] for l in y_val]})

    train_ds = train_ds.map(tokenize_function(tokenizer), batched=True)
    val_ds = val_ds.map(tokenize_function(tokenizer), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="model",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        logging_dir="logs",
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Training model")
    trainer.train()

    metrics = trainer.evaluate()
    run.summary["accuracy"] = metrics.get("eval_accuracy")

    trainer.save_model("model")

    if args.export_artifact != "null":
        artifact = wandb.Artifact(
            args.export_artifact,
            type="model_export",
            description="HuggingFace model",
        )
        artifact.add_dir("model")
        run.log_artifact(artifact)
        artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a HuggingFace transformer")
    parser.add_argument("--train_data", type=str, required=True, help="Training data artifact")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Pretrained model name")
    parser.add_argument("--export_artifact", type=str, default="null", help="Exported model artifact name")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--stratify", type=str, default="genre")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    go(args)
