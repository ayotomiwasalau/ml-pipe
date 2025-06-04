#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="test_llm")

    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path, low_memory=False)

    texts = df["text_feature"].tolist()
    labels = df["genre"].tolist()

    logger.info("Downloading model artifact")
    model_path = run.use_artifact(args.model_export).download()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encodings).logits
    preds = logits.argmax(dim=1).cpu().numpy()

    # Build label mapping from training
    unique_labels = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    true_ids = [label2id[l] for l in labels]

    acc = accuracy_score(true_ids, preds)
    run.summary["accuracy"] = acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a HuggingFace model")
    parser.add_argument("--model_export", type=str, required=True, help="Model artifact")
    parser.add_argument("--test_data", type=str, required=True, help="Test data artifact")
    args = parser.parse_args()

    go(args)
