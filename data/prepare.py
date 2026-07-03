"""
Downloads the Tiny Shakespeare dataset and splits it into train/val text files.

This script only handles raw text acquisition and splitting. Tokenization
(char-level, turning text into integer ids) happens in model/tokenizer.py,
which you write yourself in Phase 1 -- keeping this file simple and separate
means you can swap datasets later without touching any model code.

Usage:
    python data/prepare.py
"""

import os
import urllib.request

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(DATA_DIR, "shakespeare.txt")
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
VAL_PATH = os.path.join(DATA_DIR, "val.txt")

VAL_FRACTION = 0.1  # last 10% of the text held out for validation


def download_raw_text() -> str:
    if os.path.exists(RAW_PATH):
        print(f"Found existing raw file at {RAW_PATH}, skipping download.")
    else:
        print(f"Downloading Tiny Shakespeare from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, RAW_PATH)
        print(f"Saved to {RAW_PATH}")

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        return f.read()


def split_and_save(text: str) -> None:
    n = len(text)
    split_idx = int(n * (1 - VAL_FRACTION))

    train_text = text[:split_idx]
    val_text = text[split_idx:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        f.write(val_text)

    print(f"Total characters: {n:,}")
    print(f"Train: {len(train_text):,} chars -> {TRAIN_PATH}")
    print(f"Val:   {len(val_text):,} chars -> {VAL_PATH}")

    unique_chars = sorted(set(text))
    print(f"Vocabulary size (unique characters): {len(unique_chars)}")
    print(f"First 50 chars of train: {train_text[:50]!r}")


if __name__ == "__main__":
    text = download_raw_text()
    split_and_save(text)
