from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk


def get_tokenizer(name: str = "gpt2") -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_tinystories_dataset(
    tokenizer: AutoTokenizer,
    max_length: int = 512,
    num_workers: int = 4,
    cache_dir: str = None,
):
    """
    Load and tokenize TinyStories.

    If cache_dir is provided and contains a pre-tokenized dataset, load from disk.
    Otherwise download and tokenize, saving to cache_dir if provided.

    Returns:
        train_dataset, val_dataset
    """
    if cache_dir is not None:
        import os
        train_path = os.path.join(cache_dir, "train")
        val_path = os.path.join(cache_dir, "validation")
        if os.path.exists(train_path) and os.path.exists(val_path):
            print(f"Loading tokenized dataset from {cache_dir}")
            train_ds = load_from_disk(train_path)
            val_ds = load_from_disk(val_path)
            train_ds.set_format(type="torch", columns=["input_ids"])
            val_ds.set_format(type="torch", columns=["input_ids"])
            return train_ds, val_ds

    print("Downloading and tokenizing TinyStories...")
    dataset = load_dataset("skeskinen/TinyStories-hf")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,  # not needed; collator handles padding
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_workers,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch", columns=["input_ids"])

    if cache_dir is not None:
        import os
        os.makedirs(cache_dir, exist_ok=True)
        tokenized["train"].save_to_disk(os.path.join(cache_dir, "train"))
        tokenized["validation"].save_to_disk(os.path.join(cache_dir, "validation"))
        print(f"Saved tokenized dataset to {cache_dir}")

    return tokenized["train"], tokenized["validation"]


class LMDataCollator:
    """
    Collator for causal language modeling.

    - Pads sequences to the longest in the batch
    - input_ids  = tokens[:-1]  (the context)
    - targets    = tokens[1:]   (the next-token labels)
    - Padding positions in targets are set to -100 (ignored by CE loss)
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        seqs = [ex["input_ids"] for ex in examples]
        max_len = max(len(s) for s in seqs)

        input_ids = []
        targets = []

        for seq in seqs:
            seq_len = len(seq)
            pad_len = max_len - seq_len

            # Pad sequence
            padded = torch.cat([seq, torch.full((pad_len,), self.pad_token_id)])

            # Causal LM: input is all but last token, target is all but first
            input_ids.append(padded[:-1])
            target = padded[1:].clone()
            # Mask padding in targets
            if pad_len > 0:
                target[-(pad_len):] = -100
            targets.append(target)

        return {
            "input_ids": torch.stack(input_ids),  # [B, seq_len-1]
            "targets": torch.stack(targets),       # [B, seq_len-1]
        }


def make_dataloaders(
    train_dataset,
    val_dataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    collator = LMDataCollator(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
