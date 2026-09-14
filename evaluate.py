import os
from pathlib import Path
from typing import List

import dotenv
import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

dotenv.load_dotenv()
import config
import src
from src.training.metrics import compute_metrics, format_classification_report


@hydra.main(version_base="1.3", config_path="configs", config_name="evaluate")
def main(cfg: DictConfig):
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )
    print(f"Evaluation device: {device}")

    # Select dataset split
    split = getattr(cfg, "split", "test")
    if split == "test":
        dataset_cfg = cfg.dataset.test_dataset
    elif split == "valid":
        dataset_cfg = cfg.dataset.valid_dataset
    elif split == "train":
        dataset_cfg = cfg.dataset.train_dataset
    else:
        raise ValueError(f"Unknown split '{split}'. Expected 'test', 'valid', or 'train'.")

    print(f"Loading {split} dataset...")
    dataset = instantiate(dataset_cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Initialize model
    print("Instantiating model...")
    model = instantiate(cfg.model).to(device)

    # Load checkpoint
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        fallback = Path("checkpoints") / cfg.run_name / "best_model.pt"
        if fallback.exists():
            checkpoint_path = fallback
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at '{checkpoint_path}' or '{fallback}'"
            )

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        saved_epoch = checkpoint.get("epoch", "unknown")
        print(f"Loaded checkpoint from epoch {saved_epoch}")
    else:
        model.load_state_dict(checkpoint)

    # Inference loop
    model.eval()
    all_logits = []
    all_labels = []

    print(f"Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    cat_logits = torch.cat(all_logits, dim=0)
    cat_targets = torch.cat(all_labels, dim=0)
    cat_probs = torch.sigmoid(cat_logits)
    threshold = getattr(cfg, "threshold", 0.5)
    cat_preds = (cat_probs >= threshold).long()

    # Compute and print metrics (matching notebook 03 benchmark)
    categories = list(cfg.dataset.category_cols)
    metrics = compute_metrics(
        targets=cat_targets,
        logits_or_preds=cat_logits,
        threshold=threshold,
        categories=categories,
    )

    print("\n" + "=" * 40)
    print(f"EVALUATION REPORT ({split.upper()} SET)")
    print("=" * 40)
    print(format_classification_report(cat_targets, cat_logits, categories=categories, threshold=threshold))
    print("=" * 40 + "\n")

    # Save predictions if path configured
    output_path = getattr(cfg, "output_predictions_path", None)
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        y_true = cat_targets.numpy().astype(int)
        y_pred = cat_preds.numpy().astype(int)
        y_prob = cat_probs.numpy()

        label_map = {0: "ABSENT", 1: "NEG"}
        df_out = pd.DataFrame({"content": dataset.texts})

        for i, cat in enumerate(categories):
            df_out[cat] = [label_map.get(v, v) for v in y_true[:, i]]
            df_out[f"{cat}_pred"] = [label_map.get(v, v) for v in y_pred[:, i]]
            df_out[f"{cat}_prob"] = np.round(y_prob[:, i], 4)

        df_out.to_csv(out_path, index=False)
        print(f"Predictions saved to: {out_path}")

    return metrics


if __name__ == "__main__":
    main()
