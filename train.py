import os
import random
from pathlib import Path

import dotenv
import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

dotenv.load_dotenv()
import src
from src.training.metrics import compute_metrics
from src.training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Determine device
    device = torch.device(
        cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # Datasets and Dataloaders
    train_dataset = instantiate(cfg.dataset.train_dataset)
    valid_dataset = instantiate(cfg.dataset.valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and (device.type == "cuda"),
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and (device.type == "cuda"),
    )

    # Model
    model = instantiate(cfg.model).to(device)

    # Multi-label binary cross entropy loss
    criterion = torch.nn.BCEWithLogitsLoss()

    # Optimizer with weight decay excluding bias and LayerNorm
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=cfg.learning_rate)

    # Learning rate scheduler with linear warmup
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Tracker (WandB or Console)
    tracker = instantiate(cfg.tracker, config=cfg)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        compute_metrics=compute_metrics,
        primary_metric=cfg.primary_metric,
        greater_is_better=cfg.greater_is_better,
        tracker=tracker,
        checkpoint_dir=cfg.checkpoint_dir,
    )

    print(f"Starting training for {cfg.epochs} epochs ({total_steps} steps)...")
    trainer.fit(epochs=cfg.epochs)
    print(f"Training completed! Best {cfg.primary_metric}: {trainer.best_performance:.4f}")


if __name__ == "__main__":
    main()