import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional, Union

import torch
from torch import nn
from tqdm import tqdm

from src.training.tracker import ExperimentTracker


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        device: torch.device,
        compute_metrics: Callable[[torch.Tensor, torch.Tensor], Dict[str, float]],
        primary_metric: str,
        greater_is_better: bool,
        tracker: ExperimentTracker,
        checkpoint_dir: Union[Path, str] = Path("checkpoints"),
        gcs_output_dir: Optional[str] = None,
    ):
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.compute_metrics = compute_metrics
        self.primary_metric = primary_metric
        self.greater_is_better = greater_is_better
        self.best_performance = -float("inf") if greater_is_better else float("inf")
        self.tracker = tracker
        self.checkpoint_dir = Path(checkpoint_dir)
        self.gcs_output_dir = gcs_output_dir
        self.current_epoch: Optional[int] = None

    def _is_better(self, score: float) -> bool:
        """Determines if the given score improves upon the current best performance."""
        if self.greater_is_better:
            return score > self.best_performance
        return score < self.best_performance

    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            logits = self.model(input_ids, attention_mask)

            total_loss += self.criterion(logits, labels).item()

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        metrics = {"val_loss": total_loss / len(self.val_loader)}

        cat_logits = torch.cat(all_logits, dim=0)
        cat_targets = torch.cat(all_labels, dim=0)
        metric_results = self.compute_metrics(cat_targets, cat_logits)
        metrics.update({f"val_{k}": v for k, v in metric_results.items()})

        self.tracker.log_metrics(metrics, self.current_epoch)

        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.optimizer.zero_grad()
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return loss.item()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            total_loss += self.train_step(batch)
        return total_loss / len(self.train_loader)

    def fit(self, epochs: int):
        for epoch in range(epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            metrics = self.validate_epoch()

            self.save_checkpoint(metrics)

        # Upload best model checkpoint as artifact to tracker (e.g. WandB)
        exp_name = (
            self.tracker.get_experiment_name()
            if hasattr(self.tracker, "get_experiment_name")
            else "default"
        )
        best_model_path = self.checkpoint_dir / exp_name / "best_model.pt"
        if hasattr(self.tracker, "log_artifact") and best_model_path.exists():
            print(f"Uploading best model checkpoint as artifact: {best_model_path}")
            self.tracker.log_artifact(
                artifact_path=best_model_path,
                name=f"{exp_name}-best-model",
                artifact_type="model",
                metadata={
                    "best_performance": float(self.best_performance),
                    "primary_metric": self.primary_metric,
                    "epochs": epochs,
                },
            )

        # Upload best model to Google Cloud Storage bucket if configured
        gcs_dest = (
            self.gcs_output_dir
            or os.environ.get("GCS_OUTPUT_DIR")
            or os.environ.get("AIP_MODEL_DIR")
        )
        if gcs_dest and best_model_path.exists():
            gcs_target = gcs_dest.rstrip("/") + "/best_model.pt"
            self._upload_to_gcs(best_model_path, gcs_target)

        if hasattr(self.tracker, "finish"):
            self.tracker.finish()

    def _upload_to_gcs(self, local_path: Path, gcs_destination: str) -> None:
        """Uploads a local file to a Google Cloud Storage URI (gs://bucket/path)."""
        try:
            from google.cloud import storage

            if not gcs_destination.startswith("gs://"):
                return
            parts = gcs_destination[5:].split("/", 1)
            bucket_name = parts[0]
            blob_name = parts[1] if len(parts) > 1 else local_path.name
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"Uploading {local_path.name} ({size_mb:.1f} MB) to {gcs_destination}...")
            blob.upload_from_filename(str(local_path))
            print(f"Successfully uploaded {local_path.name} to {gcs_destination}!")
        except Exception as e:
            print(f"Notice: GCS upload skipped or failed ({gcs_destination}): {e}")

    def save_checkpoint(
        self,
        metric: Union[float, Dict[str, float]],
        best_filename: str = "best_model.pt",
        latest_filename: str = "latest_model.pt",
        save_latest: bool = True,
    ) -> bool:
        """
        Saves the latest checkpoint and updates the best checkpoint if the primary metric improves.

        Args:
            metric: The primary metric scalar value or the full validation metrics dictionary.
            best_filename: Filename for the best performing model checkpoint (default: 'best_model.pt').
            latest_filename: Filename for the latest model checkpoint (default: 'latest_model.pt').
            save_latest: If True, always saves the latest model checkpoint for resuming training.

        Returns:
            bool: True if this checkpoint is the new best, False otherwise.
        """
        if isinstance(metric, dict):
            if self.primary_metric in metric:
                score = metric[self.primary_metric]
            elif f"val_{self.primary_metric}" in metric:
                score = metric[f"val_{self.primary_metric}"]
            else:
                raise KeyError(
                    f"Primary metric '{self.primary_metric}' not found in metrics dictionary keys: {list(metric.keys())}"
                )
        else:
            score = float(metric)

        is_best = self._is_better(score)
        if is_best:
            self.best_performance = score

        exp_name = (
            self.tracker.get_experiment_name()
            if hasattr(self.tracker, "get_experiment_name")
            else "default"
        )
        save_dir = self.checkpoint_dir / exp_name
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_performance": self.best_performance,
            self.primary_metric: score,
        }
        if isinstance(metric, dict):
            state["metrics"] = metric

        # Always save latest checkpoint with full state (model + optimizer + scheduler) for resuming
        if save_latest:
            latest_path = save_dir / latest_filename
            torch.save(state, latest_path)

        # Save best checkpoint if performance improved (lightweight: model_state_dict + metrics only)
        if is_best:
            best_path = save_dir / best_filename
            best_state = {
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "best_performance": self.best_performance,
                self.primary_metric: score,
            }
            if isinstance(metric, dict):
                best_state["metrics"] = metric
            torch.save(best_state, best_path)

        return is_best

    def load_checkpoint(self, checkpoint_path: Union[Path, str] = "best_model.pt") -> dict:
        """
        Loads model, optimizer, and scheduler states from a checkpoint file.
        Can be a direct file path or a filename within the experiment checkpoint directory.

        Args:
            checkpoint_path: Path or filename of the checkpoint file to load (default: 'best_model.pt').

        Returns:
            dict: The loaded checkpoint dictionary.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            exp_name = (
                self.tracker.get_experiment_name()
                if hasattr(self.tracker, "get_experiment_name")
                else "default"
            )
            candidate = self.checkpoint_dir / exp_name / checkpoint_path
            if candidate.exists():
                path = candidate
            else:
                raise FileNotFoundError(f"Checkpoint not found at '{path}' or '{candidate}'")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "best_performance" in checkpoint:
            self.best_performance = checkpoint["best_performance"]
        if "epoch" in checkpoint:
            self.current_epoch = checkpoint["epoch"]
        return checkpoint
        