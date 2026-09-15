from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

import wandb
from omegaconf import DictConfig, OmegaConf


@runtime_checkable
class ExperimentTracker(Protocol):
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None: ...
    def get_experiment_name(self) -> str: ...
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...


class WandbTracker:
    """
    Weights & Biases experiment tracker implementing the ExperimentTracker protocol.
    """

    def __init__(
        self,
        project: str = "grab-voc-triage",
        name: Optional[str] = None,
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        run_config: Optional[Union[Dict[str, Any], DictConfig]] = None,
        mode: Optional[str] = None,
        dir: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        config = run_config if run_config is not None else kwargs.pop("config", None)
        # Convert OmegaConf DictConfig to native Python dict if passed from Hydra
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        self.run = wandb.init(
            project=project,
            name=name,
            entity=entity,
            tags=tags,
            notes=notes,
            config=config,
            mode=mode,
            dir=str(dir) if dir else None,
            reinit="finish_previous",
            **kwargs,
        )

        # Ensure a valid non-empty string for the experiment name (used in checkpoint paths)
        self.experiment_name = (
            name
            or (self.run.name if self.run and self.run.name else None)
            or (self.run.id if self.run and hasattr(self.run, "id") else None)
            or "default_run"
        )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Logs a dictionary of metrics to Weights & Biases."""
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)

    def get_experiment_name(self) -> str:
        """Returns the experiment / run name used for directory paths and tracking."""
        return self.experiment_name

    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Logs a file or directory as a W&B artifact (e.g. best model checkpoint)."""
        path = Path(artifact_path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found at {path}")

        artifact_name = name or f"{self.experiment_name}_{path.stem}"
        artifact = wandb.Artifact(name=artifact_name, type=artifact_type, metadata=metadata)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """Ends the W&B run cleanly."""
        if wandb.run is not None:
            wandb.finish()

    def __enter__(self) -> "WandbTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.finish()


class ConsoleTracker:
    """
    Lightweight console-based tracker implementing ExperimentTracker for local or testing runs.
    """

    def __init__(self, experiment_name: str = "default_run", **kwargs: Any):
        self.experiment_name = experiment_name

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        step_str = f" [Step {step}]" if step is not None else ""
        formatted = ", ".join(
            f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
            for k, v in metrics.items()
        )
        print(f"[{self.experiment_name}]{step_str} {formatted}")

    def get_experiment_name(self) -> str:
        return self.experiment_name

    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: Optional[str] = None,
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        print(f"[{self.experiment_name}] Saved local artifact: {artifact_path}")

    def finish(self) -> None:
        pass

    def __enter__(self) -> "ConsoleTracker":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass
