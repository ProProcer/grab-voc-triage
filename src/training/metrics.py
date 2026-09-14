from typing import Dict, List, Optional, Union
import numpy as np
import torch
from sklearn import metrics as sk_metrics
import config


def compute_metrics(
    targets: Union[torch.Tensor, np.ndarray],
    logits_or_preds: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5,
    categories: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes evaluation metrics matching notebooks/03_benchmark_openai_classification.ipynb:
    
    1. Primary Metrics:
       - {category}_macro_f1: Macro F1 per category
       - macro_f1_mean: Average macro F1 across all categories (used for model checkpointing)
    2. Secondary Metric:
       - exact_match_ratio: Percentage of samples where all category predictions exactly match ground truth
    3. Diagnostic Metrics:
       - {category}_absent_recall: Recall for class 0 (ABSENT)
       - {category}_neg_recall: Recall for class 1 (NEG)

    Args:
        targets: Ground truth binary tensor or array of shape (N, num_classes).
        logits_or_preds: Model raw logits or binary predictions of shape (N, num_classes).
        threshold: Classification probability threshold (default: 0.5).
        categories: List of category names. Defaults to config.CATEGORIES.

    Returns:
        Dict[str, float]: Dictionary of metric names and their scalar values.
    """
    categories = categories or config.CATEGORIES

    if isinstance(targets, torch.Tensor):
        y_true = targets.detach().cpu().numpy().astype(int)
    else:
        y_true = np.asarray(targets, dtype=int)

    if isinstance(logits_or_preds, torch.Tensor):
        logits = logits_or_preds.detach().cpu()
        if logits.is_floating_point():
            probs = torch.sigmoid(logits)
            y_pred = (probs >= threshold).long().numpy().astype(int)
        else:
            y_pred = logits.numpy().astype(int)
    else:
        logits_arr = np.asarray(logits_or_preds)
        if np.issubdtype(logits_arr.dtype, np.floating):
            # Check if values are raw logits or probabilities
            if (logits_arr < 0.0).any() or (logits_arr > 1.0).any():
                probs = 1.0 / (1.0 + np.exp(-logits_arr))
            else:
                probs = logits_arr
            y_pred = (probs >= threshold).astype(int)
        else:
            y_pred = logits_arr.astype(int)

    results: Dict[str, float] = {}
    macro_f1_list: List[float] = []

    for i, cat in enumerate(categories):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # Primary Metric: Macro F1
        cat_macro_f1 = float(sk_metrics.f1_score(y_t, y_p, average="macro", zero_division=0))
        results[f"{cat}_macro_f1"] = cat_macro_f1
        macro_f1_list.append(cat_macro_f1)

        # Diagnostic Metrics: ABSENT (0) and NEG (1) Recall
        absent_rec = float(sk_metrics.recall_score(y_t, y_p, pos_label=0, zero_division=0))
        neg_rec = float(sk_metrics.recall_score(y_t, y_p, pos_label=1, zero_division=0))
        results[f"{cat}_absent_recall"] = absent_rec
        results[f"{cat}_neg_recall"] = neg_rec

    # Mean Macro F1 across all categories (Primary checkpoint selection metric)
    results["macro_f1_mean"] = float(np.mean(macro_f1_list))

    # Secondary Metric: Exact Match Ratio
    results["exact_match_ratio"] = float((y_true == y_pred).all(axis=1).mean())

    return results


def format_classification_report(
    y_true: Union[torch.Tensor, np.ndarray],
    y_pred: Union[torch.Tensor, np.ndarray],
    categories: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> str:
    """
    Formats evaluation results as a readable text report matching the benchmark notebook.
    """
    metrics = compute_metrics(y_true, y_pred, threshold=threshold, categories=categories)
    categories = categories or config.CATEGORIES

    lines = [
        "--------------------------",
        "PRIMARY METRIC",
        "--------------------------",
    ]
    for cat in categories:
        lines.append(f"{cat} Macro F1-score: {metrics[f'{cat}_macro_f1']:.4f}")
    lines.append(f"Mean Macro F1-score: {metrics['macro_f1_mean']:.4f}")

    lines.extend([
        "--------------------------",
        "SECONDARY METRIC (Exact Match)",
        "--------------------------",
        f"Exact Match Ratio: {metrics['exact_match_ratio']:.4f}",
        "--------------------------",
        "DIAGNOSTIC METRIC (Recall)",
        "--------------------------",
    ])
    for cat in categories:
        lines.append(f"{cat} ABSENT Recall: {metrics[f'{cat}_absent_recall']:.4f}")
        lines.append(f"{cat} NEG Recall: {metrics[f'{cat}_neg_recall']:.4f}")

    return "\n".join(lines)
