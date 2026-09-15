from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer

from src.models.hf_pretrained import ReviewClassifier
from src.schemas.prediction import CategoryScore, SentimentLabel, TriageResponse

DEFAULT_CATEGORIES = ["DRIVER_OPS", "APP_AND_MAPS", "PRICING_AND_BILLING"]


class TriagePipeline:
    """
    High-performance inference pipeline for Grab Voice-of-Customer triage classification.
    Loads fine-tuned IndoBERT checkpoint once and handles single or batched predictions.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = "checkpoints/indobert-base-run1/best_model.pt",
        pretrained_model: str = "pretrained_weights/indobert-base-p1",
        categories: Optional[List[str]] = None,
        max_tokens_len: int = 128,
        device: Optional[str] = None,
    ):
        self.categories = categories or DEFAULT_CATEGORIES
        self.max_tokens_len = max_tokens_len

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load Tokenizer
        tokenizer_path = Path(pretrained_model)
        if not tokenizer_path.exists():
            # Fallback to HuggingFace Hub if local weights dir doesn't exist
            pretrained_model = "indobenchmark/indobert-base-p1"

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # Initialize model architecture
        self.model = ReviewClassifier(
            pretrained_model=pretrained_model,
            num_labels=len(self.categories),
        ).to(self.device)

        # Load trained weights
        ckpt_path = Path(checkpoint_path)
        if str(checkpoint_path).startswith("gs://"):
            try:
                from google.cloud import storage
                parts = str(checkpoint_path).replace("gs://", "").split("/", 1)
                bucket_name, blob_path = parts[0], parts[1]
                local_cache = Path("checkpoints") / Path(blob_path).name
                local_cache.parent.mkdir(parents=True, exist_ok=True)
                if not local_cache.exists():
                    print(f"Downloading checkpoint from {checkpoint_path}...")
                    storage_client = storage.Client()
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_path)
                    blob.download_to_filename(str(local_cache))
                ckpt_path = local_cache
            except Exception as e:
                print(f"Failed to fetch from GCS {checkpoint_path}: {e}")

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found at: {ckpt_path}. "
                "Ensure training has completed and the checkpoint exists locally or in checkpoints/."
            )

        checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    @torch.inference_mode()
    def predict_one(self, text: str, threshold: float = 0.5) -> TriageResponse:
        """Runs triage prediction for a single review string."""
        return self.predict_batch([text], threshold=threshold)[0]

    @torch.inference_mode()
    def predict_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        threshold: float = 0.5,
    ) -> List[TriageResponse]:
        """Runs efficient batched triage prediction for a list of reviews."""
        if not texts:
            return []

        results: List[TriageResponse] = []

        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            clean_chunk = [str(t) if t is not None else "" for t in chunk]

            encodings = self.tokenizer(
                clean_chunk,
                padding=True,
                truncation=True,
                max_length=self.max_tokens_len,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
            )
            probs = torch.sigmoid(logits).cpu().numpy()

            for text, row_probs in zip(chunk, probs):
                cat_dict: Dict[str, CategoryScore] = {}
                active_complaints: List[str] = []

                for cat_idx, cat_name in enumerate(self.categories):
                    prob = float(row_probs[cat_idx])
                    label = SentimentLabel.NEG if prob >= threshold else SentimentLabel.ABSENT
                    cat_dict[cat_name] = CategoryScore(label=label, probability=round(prob, 4))
                    if label == SentimentLabel.NEG:
                        active_complaints.append(cat_name)

                results.append(
                    TriageResponse(
                        content=str(text),
                        categories=cat_dict,
                        active_complaints=active_complaints,
                    )
                )

        return results
