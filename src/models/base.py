from src.schemas.prediction import ReviewClassification
from typing import List
from tqdm import tqdm


class ReviewClassificationModel:
    def classify(self, text : str) -> ReviewClassification:
        raise NotImplementedError
    def batch_classify(self, texts : List[str]) -> List[ReviewClassification]:
        result = []
        for t in tqdm(texts, desc="Labeling"):
            result.append(self.classify(t))
        return result