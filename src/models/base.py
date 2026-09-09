from typing import List
from tqdm import tqdm


class ReviewClassificationModel:
    def classify(self, text : str) -> dict:
        raise NotImplementedError
    def batch_classify(self, texts : List[str]) -> List[dict]:
        result = []
        for t in tqdm(texts, desc="Labeling"):
            result.append(self.classify(t))
        return result