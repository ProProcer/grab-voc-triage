from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import torch
from typing import List

class ReviewDataset(Dataset):
    def __init__(self, parquet_path : str, text_col : str, category_cols : List[str], pretrained_model : str, max_tokens_len : int):
        df = pd.read_parquet(parquet_path)
        self.texts = [str(t) if (pd.notna(t) and t is not None) else "" for t in df[text_col]]
        self.labels = df[category_cols].values.astype('float32')
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.max_tokens_len = max_tokens_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        encoding = self.tokenizer(
            text = text,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_tokens_len,
            return_tensors = 'pt'
        )
        return {
            'input_ids' : encoding['input_ids'].squeeze(0),
            'attention_mask' : encoding['attention_mask'].squeeze(0),
            'labels' : torch.tensor(self.labels[idx])
        }

if __name__ == "__main__":
    import config
    dataset = ReviewDataset('data/processed/test.parquet', 'content', config.CATEGORIES, 'models/indobert-base-p1', 128)
    print(dataset[0]['labels'])