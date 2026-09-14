from transformers import AutoModel
from torch import nn
try:
    import config
    DEFAULT_NUM_LABELS = len(config.CATEGORIES)
except (ImportError, AttributeError):
    DEFAULT_NUM_LABELS = 3

class ReviewClassifier(nn.Module):
    def __init__(self, pretrained_model : str, num_labels : int = DEFAULT_NUM_LABELS, dropout_prob = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(pretrained_model)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids = input_ids, attention_mask = attention_mask)

        pooler = getattr(outputs, "pooler_output", None)
        cls_output = pooler if pooler is not None else outputs.last_hidden_state[:, 0, :]

        retained = self.dropout(cls_output)
        logits = self.classifier(retained)
        return logits

if __name__ == "__main__":
    model = ReviewClassifier('models/indobert-base-p1')
    from src.data.dataset import ReviewDataset
    dataset = ReviewDataset('data/processed/test.parquet', pretrained_model='models/indobert-base-p1', max_len = 128)
    data = dataset[0]
    print(model(data['input_ids'].unsqueeze(0), data['attention_mask'].unsqueeze(0)))



