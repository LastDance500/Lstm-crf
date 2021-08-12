import torch
from torchcrf import CRF
import torch.nn as nn


class Bi_LSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, model_dim, num_tag):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=model_dim // 2,
                            bidirectional=True,
                            batch_first=True)
        self.fc = nn.Linear(model_dim, num_tag)
        self.crf = CRF(num_tag, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        x = self.lstm(x)[0]
        x = self.fc(x)
        return x

    def score(self, x, y):
        logits = self.forward(x)
        mask = torch.ne(x, 0).type(torch.uint8)
        transition_score = -self.crf(logits, y, mask=mask, reduction='mean')
        return transition_score

    def predict(self, x):
        logits = self.forward(x)
        return self.crf.decode(logits)