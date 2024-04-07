import torch
from torch import nn
from transformers import AlbertModel, BertForSequenceClassification

"""
    original model(
        seq_len,
        from_emb: bool, vocab_size: list, emb_dim, -> Embedding
        filter: list -> Feature Extracting separately
        linear_concat -> Joint Linear
        n_classes -> Respective CLS
    ):
    1. Embedding
        a. Embedding
    2. Feature Extracting separately
        a. 1 Conv1d(Context, Btype respectively) + ReLU + MaxPool1d
        b. Flatten
    3. Joint Linear
        a. BatchNorm1d + Dropout
        b. 1 Linear + ReLU
    4. Respective CLS
        a. 1 Linear(Dev, Btype respectively) -> logits
"""


class MetaModel(nn.Module):
    def __init__(self, seq_len, from_emb: bool, from_token: bool, pretrained_model, vocab_size: list, emb_dim,
                 filter: list, linear_concat,
                 n_classes: list):
        """
        Args:
            vocab_size: list, the size of C/A vocab
            emb_dim: : int, the dim of C&A emb layer
            seq_len:  MAX_SEQ_LEN
            n_classes: list, the output size of D/B
        """
        super(MetaModel, self).__init__()
        # 1. Embedding
        self.from_emb = from_emb
        self.from_token = from_token
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.emb_C = nn.Embedding(self.vocab_size[0], self.emb_dim)
        self.emb_A = nn.Embedding(self.vocab_size[1], self.emb_dim)
        self.pretrained_model = pretrained_model

        # 2. Feature Extracting separately
        self.filter_C, self.filter_A = filter
        self.seq_len = seq_len
        self.feature_C = nn.Sequential(
            # (Batch_sz, emb_dim, seq_len)
            nn.Conv1d(self.emb_dim, self.filter_C, kernel_size=3, padding='same'),
            nn.ReLU(),
            # (Batch_sz, filter_c, seq_len)
            nn.MaxPool1d(self.seq_len, 1),
            # (Batch_sz, filter_c, 1)
            nn.Flatten(),
            # (Batch_sz, filter_c)
        )
        self.feature_A = nn.Sequential(
            nn.Conv1d(self.emb_dim, self.filter_A, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(self.seq_len, 1),
            nn.Flatten(),
        )

        # 3. Joint Linear
        self.linear_concat = linear_concat
        self.fc = nn.Sequential(
            # (Batch_sz, filter_C + filter_A)
            nn.BatchNorm1d(self.filter_C + self.filter_A, affine=False),
            nn.Dropout(0.5),
            nn.Linear(self.filter_C + self.filter_A, self.linear_concat),
            # (Batch_sz, linear_concat)
            nn.ReLU()
        )

        # 4. Respective CLS
        self.n_classes_D, self.n_classes_B = n_classes
        self.fc_D = nn.Linear(self.linear_concat, self.n_classes_D)
        self.fc_B = nn.Linear(self.linear_concat, self.n_classes_B)

    def forward(self, x_C, x_A):
        # 1. Embedding
        # (Batch_sz, seq_len)
        if self.from_emb:
            pass
        elif self.from_token:
            x_C = self.pretrained_model(x_C[0], x_C[1])[0]
            x_C = nn.Dropout()(x_C)
            x_A = self.pretrained_model(x_A[0], x_A[1])[0]
            x_C = nn.Dropout()(x_A)
        else:
            x_C = self.emb_C(x_C)
            x_A = self.emb_A(x_A)
        # (Batch_sz, seq_len, emb_dim)
        # 2. Feature Extracting separately
        x_C = x_C.permute(0, 2, 1)
        x_A = x_A.permute(0, 2, 1)
        # (Batch_sz, emb_dim, seq_len)
        x_C = self.feature_C(x_C)
        x_A = self.feature_A(x_A)
        # (Batch_sz, filter)
        # 3. Joint Linear
        x = torch.concat((x_C, x_A), 1)
        # (Batch_sz, filter_C + filter_A)
        x = self.fc(x)
        # (Batch_sz, linear_concat)
        # 4. Respective CLS
        y_D = self.fc_D(x)
        # (Batch_sz, n_classes_D)
        y_B = self.fc_B(x)
        # (Batch_sz, n_classes_B)
        return y_D, y_B


if __name__ == '__main__':
    # Sanity Test
    Batch_sz = 64
    MAX_SEQ_LEN = 300
    from_emb = False
    from_token = True
    pretrained_model = AlbertModel.from_pretrained('albert-base-v2').to('cpu')
    vocab_size = [12513, 5910]
    EMB_DIM = 100 if not from_emb and not from_token else 768
    filter = [64, 64]
    linear_concat = 50
    n_classes = [396, 204]

    model = MetaModel(MAX_SEQ_LEN, from_emb, from_token, pretrained_model, vocab_size, EMB_DIM, filter, linear_concat,
                      n_classes)
    model.eval()
    with torch.no_grad():
        input_C, input_A = torch.ones((Batch_sz, MAX_SEQ_LEN)).long(), torch.ones((Batch_sz, MAX_SEQ_LEN)).long()
        output_d, output_b = model(input_C, input_A)
    print(output_d.shape)
    print(output_d)
    print(output_b.shape)
    print(output_b)
