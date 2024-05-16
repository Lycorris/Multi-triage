import torch
from torch import nn
from transformers import AlbertModel, BertForSequenceClassification
from transformers import AutoModel, AutoModelForSequenceClassification

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


class PretrainModel(nn.Module):
    def __init__(self, text_ckpt, code_ckpt, n_classes, n_linear_concat=1000, use_AST=True):
        super().__init__()
        self.text_ckpt = text_ckpt
        self.code_ckpt = code_ckpt
        self.n = n_classes[0] + n_classes[1]
        self.n_D = n_classes[0]
        self.n_B = n_classes[1]
        self.n_linear_concat = n_linear_concat
        self.use_AST = use_AST

        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            text_ckpt, num_labels=self.n, problem_type="multi_label_classification",local_files_only=True)
        self.code_model = AutoModelForSequenceClassification.from_pretrained(
            code_ckpt, num_labels=self.n, problem_type="multi_label_classification",local_files_only=True) if use_AST else None

        self.fc = nn.Sequential(
            nn.BatchNorm1d(2 * self.n, affine=False),
            nn.Dropout(0.5),
            nn.Linear(2 * self.n, self.n_linear_concat),
            nn.ReLU()
        )

        self.fc_D = nn.Linear(self.n_linear_concat, self.n_D)
        self.fc_B = nn.Linear(self.n_linear_concat, self.n_B)

    def forward(self, x_C, x_A):
        x_C = self.text_model(x_C['input_ids'].squeeze(dim=1), x_C['attention_mask'].squeeze(dim=1))[0]
        if not self.use_AST:
            return x_C
        x_A = self.code_model(x_A['input_ids'].squeeze(dim=1), x_A['attention_mask'].squeeze(dim=1))[0]

        x = torch.concat((x_C, x_A), dim=1)
        x = self.fc(x)

        y_D = self.fc_D(x)
        y_B = self.fc_B(x)
        y = torch.concat((y_D, y_B), dim=1)

        return y


# TODO: Sanity Check
class PreTrainModel2Stage(nn.Module):
    def __init__(self, text_ckpt, code_ckpt, n_classes, n_emb_dim=768, use_AST=True):
        super().__init__()
        self.text_ckpt = text_ckpt
        self.code_ckpt = code_ckpt
        self.n = n_classes[0] + n_classes[1]
        self.n_emb_dim = n_emb_dim
        self.n_D = n_classes[0]
        self.n_B = n_classes[1]
        self.use_AST = use_AST

        self.text_model = AutoModel.from_pretrained(
            text_ckpt, hidden_size=n_emb_dim,local_files_only=True)
        self.code_model = AutoModel.from_pretrained(
            code_ckpt, hidden_size=n_emb_dim,local_files_only=True) if use_AST else None

        self.fc = nn.Sequential(
            nn.BatchNorm1d(2 * self.n, affine=False),
            nn.Dropout(0.5),
            nn.Linear(2 * self.n, self.n_emb_dim),
            nn.ReLU()
        )

        self.fc_D = nn.Linear(self.n_linear_concat, self.n_D)
        self.fc_B = nn.Linear(self.n_linear_concat, self.n_B)

    def forward(self, x_C, x_A, stage):
        x_C = self.text_model(x_C['input_ids'].squeeze(dim=1), x_C['attention_mask'].squeeze(dim=1))[0]
        if stage == 1 and not self.use_AST:
            return x_C
        x_A = self.code_model(x_A['input_ids'].squeeze(dim=1), x_A['attention_mask'].squeeze(dim=1))[0]

        x = torch.concat((x_C, x_A), dim=1)
        x = self.fc(x)
        if stage == 1:
            return x

        y_D = self.fc_D(x)
        y_B = self.fc_B(x)
        y = torch.concat((y_D, y_B), dim=1)

        return y


class MetaModel(nn.Module):
    def __init__(self, n_classes: list, seq_len=512, vocab_size: list = [30700, 30700], emb_dim=100,
                 CNN_filter: int = 64, LSTM_unit=25, linear_concat=50, use_AST=True):
        super(MetaModel, self).__init__()
        self.use_AST = use_AST
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.CNN_filter = CNN_filter
        self.seq_len = seq_len
        self.LSTM_unit = LSTM_unit if use_AST else 0
        self.linear_concat = linear_concat
        self.n_classes_D, self.n_classes_B = n_classes

        self.emb_C = nn.Embedding(self.vocab_size[0], self.emb_dim)
        self.feature_C = nn.Sequential(
            nn.Conv1d(self.emb_dim, self.CNN_filter, kernel_size=2, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(self.seq_len, 1),
            nn.Flatten(),
        )

        if self.use_AST:
            self.emb_A = nn.Embedding(self.vocab_size[1], self.emb_dim)
            self.BiLSTM_A = nn.LSTM(self.emb_dim, self.LSTM_unit, batch_first=True, bidirectional=True)
            self.feature_A = nn.Sequential(
                nn.ReLU(),
                nn.MaxPool1d(self.seq_len, 1),
                nn.Flatten(),
            )

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.CNN_filter + self.LSTM_unit * 2, affine=False),
            nn.Dropout(0.5),
            nn.Linear(self.CNN_filter + self.LSTM_unit * 2, self.linear_concat),
            nn.ReLU()
        )

        self.fc_D = nn.Linear(self.linear_concat, self.n_classes_D)
        self.fc_B = nn.Linear(self.linear_concat, self.n_classes_B)

    def forward(self, x_C, x_A):
        x_C = self.emb_C(x_C['input_ids'].squeeze(dim=1))
        x_C = x_C.permute(0, 2, 1)
        x_C = self.feature_C(x_C)

        if self.use_AST:
            x_A = self.emb_A(x_A['input_ids'].squeeze(dim=1))
            x_A, _ = self.BiLSTM_A(x_A)
            x_A = x_A.permute(0, 2, 1)
            x_A = self.feature_A(x_A)
            x = torch.concat((x_C, x_A), 1)
        else:
            x = x_C

        x = self.fc(x)

        y_D = self.fc_D(x)
        y_B = self.fc_B(x)
        y = torch.concat((y_D, y_B), dim=1)

        return y


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
