import torch
from torch import nn


class MetaModel(nn.Module):
    def __init__(self, vocab_size: list, emb_dim: int, seq_len: int, num_out: list):
        """
        Args:
            vocab_size: list, the size of C/A vocab
            emb_dim: : int, the dim of C&A emb layer
            seq_len:  MAX_SEQ_LEN
            num_out: list, the output size of D/B
        """
        super(MetaModel, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.seq_len = seq_len
        self.num_out = num_out
        self.context = nn.Sequential(
            nn.Embedding(self.vocab_size[0], self.emb_dim),
            nn.Conv1d(self.emb_dim, 64, kernel_size=2),  # Change self.seq_len to self.emb_dim
            nn.ReLU(),
            nn.MaxPool1d(99, 1),
            nn.Flatten(),
        )
        self.AST = nn.Sequential(
            nn.Embedding(self.vocab_size[1], self.emb_dim),
            nn.Conv1d(self.emb_dim, 50, kernel_size=2),  # Change self.seq_len to self.emb_dim
            nn.ReLU(),
            nn.MaxPool1d(99, 1),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(114, affine=False),
            nn.Dropout(0.5),
            nn.Linear(114, 50),
            nn.ReLU()
        )
        self.fc_d = nn.Sequential(
            nn.Linear(50, self.num_out[0]),
            nn.Sigmoid(),
        )
        self.fc_b = nn.Sequential(
            nn.Linear(50, self.num_out[1]),
            nn.Sigmoid(),
        )

    def forward(self, x_C, x_A):
        # x, y
        x_C = self.context(x_C)
        x_A = self.AST(x_A)
        x = torch.concat((x_C, x_A), 1)
        x = self.fc(x)
        y_d = self.fc_d(x)
        y_b = self.fc_b(x)
        return y_d, y_b


if __name__ == '__main__':
    vocab_size = [12513, 5910]
    num_out = [396, 204]
    EMB_DIM = 100
    MAX_SEQ_LEN = 300

    model = MetaModel(vocab_size, EMB_DIM, MAX_SEQ_LEN, num_out)
    model.eval()
    with torch.no_grad():
        input_C, input_A = torch.ones((1, 300)).long(), torch.ones((1, 300)).long()
        output_d, output_b = model(input_C, input_A)
    print(output_d.shape)
    print(output_d)
    print(output_b.shape)
    print(output_b)
