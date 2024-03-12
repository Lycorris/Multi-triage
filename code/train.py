from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from datasets import *
from model import *

# dataset
train_path = 'Data/powershell/C_uA_Train.csv'
test_path = 'Data/powershell/C_uA_Test.csv'
MAX_SEQ_LEN = 300
Learning_Rate = 1e-3
EPOCH = 20

train_dataset = TextCodeDataset(train_path, pad_seq_len=MAX_SEQ_LEN)
test_dataset = TextCodeDataset(test_path, pad_seq_len=MAX_SEQ_LEN)

tokenize_dataset_input(train_dataset, test_dataset)
map_dataset_output(train_dataset, test_dataset)

# dataloader
train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# model
model = Model()

# loss_func
loss_fn = BCELoss()

# optimizer
optimizer = Adam(model.parameters(), lr=Learning_Rate)

