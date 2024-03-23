import time

import torch
from torch.nn import BCELoss
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import *
from model import *
from metrics import *

# dataset
train_path = 'Data/powershell/C_uA_Train.csv'
test_path = 'Data/powershell/C_uA_Test.csv'
MAX_SEQ_LEN = 300
# why 100?
EMB_DIM = 100
Learning_Rate = 1e-3
EPOCH = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = TextCodeDataset(train_path, pad_seq_len=MAX_SEQ_LEN)
test_dataset = TextCodeDataset(test_path, pad_seq_len=MAX_SEQ_LEN)

vocab_size = tokenize_dataset_input(train_dataset, test_dataset)
idx2label = map_dataset_output(train_dataset, test_dataset)
num_out = [len(x) for x in idx2label]

# dataloader
train_loader = DataLoader(train_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# model
model = MetaModel(vocab_size, EMB_DIM, MAX_SEQ_LEN, num_out)

# loss_func
loss_fn = BCELoss()

# optimizer
optimizer = SGD(model.parameters(), lr=Learning_Rate)
# optimizer adam
# optimizer = Adam(model.parameters(), lr=Learning_Rate)

def one_forward(data):
    (x_context, x_AST), (y_dev, y_btype) = data
    print(f"x_context: {x_context.shape}, x_AST: {x_AST.shape}, y_dev: {y_dev.shape}, y_btype: {y_btype.shape}")
    y_dev_pred, y_btype_pred = model(x_context, x_AST)
    y = torch.concat((y_dev, y_btype), 1)
    y_pred = torch.concat((y_dev_pred, y_btype_pred), 1)
    loss, acc = loss_fn(y, y_pred), metrics_acc(y, y_pred)
    return loss, acc


def one_backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


start_time = time.time()
for epoch in range(EPOCH):
    print('epoch{} begin:'.format(epoch))

    # train
    print('train:')
    model.train()
    for step, data in tqdm(enumerate(train_loader)):
        loss, acc = one_forward(data)

        one_backward(optimizer, loss)

        if step % 100 == 0:
            print('epoch{} step{} time:{}, loss:{}, acc:{}'.format(epoch, step, time.time() - start_time, loss, acc))
            start_time = time.time()

    # val
    print('val:')
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for step, data in tqdm(enumerate(test_loader)):
            loss, acc = one_forward(data)
            val_loss += loss.item()
            val_acc += acc
    print('{}th epoch val_loss: {}, val_acc:{}'.format(epoch, val_loss, val_acc))

    # save model
    torch.save(model, 'savedmodel/model_epoch{}.pth'.format(epoch))
