import time

import torch
from torch.nn import BCELoss
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from configuration import *
from datasets import *
from model import *
from embed import *
from losses import *
from metrics import *

### Configuration
# data
train_path = 'Data/powershell/C_uA_Train.csv'
test_path = 'Data/powershell/C_uA_Test.csv'
B_sz = 64
# emb
TOKENIZER = "Albert"
# model
MAX_SEQ_LEN = 300
from_emb = True
EMB_DIM = 100
filter = [64, 64]
linear_concat = 50
# loss_fn
loss_name = "ASL"
# TODO: config loss_fn type & params
# optimizer
# TODO: config optimizer type
# train
EPOCH = 20
Learning_Rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### train
# dataset
train_dataset = TextCodeDataset(train_path, pad_seq_len=MAX_SEQ_LEN, from_emb=from_emb)
test_dataset = TextCodeDataset(test_path, pad_seq_len=MAX_SEQ_LEN, from_emb=from_emb)

vocab_size = tokenize_dataset_input(train_dataset, test_dataset)  # vocab_sz
idx2label = map_dataset_output(train_dataset, test_dataset)
n_classes = [len(x) for x in idx2label]  # n_classes

# dataloader
train_loader = DataLoader(train_dataset, batch_size=B_sz)
test_loader = DataLoader(test_dataset, batch_size=B_sz)

# model
model = MetaModel(MAX_SEQ_LEN, from_emb, vocab_size, EMB_DIM, filter, linear_concat, n_classes)

# loss_func
loss_fn = AsymmetricLossOptimized()

# optimizer
optimizer = Adam(model.parameters(), lr=Learning_Rate)


def one_forward(data):
    (x_context, x_AST), (y_dev, y_btype) = data
    if from_emb:
        x_context = get_word_embedding(x_context, tokenizer=TOKENIZER, device=device)
        x_AST = get_word_embedding(x_AST, tokenizer=TOKENIZER, device=device)
    y_dev_pred, y_btype_pred = model(x_context.long().to(device), x_AST.long().to(device))
    y = torch.concat((y_dev, y_btype), 1).to(device)
    y_pred = torch.concat((y_dev_pred, y_btype_pred), 1)
    loss = loss_fn(y_pred, y)
    metric = metrics(y, y_pred, split_pos=n_classes)
    return loss, metric['acc'], metric['precision'], metric['recall'], metric['F1']


def one_backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()


# torch.autograd.set_detect_anomaly(True)
# scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter('./tf-logs')
for epoch in trange(EPOCH):

    # train
    model.train()
    for step, data in enumerate(train_loader):
        loss, acc, precision, recall, f1 = one_forward(data)
        one_backward(optimizer, loss)

    # val
    model.eval()
    val_loss, val_acc, val_precision, val_recall, val_F1 = 0, [0, 0], [0, 0], [0, 0], [0, 0]
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            loss, acc, precision, recall, f1 = one_forward(data)
            l = len(test_loader)
            val_loss += loss.item() / l
            val_acc[0] += acc[0] / l
            val_acc[1] += acc[1] / l
            val_precision[0] += precision[0] / l
            val_precision[1] += precision[1] / l
            val_recall[0] += recall[0] / l
            val_recall[1] += recall[1] / l
            val_F1[0] += f1[0] / l
            val_F1[1] += f1[1] / l

    print(
        '{}th epoch\n val_loss: {}\n val_acc:{}\n val_precision:{}\n val_recall:{}\n val_f1: {}'.format(epoch, val_loss,
                                                                                                        val_acc,
                                                                                                        val_precision,
                                                                                                        val_recall,
                                                                                                        val_F1))
    writer.add_scalar('val_loss' + loss_name + str(Learning_Rate), val_loss, epoch)
    writer.add_scalar('val_acc_d' + loss_name + str(Learning_Rate), val_acc[0], epoch)
    writer.add_scalar('val_acc_b' + loss_name + str(Learning_Rate), val_acc[1], epoch)
    # writer.add_scalar('val_precision'+loss_name+str(Learning_Rate), val_precision, epoch)
    # writer.add_scalar('val_recall'+loss_name+str(Learning_Rate), val_recall, epoch)
    writer.add_scalar('val_f1_d' + loss_name + str(Learning_Rate), val_F1[0], epoch)
    writer.add_scalar('val_f1_b' + loss_name + str(Learning_Rate), val_F1[1], epoch)

    # save model
    # if epoch % 20 == 0:
    #   torch.save(model, '../savedmodel/model_epoch{}.pth'.format(epoch))

writer.close()
