import time
import argparse
import json
import torch
from torch.nn import BCELoss
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

from datasets import *
from model import *
from embed import *
from losses import *
from metrics import *

# Load default configuration from JSON file
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'])
parser.add_argument('--max_seq_len', type=int, default=CONFIG['max_seq_len'])
parser.add_argument('--from_emb', type=bool, default=CONFIG['from_emb'])
parser.add_argument('--vocab_size', type=list, default=CONFIG['vocab_size'])
parser.add_argument('--emb_dim', type=int, default=CONFIG['emb_dim'])
parser.add_argument('--filter', type=list, default=CONFIG['filter'])
parser.add_argument('--linear_concat', type=int, default=CONFIG['linear_concat'])
parser.add_argument('--n_classes', type=list, default=CONFIG['n_classes'])
parser.add_argument('--train_path', type=str, default=CONFIG['train_path'])
parser.add_argument('--test_path', type=str, default=CONFIG['test_path'])
parser.add_argument('--learning_rate', type=float, default=CONFIG['learning_rate'])
parser.add_argument('--epochs_num', type=int, default=CONFIG['epochs_num'])
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

# Parse the arguments
args = parser.parse_args()

# Update CONFIG with command line arguments
CONFIG.update(vars(args))

### Load Configuration
# data
train_path = CONFIG['train_path']
test_path = CONFIG['test_path']
B_sz = CONFIG['batch_size']
# emb
TOKENIZER = CONFIG['tokenizer_name']
# model
MAX_SEQ_LEN = CONFIG['max_seq_len']
from_emb = CONFIG['from_emb']
# TODO: update in config
from_token = CONFIG['from_token']
EMB_DIM = CONFIG['emb_dim']
filter = CONFIG['filter']
linear_concat = CONFIG['linear_concat']
model_name = CONFIG['model_name']
# loss_fn
loss_name = CONFIG['loss_name']
# optimizer
optimizer_name = CONFIG['optimizer_name']
# train
EPOCH = CONFIG['epochs_num']
Learning_Rate = CONFIG['learning_rate']
device = CONFIG['device']

### train
# dataset
train_dataset = TextCodeDataset(train_path, pad_seq_len=MAX_SEQ_LEN, from_emb=from_emb)
test_dataset = TextCodeDataset(test_path, pad_seq_len=MAX_SEQ_LEN, from_emb=from_emb)

# TODO: optimize
pretrained_model = None
if from_emb:
    train_dataset.get_embedded(tokenizer=TOKENIZER, device=device, max_seq_len=MAX_SEQ_LEN)
    test_dataset.get_embedded(tokenizer=TOKENIZER, device=device, max_seq_len=MAX_SEQ_LEN)
elif from_token:
    train_dataset.get_tokenized(tokenizer=TOKENIZER, device=device, max_seq_len=MAX_SEQ_LEN)
    test_dataset.get_tokenized(tokenizer=TOKENIZER, device=device, max_seq_len=MAX_SEQ_LEN)
    _, pretrained_model = get_tokenizer_models(TOKENIZER, device)
vocab_size = tokenize_dataset_input(train_dataset, test_dataset)  # vocab_sz
idx2label = map_dataset_output(train_dataset, test_dataset)
n_classes = [len(x) for x in idx2label]  # n_classes

# dataloader
train_loader = DataLoader(train_dataset, batch_size=B_sz)
test_loader = DataLoader(test_dataset, batch_size=B_sz)

#TODO : add elif for other models, losses and optimizers
# model
if model_name == 'TextCNN':
    model = MetaModel(MAX_SEQ_LEN, from_emb, from_token, pretrained_model, vocab_size, EMB_DIM, filter, linear_concat, n_classes).to(device)

# loss_func
if loss_name == 'ASL':
    loss_fn = AsymmetricLossOptimized().to(device)

# optimizer
if optimizer_name == 'Adam':
    optimizer = Adam(model.parameters(), lr=Learning_Rate)


def one_forward(data):
    (x_context, x_AST), (y_dev, y_btype) = data
    # print(x_context.shape, x_AST.shape)
    y_dev_pred, y_btype_pred = model(x_context.to(device), x_AST.to(device))
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
