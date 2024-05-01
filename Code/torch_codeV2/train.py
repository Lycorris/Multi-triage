import pandas as pd

from transformers import AdamW, get_scheduler
from tqdm import trange, tqdm

from dataset import *
from model import *
from losses import *
from metrics import *


def update_metric(hist_metric, loss, metric, l):
    hist_metric[0] += loss.item() / l
    hist_metric[1] += metric['acc'][0] / l
    hist_metric[2] += metric['acc'][1] / l
    hist_metric[3] += metric['F1'][0] / l
    hist_metric[4] += metric['F1'][1] / l
    return hist_metric


def precess_data(x, y, device):
    x_C = {k: v.to(device) for k, v in x[0].items()}
    x_A = {k: v.to(device) for k, v in x[1].items()}
    y = y.to(device)
    return x_C, x_A, y


def train_imm(_path, _logname, _loss_fn, _code_format='None', _model_type='Multi-triage',
              _num_epochs=20, _bsz=8, _lr=3e-5,
              _ckpt='bert-base-uncased', _code_ckpt='codebert-base',
              device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
        _code_format = 'None'  -> ignore Code Snippet
                        'Front' -> add Code BEFORE Text
                        'Back'  -> add Code BEHIND Text
                        'Separate' -> consider Code as an independent input
        _model_type =  'Multi-triage'
                        'PreTrain'
    """
    # DATASET Read Data: extract data
    dataset = pd.read_csv(_path)
    dataset = dataset.rename(
        columns={'Title_Description': 'Context', 'AST': 'AST', 'FixedByID': 'Dev', 'Name': 'Btype'})
    dataset = dataset[['Context', 'AST', 'Dev', 'Btype']]

    # DATASET Output: convert label to tensor
    dataset, D_ids2token, B_ids2token = label_vectorize(dataset)
    n_classes = [len(D_ids2token), len(B_ids2token)]
    # log
    logname = '../res_log/' + _logname + '.txt'
    logstr = _logname + '\\n' + '-' * 60 + '\\n' + 'dataset shape:{}\nn_classes: {}'.format(dataset.shape,
                                                                                            n_classes) + '-' * 60 + '\n\n'
    print('dataset shape:{}\nn_classes: {}'.format(dataset.shape, n_classes))

    # DATASET Input: convert text to tensor
    dataset = text_tensorize(dataset, _ckpt, _code_format)

    # DATASET Format: split train/val/test dataset and wrap into dataloader
    train_dataloader, val_dataloader, test_dataloader = \
        split_and_wrap_dataset(dataset, _bsz)

    # MODEL load model
    if _model_type == 'Multi-triage':
        model = MetaModel(n_classes=n_classes, use_AST=(_code_format == 'Separate'))
    else:
        model = PretrainModel(text_ckpt=_ckpt, code_ckpt=_code_ckpt, n_classes=n_classes,
                              use_AST=(_code_format == 'Separate'))
    model = model.to(device)

    # loss
    loss_fn = _loss_fn.to(device)

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=_lr)

    # lr_scheduler
    num_epochs = _num_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps, )

    # train process
    for epoch in trange(num_epochs):
        # train
        model.train()
        train_loss = 0.0
        for x, y in train_dataloader:
            x_C, x_A, y = precess_data(x, y, device)
            outputs = model(x_C, x_A)
            loss = loss_fn(outputs, y.float())
            train_loss += loss.item() / len(train_dataloader)
            # back
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
        logstr += '{}th epoch\n train_loss: {}\n'.format(epoch, train_loss)

        # val
        model.eval()
        val_metric = [0.0] * 5
        for x, y in val_dataloader:
            x_C, x_A, y = precess_data(x, y, device)
            outputs = model(x_C, x_A)
            loss = loss_fn(outputs, y.float())

            metric = metrics(y, outputs, split_pos=n_classes)
            val_metric = update_metric(val_metric, loss, metric, len(val_dataloader))

        logstr += '-' * 60 + '{}th epoch\n val_loss: {}\n val_acc:{}\n val_f1: {}\n'.format(epoch, val_metric[0],
                                                                                            val_metric[1:3],
                                                                                            val_metric[3:])

    # test
    model.eval()
    test_metric = [0.0] * 5
    for x, y in tqdm(test_dataloader):
        x_C, x_A, y = precess_data(x, y, device)
        outputs = model(x_C, x_A)
        loss = loss_fn(outputs, y.float())

        metric = metrics(y, outputs, split_pos=n_classes)
        test_metric = update_metric(test_metric, loss, metric, len(test_dataloader))

    logstr += '-' * 60 + '\ntest_loss: {}\n test_acc:{}\n test_f1: {}'.format(test_metric[0], test_metric[1:3],
                                                                              test_metric[3:])
    print('test_loss: {}\n test_acc:{}\n test_f1: {}'.format(test_metric[0], test_metric[1:3], test_metric[3:]))

    with open(logname, 'w') as f:
        f.write(logstr)


if __name__ == '__main__':
    pass
