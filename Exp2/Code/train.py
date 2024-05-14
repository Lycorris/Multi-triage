import pandas as pd

from transformers import AdamW, get_scheduler
from tqdm import trange, tqdm

from dataset import *
from model import *
from losses import *
from metrics import *

NUM_METRICS = 17

def update_metric(hist_metric, loss, metric, l):
    hist_metric[0] += loss.item() / l
    hist_metric[1] += metric['acc'][0] / l
    hist_metric[2] += metric['acc'][1] / l
    hist_metric[3] += metric['F1'][0] / l
    hist_metric[4] += metric['F1'][1] / l
    hist_metric[5] += metric['acc@1_d'] / l
    hist_metric[6] += metric['acc@2_d'] / l
    hist_metric[7] += metric['acc@3_d'] / l
    hist_metric[8] += metric['acc@5_d'] / l
    hist_metric[9] += metric['acc@10_d'] / l
    hist_metric[10] += metric['acc@20_d'] / l
    hist_metric[11] += metric['acc@1_b'] / l
    hist_metric[12] += metric['acc@2_b'] / l
    hist_metric[13] += metric['acc@3_b'] / l
    hist_metric[14] += metric['acc@5_b'] / l
    hist_metric[15] += metric['acc@10_b'] / l
    hist_metric[16] += metric['acc@20_b'] / l
    return hist_metric


def precess_data(x, y, device):
    x_C = {k: v.to(device) for k, v in x[0].items()}
    x_A = {k: v.to(device) for k, v in x[1].items()}
    y = y.to(device)
    return x_C, x_A, y


def test_process(model, test_dataloader, loss_fn, n_classes,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    test_metric = [0.0] * NUM_METRICS
    for x, y in tqdm(test_dataloader):
        x_C, x_A, y = precess_data(x, y, device)
        outputs = model(x_C, x_A)
        loss = loss_fn(outputs, y.float())

        metric = metrics(y, outputs, split_pos=n_classes)
        test_metric = update_metric(test_metric, loss, metric, len(test_dataloader))

    return test_metric


def train_process(model, train_dataloader, loss_fn, optimizer, lr_scheduler,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
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

    return train_loss

# TODO: more metrics to be added
def update_logstr(log_str, epoch=None, train_loss=None, val_metric=None, t=None,
                  test_metric=None, avg_test_metric=None):
    if train_loss is not None:
        sub_str = '{}th epoch\n train_loss: {}\n'.format(epoch, train_loss)
    if val_metric is not None:
        sub_str = ('-' * 60 + '{}th epoch\n val_loss: {}\n val_acc:{}\n val_f1: {}\n'.
                    format(epoch, val_metric[0], val_metric[1:3], val_metric[3:5]))
    if t is not None:
        sub_str = '-' * 60 + f'TESTSET{t}' + '-' * 60 + '\n'
        sub_str += ('-' * 60 + '\ntest_loss: {}\n test_acc:{}\n test_f1: {}'.
                    format(test_metric[0], test_metric[1:3], test_metric[3:5]))
        print(sub_str)
        log_str += sub_str
    if avg_test_metric is not None:
        sub_str = '-' * 60 + f'ALLTESTSET' + '-' * 60 + '\n'
        sub_str += ('-' * 60 + '\ntest_loss: {}\n test_acc:{}\n test_f1: {}'.
                    format(avg_test_metric[0], avg_test_metric[1:3], avg_test_metric[3:5]))
        print(sub_str)

    log_str += sub_str
    return log_str



def train_imm(_path, _logname, _loss_fn, _code_format='None', _model_type='Multi-triage',
              _num_epochs=20, _bsz=4, _lr=3e-5,
              _ckpt='bert-base-uncased', _code_ckpt='codebert-base', use_AST=False, exp=1,
              device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
        _code_format =  'None'  -> ignore Code Snippet
                        'Front' -> add Code BEFORE Text
                        'Back'  -> add Code BEHIND Text
                        'Separate' -> consider Code as an independent input
                        'raw'   -> add the original Code Snippet
        _model_type =   'Multi-triage'
                        'PreTrain'
    """
    # Dataset Preprocess on Input & Output
    datasets, (D_ids2token, B_ids2token) = dataset_preprocess_multi_repo(_path, _code_format, _ckpt)

    # small dataset
    datasets = datasets[:100]

    # DATASET Format: split train/val/test dataset and wrap into dataloader
    train_dataloader, val_dataloader, test_dataloaders = \
        split_and_wrap_dataset_multi_repo(datasets, _bsz)

    if exp == 2:
        # ckpt
        model_path = '_'.join(_logname.split(' ')[1:4])
        print(f'model_path: {model_path}')

    # log
    n_classes = [len(D_ids2token), len(B_ids2token)]
    res = []
    logname = '../res_log/' + _logname + '.txt'
    logstr = (_logname + '\\n' + '-' * 60 + '\\n' + 'dataset shape:{}\nn_classes: {}'.
              format(len(train_dataloader) / 0.8, n_classes) + '-' * 60 + '\n\n')
    print('dataset shape:{}\nn_classes: {}'.format(len(train_dataloader) / 0.8, n_classes))

    # MODEL: load model
    if _model_type == 'Multi-triage':
        model = MetaModel(n_classes=n_classes, use_AST=(_code_format == 'Separate'))
    elif exp in [1, 2]:
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
        train_loss = train_process(model, train_dataloader, loss_fn, optimizer, lr_scheduler)
        logstr = update_logstr(logstr, epoch=epoch, train_loss=train_loss)

        # val
        val_metric = test_process(model, val_dataloader, loss_fn, n_classes)
        logstr = update_logstr(logstr, epoch=epoch, train_loss=None, val_metric=val_metric)

        if epoch % 5 == 4:
            avg_test_metric = [0.0] * NUM_METRICS
            t_ds_len = sum([len(t_d) for t_d in test_dataloaders])
            for t, test_dataloader in enumerate(test_dataloaders):
                test_metric = test_process(model, test_dataloader, loss_fn, n_classes)
                avg_test_metric = [avg_test_metric[i] + test_metric[i] * len(test_dataloader) / t_ds_len
                                   for i in range(len(avg_test_metric))]
                logstr = update_logstr(logstr, t=t, test_metric=test_metric)

            logstr = update_logstr(logstr, avg_test_metric=avg_test_metric)
            res.append(avg_test_metric)
            if exp == 2:
                torch.save(model, f'../model_ckpts/{epoch}th_epoch_' + model_path)  # 保存模型
                # net_ = torch.load(pth) # 读取模型

    with open(logname, 'w') as f:
        f.write(logstr)

    # Return metrics
    return res


if __name__ == '__main__':
    pass
