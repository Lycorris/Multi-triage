from transformers import AdamW, get_scheduler
from tqdm import trange, tqdm

from dataset import *
from model import *
from losses import *
from metrics import *

TOPK = [1, 2, 3, 5, 10, 20]
METRICS_NAME = ['acc', 'precision', 'recall', 'F1'] + [f'acc@{k}' for k in TOPK]


def process_data(x, y, device):
    x_C = {k: v.to(device) for k, v in x[0].items()}
    x_A = {k: v.to(device) for k, v in x[1].items()}
    y = y.to(device)
    return x_C, x_A, y


def train_process(model, train_dataloader, loss_fn, optimizer, lr_scheduler,
                  device='cuda' if torch.cuda.is_available() else 'cpu'):
    # train
    model.train()
    train_loss = 0.0
    for x, y in train_dataloader:
        x_C, x_A, y = process_data(x, y, device)
        outputs = model(x_C, x_A)
        loss = loss_fn(outputs, y.float())
        train_loss += loss.item() / len(train_dataloader)
        # back
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    return train_loss


def test_process(model, test_dataloader, loss_fn, n_classes,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.eval()
    test_metric = {k: (0.0, 0.0) for k in METRICS_NAME}
    for x, y in tqdm(test_dataloader):
        x_C, x_A, y = process_data(x, y, device)
        outputs = model(x_C, x_A)
        loss = loss_fn(outputs, y.float())

        metric = metrics(y, outputs, split_pos=n_classes)
        test_metric = update_metric(test_metric, metric, len(test_dataloader))

    return test_metric


def update_metric(hist_metric, metric, l):
    # a(acc), p(precision), r(recall), F(F1)
    # acc @ (1, 2, 3, 5, 10, 20)
    hist_metric = {
        k: (v[0] + (metric[k][0] / l), v[1] + (metric[k][1] / l))
        for k, v in hist_metric.items()
    }
    return hist_metric


def update_logstr(log_str, epoch=None, train_loss=None, val_metric=None, t=None,
                  test_metric=None, avg_test_metric=None):
    if train_loss is not None:
        sub_str = '{}th epoch\n train_loss: {}\n'.format(epoch, train_loss)
    if val_metric is not None:
        sub_str = ('-' * 60 + '\n{}th epoch\n {}\n\n'.format(epoch, val_metric))
    if t is not None:
        sub_str = '-' * 60 + f'TESTSET{t}' + '-' * 60 + '\n' + '-' * 120 + '\n'
        for k, v in test_metric.items():
            sub_str += f'{k}: {v}\n'
        print(sub_str)
        log_str += sub_str
    if avg_test_metric is not None:
        sub_str = '-' * 60 + f'ALLTESTSET' + '-' * 60 + '\n' + '-' * 120 + '\n'
        for k, v in avg_test_metric.items():
            sub_str += f'{k}: {v}\n'
        print(sub_str)

    log_str += sub_str
    return log_str


def train_imm(_path, _logname, _loss_fn, _code_format='None',
              _num_epochs=20, _bsz=4, _lr=3e-5,
              _ckpt='bert-base-uncased', _code_ckpt='codebert-base',
              _model_ckpt='../model_ckpts/9th_epoch__Bert_ASL.pkl',
              _model_type='Multi-triage', exp=1,
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

    # DATASET Format: split train/val/test dataset and wrap into dataloader
    train_dataloader, val_dataloader, test_dataloaders = \
        split_and_wrap_dataset_multi_repo(datasets, _bsz)

    # log
    n_classes = [len(D_ids2token), len(B_ids2token)]
    res = []
    logname = '../res_log/' + _logname + '.txt'
    logstr = (_logname + '\\n' + '-' * 60 + '\\n' + 'dataset shape:{}\nn_classes: {}'.
              format(len(train_dataloader) / 0.8, n_classes) + '-' * 60 + '\n\n')
    print('dataset shape:{}\nn_classes: {}'.format(len(train_dataloader) / 0.8, n_classes))

    # MODEL: load model
    model = get_model(_model_type, exp, n_classes, _code_format,
                      _ckpt, _code_ckpt, _model_ckpt, device)

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

        # # 3，4由于已经domain-align过可能需要一个epoch一test
        # if epoch % 5 == 4 or exp in [3, 4]:
        if epoch % 5 == 4:
            avg_test_metric = {k: (0.0, 0.0) for k in METRICS_NAME}
            t_ds_len = sum([len(t_d) for t_d in test_dataloaders])

            for t, test_dataloader in enumerate(test_dataloaders):
                test_metric = test_process(model, test_dataloader, loss_fn, n_classes)
                logstr = update_logstr(logstr, t=t, test_metric=test_metric)

                avg_test_metric = {k: (v[0] + test_metric[k][0] * len(test_dataloader) / t_ds_len,
                                       v[1] + test_metric[k][1] * len(test_dataloader) / t_ds_len)
                                   for k, v in avg_test_metric.items()}
            if len(test_dataloaders) > 1:
                logstr = update_logstr(logstr, avg_test_metric=avg_test_metric)
            res.append(avg_test_metric)
            if exp == 2:
                model_path = '_'.join(_logname.split(' ')[1:4])
                print(f'model_path: {model_path}')
                torch.save(model, f'../model_ckpts/{epoch}th_epoch_{model_path}.pkl')  # 保存模型

    with open(logname, 'w') as f:
        f.write(logstr)

    # Return metrics
    return res


if __name__ == '__main__':
    pass
