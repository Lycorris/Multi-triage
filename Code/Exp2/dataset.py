from functools import partial

import pandas as pd
from transformers import AutoTokenizer

import torch
from torch.utils.data import Dataset, DataLoader

label_split_token = '|'


# dataset utils

### Output: Label Process ###
# obtain label(split by '|) set & ont-hot dictionary
def get_map(labels):
    l_set = set()
    for label in labels:
        l_set.update(label)
    ids2token = list(l_set)
    token2ids = {ids2token[i]: i for i in range(len(ids2token))}
    return ids2token, token2ids


# obtain one-hot vec
def onehot(labels, token2ids):
    vec = [0 for i in token2ids]
    for label in labels.split(label_split_token):
        vec[token2ids[label]] = 1
    return vec


def label_vectorize(data, codeFormat='None'):
    # avoid NaN in dataset
    data['Dev'].fillna('unknown', inplace=True)
    data['Btype'].fillna('unknown', inplace=True)

    D_labels = [label.split(label_split_token) for label in data['Dev']]
    _D_ids2token, D_token2ids = get_map(D_labels)
    data['Dev_vec'] = data['Dev'].map(partial(onehot, token2ids=D_token2ids))

    B_labels = [label.split(label_split_token) for label in data['Btype']]
    _B_ids2token, B_token2ids = get_map(B_labels)
    data['Btype_vec'] = data['Btype'].map(partial(onehot, token2ids=B_token2ids))

    data['y'] = data['Dev_vec'] + data['Btype_vec']
    data['y'] = data['y'].apply(lambda example: torch.tensor(example))

    return data, _D_ids2token, _B_ids2token


def label_vectorize_by_dict(data, label_dict):
    data['Dev_vec'] = data['Dev'].map(partial(onehot, token2ids=label_dict[0]))
    data['Btype_vec'] = data['Btype'].map(partial(onehot, token2ids=label_dict[1]))

    data['y'] = data['Dev_vec'] + data['Btype_vec']
    data['y'] = data['y'].apply(lambda example: torch.tensor(example))
    return data


### Input: Text/Code Process ###
def tokenize_function(_tokenizer, example, max_seq_len=512):
    example = example if type(example) == str else _tokenizer.unk_token
    return _tokenizer(example, padding='max_length',
                      truncation=True, max_length=max_seq_len, return_tensors="pt")


def text_tensorize(data, _ckpt, _code_format):
    """
        _code_format = 'None'  -> ignore Code Snippet
                       'Front' -> add Code BEFORE Text
                       'Back'  -> add Code BEHIND Text
                       'Separate' -> consider Code as an independent input
    """
    # avoid NaN in dataset
    data['Context'].fillna('[UNK]', inplace=True)
    data['AST'].fillna('[UNK]', inplace=True)
    # obtain tokenizer
    check_point = _ckpt
    tokenizer = AutoTokenizer.from_pretrained(check_point, local_files_only=True)
    # process code
    # TODO: sanity check
    if _code_format == 'Front':
        data['Context'] = data['AST'] + data['Context']
    elif _code_format == 'Back':
        data['Context'] = data['Context'] + data['AST']
    elif _code_format == 'raw':
        data['Context'] = data['raw_Title_Description']
    # dataset tensorize
    data['x_C'] = data['Context'].map(partial(tokenize_function, tokenizer))
    data['x_A'] = data['AST'].map(partial(tokenize_function, tokenizer))
    return data


### Format: wrap into Pytorch Dataloader ###
class TextCodeDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, item):
        return (self.data['x_C'][item], self.data['x_A'][item]), self.data['y'][item]

    def __len__(self):
        return len(self.data)


### One Repo: split train/test and dataset process ###
def dataset_preprocess(_path, _code_format, _ckpt):
    # DATASET Read Data: extract data
    dataset = pd.read_csv(_path)

    # 缩小数据集
    # dataset = dataset[:50]

    dataset = dataset.rename(
        columns={'Title_Description': 'Context', 'AST': 'AST', 'FixedByID': 'Dev', 'Name': 'Btype'})
    dataset = dataset[['Context', 'AST', 'Dev', 'Btype', 'raw_Title_Description']]

    # DATASET Input: convert text to tensor
    dataset = text_tensorize(dataset, _ckpt, _code_format)

    # DATASET Output: convert label to tensor
    dataset, D_ids2token, B_ids2token = label_vectorize(dataset, codeFormat=_code_format)

    return dataset, (D_ids2token, B_ids2token)


def split_and_wrap_dataset(data, _bsz, test_ratio=0.2, train_val_ratio=0.8):
    train_ratio = 1 - test_ratio

    # split train/val/test
    t_dataset = data[:int(train_ratio * len(data))].reset_index(drop=True)
    train_dataset = t_dataset.sample(frac=train_val_ratio, random_state=0, axis=0).reset_index(drop=True)
    val_dataset = t_dataset[~t_dataset.index.isin(train_dataset.index)].reset_index(drop=True)
    test_dataset = data[int(train_ratio * len(data)):].reset_index(drop=True)

    # wrap dataset & dataloader
    train_dataset = TextCodeDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=_bsz, drop_last=True)
    val_dataset = TextCodeDataset(val_dataset)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=_bsz, drop_last=True)
    test_dataset = TextCodeDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=_bsz, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader


### Multi Repo: split train/test and dataset process ###
# TODO: extra training repo (_extra_paths)
def dataset_preprocess_multi_repo(_paths, _code_format, _ckpt):
    """
    params:
        _paths: list of target repo path(train & test)
        _code_format: details in 'text_tensorize'
        _ckpt: tokenizer ckpt for text/code input
    return:
        a list of processed dataset(in Tensor format)
    """
    zips = [dataset_preprocess(_path, _code_format, _ckpt) for _path in _paths]

    datasets = [z[0] for z in zips]
    D_ids2tokens, B_ids2tokens = [z[1][0] for z in zips], [z[1][1] for z in zips]

    D_ids2token = list(set(D for D_ids2token in D_ids2tokens for D in D_ids2token))
    B_ids2token = list(set([B for B_ids2token in B_ids2tokens for B in B_ids2token]))

    D_tokens2ids = {D_token: D_id for D_id, D_token in enumerate(D_ids2token)}
    B_tokens2ids = {B_token: B_id for B_id, B_token in enumerate(B_ids2token)}

    # reconvert datasets output
    datasets = [label_vectorize_by_dict(dataset, (D_tokens2ids[i], B_tokens2ids[i]))
                for i, dataset in enumerate(datasets)]

    return datasets, (D_ids2token, B_ids2token)


def split_and_wrap_dataset_multi_repo(datas, _bsz, test_ratio=0.2, train_val_ratio=0.8):
    """
    params:
        datas: a list of precessed data of different repos
    return:
        train/val_dataloader
        test_dataloaders: a list of test_dataloader for each repo
    """
    train_ratio = 1 - test_ratio

    # extract 80%(train_ratio) data from each repo and cancat them as Stage1 train-set
    t_datas = pd.concat([data[:int(train_ratio * len(data))] for data in datas],
                        ignore_index=True)
    # train/val
    train_dataset = t_datas.sample(frac=train_val_ratio, random_state=0, axis=0).reset_index(drop=True)
    val_dataset = t_datas[~t_datas.index.isin(train_dataset.index)].reset_index(drop=True)

    # extract 20%(test_ratio) data from each repo
    test_datas = [data[int(train_ratio * len(data)):] for data in datas]

    # wrap dataset & dataloader
    train_dataset = TextCodeDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=_bsz, drop_last=True)
    val_dataset = TextCodeDataset(val_dataset)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=_bsz, drop_last=True)
    test_datasets = [TextCodeDataset(test_data) for test_data in test_datas]
    test_dataloaders = [DataLoader(test_dataset, shuffle=True, batch_size=_bsz, drop_last=True)
                        for test_dataset in test_datasets]

    return train_dataloader, val_dataloader, test_dataloaders
