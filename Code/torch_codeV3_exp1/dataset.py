from functools import partial
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


def label_vectorize(data, codeFormat='None', use_AST=False):
    # avoid NaN in dataset
    data['Context'].fillna('[UNK]', inplace=True)
    data['AST'].fillna('[UNK]', inplace=True)
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
    # obtain tokenizer
    check_point = _ckpt
    tokenizer = AutoTokenizer.from_pretrained(check_point,local_files_only=True)
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


def split_and_wrap_dataset(data, _bsz, test_ratio=0.2, train_val_ratio=0.8):
    # split train/val/test
    t_dataset = data[:int((1 - test_ratio) * len(data))].reset_index(drop=True)
    train_dataset = t_dataset.sample(frac=train_val_ratio, random_state=0, axis=0).reset_index(drop=True)
    val_dataset = t_dataset[~t_dataset.index.isin(train_dataset.index)].reset_index(drop=True)
    test_dataset = data[int((1 - test_ratio) * len(data)):].reset_index(drop=True)

    # wrap dataset & dataloader
    train_dataset = TextCodeDataset(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=_bsz, drop_last=True)
    val_dataset = TextCodeDataset(val_dataset)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=_bsz, drop_last=True)
    test_dataset = TextCodeDataset(test_dataset)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=_bsz, drop_last=True)

    return train_dataloader, val_dataloader, test_dataloader
