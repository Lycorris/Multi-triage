import pandas as pd
import torch
from torch.utils.data import Dataset
from functools import partial


def get_map(labels):
    l_set = set()
    for label in labels:
        l_set.update(label)
    ids2token = list(l_set)
    token2ids = {ids2token[i] : i for i in range(len(ids2token))}
    return ids2token, token2ids

def onehot(labels, token2ids):
    vec = [0 for i in token2ids]
    for label in labels.split('| '):
        vec[token2ids[label]] = 1
    return vec

def lab(labels, token2ids):
    return [token2ids[label] for label in labels.split('| ')]

def label_vectorize(data):
    data = data.rename(columns={'Title_Description' : 'Context', 'AST' : 'AST', 'FixedByID' : 'Dev', 'Name' : 'Btype'})
    data = data[['Context', 'AST', 'Dev', 'Btype']]
    # avoid NaN in dataset
    data['Context'].fillna('[UNK]', inplace=True)
    data['AST'].fillna('[UNK]', inplace=True)
    data['Dev'].fillna('unknown', inplace=True)
    data['Btype'].fillna('unknown', inplace=True)
    
    D_labels = [label.split('| ') for label in data['Dev']]
    _D_ids2token, D_token2ids = get_map(D_labels)
    data['Dev_l'] = data['Dev'].map(partial(lab, token2ids = D_token2ids))
    data['Dev_vec'] = data['Dev'].map(partial(onehot, token2ids = D_token2ids))
    
    B_labels = [label.split('| ') for label in data['Btype']]
    _B_ids2token, B_token2ids = get_map(B_labels)
    data['Btype_l'] = data['Btype'].map(partial(lab, token2ids = B_token2ids))
    data['Btype_vec'] = data['Btype'].map(partial(onehot, token2ids = B_token2ids))
    
    return data, _D_ids2token, _B_ids2token

def tokenize_function(_tokenizer, example, max_seq_len = 512):
    example = example if type(example) == str else _tokenizer.unk_token
    return _tokenizer(example, padding='max_length',
                                truncation=True, max_length=max_seq_len, return_tensors="pt")

def tensor_func(example):
    return torch.tensor(example)

class TextCodeDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def __getitem__(self, item):
        return (self.data['x_C'][item], self.data['x_A'][item]), self.data['y'][item]
    
    def __len__(self):
        return len(self.data)

