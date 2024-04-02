# TODO: complete this file
# Description: This file contains the configuration for the project
import torch

CONFIG = {
    'batch_size': 64,
    'max_seq_len': 300,
    'from_emb': False,
    'vocab_size':[12513, 5910],
    'emb_dim': 100,
    'filter': [64, 64],
    'linear_concat': 128,
    'n_classes': [2, 2],
    'train_path':'Data/powershell/C_uA_Train.csv',
    'test_path' :'Data/powershell/C_uA_Test.csv',
    'learning_rate' :1e-3,
    'epochs_num' :20,
    'device' :'cuda' if torch.cuda.is_available() else 'cpu',
}
