# TODO: complete this file
# Description: This file contains the configuration for the project
import torch
import json


CONFIG = {
    'batch_size': 64,
    'max_seq_len': 300,
    'from_emb': False,
    'vocab_size':[12513, 5910],
    'emb_dim': 100,
    'filter': [64, 64],
    'linear_concat': 50,
    'linear_concat': 128,
    'n_classes': [2, 2],
    'train_path':'Data/powershell/C_uA_Train.csv',
    'test_path' :'Data/powershell/C_uA_Test.csv',
    'learning_rate' :1e-3,
    'epochs_num' :30,
    'device' :'cuda' if torch.cuda.is_available() else 'cpu',
    'tokenizer_name': 'Albert',
    'loss_name': 'ASL',
    'optimizer_name': 'Adam',
    'model_name': 'TextCNN',
}

# Update the emb_dim according to the tokenizer
if(CONFIG['from_emb'] and CONFIG['tokenizer_name'] == 'Albert'):
    CONFIG['emb_dim'] = 768

# Save the CONFIG dictionary as a JSON file
with open('config.json', 'w') as f:
    json.dump(CONFIG, f)
