# TODO: complete this file
# Description: This file contains the configuration for the project
import torch
import json

CONFIG = {
    'batch_size': 16,
    'max_seq_len': 300,

    'from_emb': False,
    'from_token': False,
    'vocab_size': [12513, 5910],
    'emb_dim': 100,
    'filter': [64, 64],
    'linear_concat': 50,
    'n_classes': [2, 2],

    'train_path': '../Data/powershell/C_uA_Train.csv',
    'test_path': '../Data/powershell/C_uA_Test.csv',
    # 'train_path': '../Data/aspnet/trainAC40.csv',
    # 'test_path': '../Data/aspnet/testAC40.csv',
    'learning_rate': 1e-4,
    'epochs_num': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    # Albert / Bert / Bert-L / RoBerta
    'tokenizer_name': 'RoBerta',
    # BCE&MSE / C-BCE / ASL / BCE-Lg -- ASL & C-BCE GOOD!
    'loss_name': 'C-BCE',
    'optimizer_name': 'Adam',
    'model_name': 'TextCNN',
}

# Update the emb_dim according to the tokenizer
if not CONFIG['from_emb'] and not CONFIG['from_token']:
    CONFIG['emb_dim'] = 100
elif '-L' in CONFIG['tokenizer_name']:
    CONFIG['emb_dim'] = 1024
else:
    CONFIG['emb_dim'] = 768
    

# Save the CONFIG dictionary as a JSON file
with open('config.json', 'w') as f:
    json.dump(CONFIG, f)
