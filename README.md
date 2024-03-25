# Multi-triage
## requirements
pip3 install numpy pandas==1.2.0 Pyarrow nltk regex nlpaug tqdm transformers datetime keras tensorflow keras_preprocessing SentencePiece

## Parameters for train.py

| Parameter | Type | Default Value |
| --- | --- | --- |
| `--batch_size` | int | `CONFIG['batch_size']` |
| `--max_seq_len` | int | `CONFIG['max_seq_len']` |
| `--from_emb` | bool | `CONFIG['from_emb']` |
| `--vocab_size` | list | `CONFIG['vocab_size']` |
| `--emb_dim` | int | `CONFIG['emb_dim']` |
| `--filter` | list | `CONFIG['filter']` |
| `--linear_concat` | int | `CONFIG['linear_concat']` |
| `--n_classes` | list | `CONFIG['n_classes']` |
| `--train_path` | str | `CONFIG['train_path']` |
| `--test_path` | str | `CONFIG['test_path']` |
| `--learning_rate` | float | `CONFIG['learning_rate']` |
| `--epochs_num` | int | `CONFIG['epochs_num']` |
| `--device` | str | `'cuda' if torch.cuda.is_available() else 'cpu'` |

## Quick Start
```bash
python train.py --batch_size 128 --max_seq_len 500 --epochs_num 30 --learning_rate 0.001
```