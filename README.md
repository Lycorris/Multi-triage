# Multi-triage
## requirements
```bash
pip3 install numpy pandas==1.2.0 Pyarrow nltk regex nlpaug tqdm transformers datetime keras tensorflow keras_preprocessing SentencePiece
```
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

## File Tree
```bash
.
├─Code
│  ├─tf_codeV0
│  ├─torch_codeV0
│  ├─torch_codeV1
│  ├─torch_codeV2
│  └─utils
├─Data
│  ├─aspnet
│  ├─efcore
│  ├─elasticSearch
│  ├─mixedRealityToolUnity
│  ├─monoGame
│  ├─powershell
│  ├─realmJava
│  └─roslyn
├─Papers
│  ├─Baseline
│  ├─Embedding
│  │  ├─Code Embedding
│  │  └─Word Embedding
│  ├─Feature_Extracting
│  └─Multi-label_Methodology
│      ├─Multi-label
│      └─Semi-supervised
└─res_log
    ├─efcore
    ├─elasticSearch
    ├─mixedRealityToolUnity
    ├─monogame
    ├─powershell
    ├─realmJava
    └─roslyn
```

## Datasets
The meaning of the number following the dataset name is here:
```
'1' : Dataset with 'unknown' labels, and with empty AST
'2' : Dataset with 'unknown' labels, and without empty AST
'3' : Dataset without 'unknown' labels, and with empty AST
'4' : Dataset without 'unknown' labels, and without empty AST
```
```bash
Data
│  .DS_Store
│  README.md
│
├─aspnet
│      .DS_Store
│      aspnet.zip
│      aspnet_1.csv
│      aspnet_2.csv
│      aspnet_3.csv
│      aspnet_4.csv
│      IssueaspnetcoreWebScrap.csv
│
├─efcore
│      efcore_1.csv
│      efcore_2.csv
│      efcore_3.csv
│      efcore_4.csv
│      IssueefcoreWebScrap.csv
│
├─elasticSearch
│      .DS_Store
│      elasticSearch_1.csv.zip
│      elasticSearch_2.csv.zip
│      elasticSearch_3.csv.zip
│      elasticSearch_4.csv.zip
│      IssueelasticsearchWebScrap.csv.zip
│
├─mixedRealityToolUnity
│      IssuemixedrealitytoolkitunityWebScrap.csv
│      mixedRealityToolUnity_1.csv
│      mixedRealityToolUnity_2.csv
│      mixedRealityToolUnity_3.csv
│      mixedRealityToolUnity_4.csv
│
├─monoGame
│      IssuemonogameWebScrap.csv
│      monoGame_1.csv
│      monoGame_2.csv
│      monoGame_3.csv
│      monoGame_4.csv
│
├─powershell
│      .DS_Store
│      Issueazure-powershellWebScrap.csv
│      powerShell_1.csv
│      powerShell_2.csv
│      powerShell_3.csv
│      powerShell_4.csv
│
├─realmJava
│      IssuerealmjavaWebScrap.csv
│      realmJava_1.csv
│      realmJava_2.csv
│      realmJava_3.csv
│      realmJava_4.csv
│
└─roslyn
        IssueroslynWebScrap.csv
        roslyn_1.csv
        roslyn_2.csv
        roslyn_3.csv
        roslyn_4.csv
```