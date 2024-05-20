from train import *
import datetime

EXP = 2

# Datasets
# path, name, avg_label_types, total_label_types
pathList = [
    (['../Data/cockroach.csv',
      '../Data/dotnet_aspnetcore.csv',
      '../Data/dotnet_roslyn.csv',
      '../Data/dotnet_runtime.csv',
      '../Data/golang.csv',
      '../Data/grpc.csv',
      '../Data/pytorch.csv',
      '../Data/spring-framework.csv',
      '../Data/TensorFlow.csv', ],
     ['cockroach-Go', 'dotnet_aspnetcore-C#', 'dotnet_roslyn-C#',
      'dotnet_runtime-C#', 'golang-Go', 'grpc-C++',
      'pytorch-C++', 'spring_framework-Java', 'TensorFlow-C++'],
     4, 2600)
]

# Models
# MT, BERT, RoBERTa, ALBERT, CodeBERT
ckptList = [
    # ('bert-base-uncased', 'Multi-triage', 20),  # just for tokenize
    ('bert-base-uncased', ' Bert', 10),
    ('roberta-base', 'Robert', 10),
    ('albert-base-v2', 'albert', 10),
    ('codebert-base', 'codebert', 10),
]

# Losses
# BCE, SBCE, ASL, SACL, Focal
lossList = [
    (nn.BCEWithLogitsLoss(), 'BCE'),
    ('SparceBCELoss', 'SBCE'),
    (AsymmetricLossOptimized(), 'ASL'),
    ('SparceASL', 'SASL'),
    (FocalLoss(), 'Focal'),
]


def get_loss(loss, alt, tlt):
    loss_fn, loss_name = loss
    if loss_name == 'SBCE':
        loss_fn = SparceBCELoss(avg_label_types=alt, total_label_types=tlt)
    elif loss_name == 'SASL':
        loss_fn = SparceAsymmetricLoss(avg_label_types=alt, total_label_types=tlt)
    return loss_fn


# Code formats
# Raw, Front, Back, None, Separate
codeFormatList = [
    'Raw',
    # 'Front',
    # 'Back',
    # 'None',
    # 'Separate',

]

# Metrics
TOPK = [1, 2, 3, 5, 10, 20]
METRICS_NAME = ['F1'] + [f'acc@{k}' for k in TOPK]
OUTPUT = ['d', 'b']

# Result Log
# result_col for each dataset
res_columns = ['train_method', 'loss_type', 'model_type', 'code_format', 'proj_name', 'epoch', ] \
              + [m_name + '_' + o for m_name in METRICS_NAME for o in OUTPUT]

for path in pathList:
    # create 'res_DataFrame' for each dataset
    result = pd.DataFrame(columns=res_columns)

    for ckpt in ckptList:
        model_type = 'Multi-triage' if ckpt[1] == 'Multi-triage' else 'PreTrain'

        for loss in lossList:
            loss_fn = get_loss(loss, path[2], path[3])

            for code_format in codeFormatList:
                # set log
                logname = ' '.join([path[1], ckpt[1], loss[1], code_format, str(ckpt[2])])
                print('-' * 100, logname, '-' * 100, sep='\n')

                # try:
                if True:
                    # train model and obtain results
                    ress = train_imm(_path=path[0], _logname=logname,
                                     _num_epochs=ckpt[2],
                                     _loss_fn=loss_fn, _code_format=code_format, _model_type=model_type,
                                     _ckpt=ckpt[0], exp=EXP)
                    # record each result in ress
                    n_repo = len(path[1])
                    n_epoch = int(len(ress) / n_repo)
                    assert n_epoch % n_repo == 0
                    # TODO: check MT 4 / PTM 2
                    for i in range(n_epoch):
                        for j in range(n_repo):
                            res = ress[i * n_repo + j]
                            res_val = [v[i] for k, v in res.items() for i in range(2) if k in METRICS_NAME]
                            result.loc[result.shape[0]] = ['Domain-Alignment', loss[1], ckpt[1], code_format,
                                                           path[1][j], 5 * (i + 1)] + res_val

                # except Exception as e:
                #     print(e)

        # save 'res_DataFrame' for each dataset
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result.to_csv(f'../res/exp2_{now}_result.csv', index=False)



