from train import *
import datetime

EXP = 1

# Datasets
# path, name, avg_label_types, total_label_types
pathList = [
    (['../Data/cockroach.csv'], 'cockroach-Go', 4.379664179, 661),
    (['../Data/dotnet_aspnetcore.csv'], 'dotnet_aspnetcore-C#', 4.019057172, 464),
    (['../Data/dotnet_roslyn.csv'], 'dotnet_roslyn-C#', 3.961, 454),
    (['../Data/dotnet_runtime.csv'], 'dotnet_runtime-C#', 3.235663458, 1160),  # 4778 20h
    (['../Data/golang.csv'], 'golang-Go', 3.549056604, 275),  # 530 2h
    (['../Data/grpc.csv'], 'grpc-C++', 4.36770428, 297),  # 514 2h
    (['../Data/pytorch.csv'], 'pytorch-C++', 4.082221036, 1140),  # 1873 6h
    (['../Data/spring-framework.csv'], 'spring_framework-Java', 3.1979, 57),
    (['../Data/TensorFlow.csv'], 'TensorFlow-C++', 4.994142627, 735),
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
                    for i, res in enumerate(ress):
                        res_val = [v[i] for k, v in res.items() for i in range(2) if k in METRICS_NAME]
                        result.loc[result.shape[0]] = ['NaiveTraining', loss[1], ckpt[1], code_format,
                                                       path[1], 5 * (i + 1)] + res_val

                # except Exception as e:
                #     print(e)

        # save 'res_DataFrame' for each dataset
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result.to_csv(f'../res/{path[1]}_{now}_result.csv', index=False)