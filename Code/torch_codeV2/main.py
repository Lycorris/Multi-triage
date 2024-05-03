from train import *
import datetime

# path, name, avg_label_types, total_label_types
pathList = [
    # ('../Data/spring-framework.csv','spring_framework-Java',3.1979,57),
    # ('../Data/dotnet_roslyn.csv','dotnet_roslyn-C#',3.961,454),
    # ('../Data/TensorFlow.csv','TensorFlow-C++',4.994142627,735),
    # ('../Data/cockroach.csv','cockroach-Go',4.379664179,661),
    # ('../Data/dotnet_.csv','dotnet_aspnetcore-C#',4.019057172,464),
    ('../Data/dotnet_runtime.csv', 'dotnet_runtime-C#', 3.235663458, 1160),
    ('../Data/grpc.csv', 'grpc-C++', 4.36770428, 297),
    ('../Data/pytorch.csv', 'pytorch-C++', 4.082221036, 1140),
    ('../Data/golang.csv', 'golang-Go', 3.549056604, 275),
]

lossList = [
    (nn.BCEWithLogitsLoss(), 'BCE'),
    ('SparceBCELoss', 'SBCE'),
    (AsymmetricLossOptimized(), 'ASL'),
]
# TODO: XLnet
ckptList = [
    ('bert-base-uncased', 'Multi-triage', 20),  # just for tokenize
    ('bert-base-uncased', ' Bert', 10),
    ('roberta-base', 'Robert', 10),
    ('albert-base-v2', 'albert', 10),
    ('codebert-base', 'codebert', 10),
    #XLnet
]
# TODO: code-only
codeFormatList = [
    'None',
    'Front',
    'Back',
    'Separate',
    'raw',
    # 'code',
]

# result_col for each dataset
res_columns = ['train_method', 'loss_type', 'model_type', 'code_format', 'proj_name', 'epoch', 'res1', 'res2']

for path in pathList:
    # create 'res_DataFrame' for each dataset
    result = pd.DataFrame(columns=res_columns)
    for ckpt in ckptList:
        for loss in lossList:
            for code_format in codeFormatList:
                try:
                    # obtain loss_fn & model_type
                    loss_fn = loss[0] if loss[1] != 'SBCE' else SparceBCELoss(avg_label_types=path[2],
                                                                              total_label_types=path[3])
                    model_type = 'Multi-triage' if ckpt[1] == 'Multi-triage' else 'PreTrain'
                    # set log
                    logname = ' '.join([path[1], ckpt[1], loss[1], code_format, str(ckpt[2])])
                    print('-' * 100, logname, '-' * 100, sep='\n')
                    # train model
                    ress = train_imm(_path=path[0], _logname=logname,
                                     _num_epochs=ckpt[2],
                                     _loss_fn=loss_fn, _code_format=code_format, _model_type=model_type,
                                     _ckpt=ckpt[0])
                    # save f1-score 'for each head(dev&btype)' 'for every 5 epoch'
                    for i, res in enumerate(ress):
                        print(res)
                        result.loc[result.shape[0]] = ['NaiveTraining', loss[1], ckpt[1], code_format, path[1], 5 * (i + 1), res[0], res[1]]
                except Exception as e:
                    print(e)
                    continue
    # save 'res_DataFrame' for each dataset
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result.to_csv(f'../res/{path[1]}_{now}_result.csv', index=False)
