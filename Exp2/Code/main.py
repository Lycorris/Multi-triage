from train import *
import datetime

EXP = 2

# path, name, avg_label_types, total_label_types
pathList = [
    # ('/root/autodl-tmp/Code-copy/Data/spring-framework_processed_md_ast_without_empty_all_Java.csv','spring_framework-Java',3.1979,57),
    # ('/root/autodl-tmp/Code-copy/Data/dotnet_roslyn_processed_md_ast_without_empty_all_C#.csv','dotnet_roslyn-C#',3.961,454),
    # ('/root/autodl-tmp/Code-copy/Data/TensorFlow_processed_md_ast_without_empty_all_C++.csv','TensorFlow-C++',4.994142627,735),
    # ('/root/autodl-tmp/Code-copy/Data/cockroach_processed_md_ast_without_empty_all_Go.csv','cockroach-Go',4.379664179,661),
    # ('/root/autodl-tmp/Code-copy/Data/dotnet_aspnetcore_processed_md_ast_without_empty_all_C#.csv','dotnet_aspnetcore-C#',4.019057172,464), 
    # ('../Data/grpc.csv', 'grpc-C++', 4.36770428, 297), #514 2h
    # ('../Data/golang.csv', 'golang-Go', 3.549056604, 275), # 530 2h
    # ('../Data/pytorch.csv', 'pytorch-C++', 4.082221036, 1140), #1873 6h
    # ('../Data/dotnet_runtime.csv', 'dotnet_runtime-C#', 3.235663458, 1160), #4778 20h
    (['../Data/cockroach.csv', '../Data/dotnet_aspnetcore.csv',
     '../Data/dotnet_roslyn.csv', '../Data/dotnet_runtime.csv',
     '../Data/golang.csv', '../Data/grpc.csv',
     '../Data/pytorch.csv', '../Data/spring-framework.csv',
     '../Data/TensorFlow.csv',], 
     'exp2_combined_dataset', 4, 2600) # TODO: avg_label, total_label, len
]

lossList = [
    # (nn.BCEWithLogitsLoss(), 'BCE'),
    # ('SparceBCELoss', 'SBCE'),
    # (AsymmetricLossOptimized(), 'ASL'),
    (FocalLoss(), 'Focal'),
]
# TODO: Roberta, albert, coderbert
# TODO: XLnet
ckptList = [
    ('bert-base-uncased', 'Multi-triage', 20),  # just for tokenize
    # ('bert-base-uncased', ' Bert', 10),
    # ('roberta-base', 'Robert', 10),
    # ('albert-base-v2', 'albert', 10),
    # ('codebert-base', 'codebert', 10),
    # XLnet
]
# TODO: 4 other codeformat
codeFormatList = [
    # 'None',
    # 'Front',
    # 'Back',
    # 'Separate',
    'raw',
]

# result_col for each dataset
res_columns = ['train_method', 'loss_type', 'model_type', 'code_format',
               'proj_name', 'epoch',
               'f1_d', 'f1_b','acc@1_d', 'acc@1_b', 'acc@2_d', 'acc@2_b', 'acc@3_d', 'acc@3_b', 'acc@5_d', 'acc@5_b', 'acc@10_d', 'acc@10_b', 'acc@20_d', 'acc@20_b']

for path in pathList:
    # create 'res_DataFrame' for each dataset
    result = pd.DataFrame(columns = res_columns)
    for ckpt in ckptList:
        for loss in lossList:
            for code_format in codeFormatList:
                # try:
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
                                     _ckpt=ckpt[0], exp=EXP)
                    # save f1-score 'for each head(dev&btype)' 'for every 5 epoch'
                    for i, res in enumerate(ress):
                        print(res)
                        res_val = []
                        for k, v in res.items():
                            if 'F1' in k or '@' in k:
                                res_val += [v[0], v[1]]
                        result.loc[result.shape[0]] = ['NaiveTraining', loss[1], ckpt[1], code_format,
                                                       path[1], 5 * (i + 1)] + res_val
                # except Exception as e:
                #     print(e)
        # save 'res_DataFrame' for each dataset
        now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        result.to_csv(f'../res/{path[1]}_{now}_result.csv', index=False)
