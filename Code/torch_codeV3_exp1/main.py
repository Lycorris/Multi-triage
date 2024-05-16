# %% [markdown]
# 1. early-stop --> epoch
# 2. hyper-prarm with list
# - datasets * 8
# - model * 3
# - loss * 3
# - use_ast
#   
# 3. other hyper-prarm
# - lr = 3e-5
# - bsz = 8 (24g4090 + 2Bert)
# 
# 4. TBC
# - customized loss
# - ast ckpt
import pandas as pd

from transformers import AdamW, get_scheduler
from tqdm import trange, tqdm

from dataset import *
from model import *
from losses import *
from metrics import *
from train import *
import datetime
# 所有实验结果的表result
# result = pd.DataFrame(columns=['loss_type','proj_name', 'proj_lang', 'model_type', 'code', 'res1', 'res2', 'class1', 'class2','data_len','AST_method'])
result = pd.DataFrame(columns=['train_method','loss_type', 'model_type', 'code_format', 'proj_name', 'epoch','f1_d','f1_b','acc@1_d','acc@2_d','acc@3_d','acc@5_d','acc@10_d','acc@20_d','acc@1_b','acc@2_b','acc@3_b','acc@5_b','acc@10_b','acc@20_b'])
# 该全局变量在主循环中获取配置信息
logs=[]

'''
    _code_format = 'None'  -> ignore Code Snippet
                   'Front' -> add Code BEFORE Text
                   'Back'  -> add Code BEHIND Text
                   'Separate' -> consider Code as an independent input
                   'raw'   -> add the original Code Snippet
    _model_type =  'Multi-triage'
                   'PreTrain'
'''

# 8 datasets
# path, name, dataset size, avg_label_types, total_label_types
pathList = [
    # ('/root/autodl-tmp/Code-copy/Data/spring-framework_processed_md_ast_without_empty_all_Java.csv','spring_framework-Java',3.1979,57),
    # this is included in original datasets ('/root/autodl-tmp/Code-copy/Data/dotnet_roslyn_processed_md_ast_without_empty_all_C#.csv','dotnet_roslyn-C#',3.961,454),
    # ('/root/autodl-tmp/Code-copy/Data/TensorFlow_processed_md_ast_without_empty_all_C++.csv','TensorFlow-C++',4.994142627,735),
    # ('/root/autodl-tmp/Code-copy/Data/cockroach_processed_md_ast_without_empty_all_Go.csv','cockroach-Go',4.379664179,661),
    # ('/root/autodl-tmp/Code-copy/Data/dotnet_aspnetcore_processed_md_ast_without_empty_all_C#.csv','dotnet_aspnetcore-C#',4.019057172,464),
    # ('/root/autodl-tmp/Code-copy/Data/dotnet_runtime_processed_md_ast_without_empty_all_C#.csv','dotnet_runtime-C#',3.235663458,1160),
    # ('/root/autodl-tmp/Code-copy/Data/grpc_processed_md_ast_without_empty_all_C++.csv','grpc-C++',4.36770428,297),
    # ('/root/autodl-tmp/Code-copy/Data/pytorch_processed_md_ast_without_empty_all_C++.csv','pytorch-C++',4.082221036,1140),
    # ('/root/autodl-tmp/Code-copy/Data/golang_processed_md_ast_without_empty_all_Go.csv','golang-Go',3.549056604,275),
    ('/root/autodl-tmp/Code-copy/Data_5_11/efcore_3.csv','efcore-C#',3.4162873,111),
    ('/root/autodl-tmp/Code-copy/Data_5_11/elasticSearch_3.csv','elasticSearch-Java',3.5669619,406),
    ('/root/autodl-tmp/Code-copy/Data_5_11/mixedRealityToolUnity_3.csv','mixedRealityToolUnity-C#',3.641061,216),
    ('/root/autodl-tmp/Code-copy/Data_5_11/powerShell_3.csv','powershell-C++',3.238929,405),
    ('/root/autodl-tmp/Code-copy/Data_5_11/roslyn_3.csv','roslyn-C#',6.629381,282),
    ('/root/autodl-tmp/Code-copy/Data_5_11/realmJava_3.csv','realmjava-Java',2.4358974,50),
    ('/root/autodl-tmp/Code-copy/Data_5_11/nunit_3.csv','nunit-C#',3.666667,41),
]
lossList = [
    (nn.BCEWithLogitsLoss(), 'BCE'),
    ('SparceBCELoss', 'SBCE'),
    (AsymmetricLossOptimized(), 'ASL'),
    (FocalLoss(), 'Focal'),
]

ckptList = [
    ('../bert-base-uncased', 'Multi-triage',20),  # just for tokenize
    ('../bert-base-uncased', ' Bert',10),
    ('../roberta-base', 'Robert',10),
    ('../albert-base-v2', 'albert',10),
    ('../codebert-base', 'codebert',10),
]

codeFormatList = [
    'None',
    'Front',
    'Back',
    'Separate',
    'raw',
]

for path in pathList:
    for ckpt in ckptList:
        for loss in lossList:
            for code_format in codeFormatList:
                try:
                # 赋值loss_fn
                    print(f"avg_label_types={path[2]}, total_label_types={path[3]}")
                    loss_fn = loss[0] if loss[1] != 'SBCE' else SparceBCELoss(avg_label_types=path[2],total_label_types=path[3])
                    model_type = 'Multi-triage' if ckpt[1] == 'Multi-triage' else 'PreTrain'
                    logname = ' '.join([path[1], ckpt[1], loss[1], code_format, str(ckpt[2])])
                    print('-'*100, logname, '-'*100, sep='\n')
                    ress,metric = train_imm(_path = path[0], _logname = logname, 
                              _num_epochs = ckpt[2],
                          _loss_fn = loss_fn, _code_format = code_format, _model_type = model_type, 
                          _ckpt = ckpt[0])
                    print(metric)
                    # 保存结果
                    for i,res in enumerate(ress):
                        result.loc[result.shape[0]] = ['NaiveTraining',loss[1],ckpt[1],code_format,path[1],5*(i+1),res[0],res[1],metric['acc@1_d'],metric['acc@2_d'],metric['acc@3_d'],metric['acc@5_d'],metric['acc@10_d'],metric['acc@20_d'],metric['acc@1_b'],metric['acc@2_b'],metric['acc@3_b'],metric['acc@5_b'],metric['acc@10_b'],metric['acc@20_b']]
                        # 跑到了spring_framework-Java Robert BCE Separate 10
                        # torch.cuda.empty_cache()
                except Exception as e:
                    print(e)
                    continue
    # TODO:保存日志
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    result.to_csv(f'{path[1]}_{now}_result.csv',index=False)
    # 清空result
    result = pd.DataFrame(columns=['train_method','loss_type', 'model_type', 'code_format', 'proj_name', 'epoch','f1_d','f1_b','acc@1_d','acc@2_d','acc@3_d','acc@5_d','acc@10_d','acc@20_d','acc@1_b','acc@2_b','acc@3_b','acc@5_b','acc@10_b','acc@20_b'])