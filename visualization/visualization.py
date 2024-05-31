import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import numpy as np

def loss_plot(dataset, model_type, epoch, code_format):
    # 设置图标名字
    plt.title(f'res_avg of different loss_type({model_type}, {code_format}, {epoch})')
    # 横坐标为proj_name，纵坐标为res_avg，不同的loss_type用不同的线条表示
    dataset = dataset[dataset['model_type'] == model_type]
    dataset = dataset[dataset['code_format'] == code_format]
    dataset = dataset[dataset['epoch'] == epoch]
    print(dataset.head())

    for loss_type in dataset['loss_type'].unique():
        sorted_dataset = dataset[dataset['loss_type'] == loss_type].sort_values('proj_name')
        plt.plot(sorted_dataset['proj_name'], sorted_dataset['res_avg'], label=loss_type)
    plt.legend()
    plt.show()

def model_plot(dataset, loss_type='ASL', code_format='None', epoch=10):
    # 设置图标名字
    plt.title(f'res_avg of different model_type({loss_type}, {code_format}, {epoch})')
    # 横坐标为proj_name，纵坐标为res_avg，不同的model_type用不同的线条表示
    dataset = dataset[dataset['loss_type'] == loss_type]
    dataset = dataset[dataset['code_format'] == code_format]
    dataset = dataset[dataset['epoch'] == epoch]

    for model_type in dataset['model_type'].unique():
        sorted_dataset = dataset[dataset['model_type'] == model_type].sort_values('proj_name')
        plt.plot(sorted_dataset['proj_name'], sorted_dataset['res_avg'], label=model_type)
    plt.legend()
    plt.show()

def code_format_plot(dataset, loss_type='ASL', model_type='codebert', epoch=10):
    # 设置图标名字
    plt.title(f'res_avg of different code_format({loss_type}, {model_type}, {epoch})')
    # 横坐标为proj_name，纵坐标为res_avg，不同的code_format用不同的线条表示
    dataset = dataset[dataset['loss_type'] == loss_type]
    dataset = dataset[dataset['model_type'] == model_type]
    dataset = dataset[dataset['epoch'] == epoch]

    for code_format in dataset['code_format'].unique():
        sorted_dataset = dataset[dataset['code_format'] == code_format].sort_values('proj_name')
        plt.plot(sorted_dataset['proj_name'], sorted_dataset['res_avg'], label=code_format)
    plt.legend()
    plt.show()


def train_method_plot(dataset, loss_type='SASL', model_type='Robert', code_format='Raw'):
    # 设置图标名字
    plt.title(f'res_avg of different train method({loss_type}, {model_type}, {code_format})')
    # 横坐标为proj_name，纵坐标为res_avg，不同的epoch用不同的线条表示
    dataset = dataset[dataset['loss_type'] == loss_type]
    dataset = dataset[dataset['model_type'] == model_type]
    dataset = dataset[dataset['code_format'] == code_format]
    dataset = dataset[dataset['epoch'] == 10]
    dataset = dataset[dataset['proj_name'] != 'avg']

    for epoch in dataset['train_method'].unique():
        sorted_dataset = dataset[dataset['train_method'] == epoch].sort_values('proj_name')
        plt.plot(sorted_dataset['proj_name'], sorted_dataset['res_avg'], label=epoch)
    plt.yticks(np.arange(min(sorted_dataset['res_avg']), max(sorted_dataset['res_avg'])+1, 0.05))
    plt.legend()
    plt.show()


def wisconsin_test_and_cliffs_delta(df1, df2 , col1, col2):
    # 威斯康星符号检验
    u, p = mannwhitneyu(df1[col1], df2[col2])
    
    # 计算Cliff's Delta
    # d = np.abs(np.mean(df1[col1].reset_index(drop=True) > df2[col2].reset_index(drop=True)) - np.mean(df1[col1].reset_index(drop=True) < df2[col2].reset_index(drop=True)))
    
    return u, p

# 横坐标为proj_name , 纵坐标为res_avg_avg，这里的res_avg_avg是指的对于每个proj_name，res_avg的平均值
def plot_avg_avg(dataset,x_name):
    print(dataset['proj_name'].unique())
    # 打印loss_type=SASL的res_avg列的平均值
    data = dataset.groupby(x_name)['res_avg'].mean()
    print(data)
    # 以data第一列为横坐标，第二列为纵坐标
    plt.plot(data.index, data)
    plt.title(f'res_avg_avg of different {x_name}')
    plt.show()


if __name__ == '__main__':
    # 读取数据
    csv_name = 'concated_res.csv'
    dataset = pd.read_csv(csv_name)

    # 去除proj_name某些值
    # dataset = dataset[dataset['proj_name'] != 'grpc-C++']
    # dataset = dataset[dataset['proj_name'] != 'pytorch-C++']
    # dataset = dataset[dataset['proj_name'] != 'dotnet_runtime-C#']
    # dataset = dataset[dataset['proj_name'] != 'dotnet_aspnetcore-C#']

    # 获取proj_name列的所有可能取值
    proj_name = dataset['proj_name'].unique()

    # 获取res1, res2列的平均值，单独算出一列
    dataset['res_avg'] = dataset[['res1', 'res2']].mean(axis=1)
    # 将avg列重命名为res_avg
    # dataset.rename(columns={'avg': 'res_avg'}, inplace=True)

    loss_plot(dataset, model_type='Robert', epoch=10, code_format='Raw')
    model_plot(dataset, loss_type='SASL', epoch=10, code_format='Raw')
    code_format_plot(dataset, loss_type='SASL', model_type='Robert', epoch=10)
    train_method_plot(dataset, loss_type='SASL', model_type='Robert', code_format='Raw')

    # 检验模型
    model_types = dataset['model_type'].unique()

    for model_type1 in model_types:
        for model_type2 in model_types:
            if model_type1 == model_type2:
                    continue
            u, p = wisconsin_test_and_cliffs_delta(dataset[dataset['model_type'] == model_type1], dataset[dataset['model_type'] == model_type2], 'res_avg', 'res_avg')
            print(f'{model_type1} vs {model_type2}: u={u}, p={p}')

    # 检验loss
    loss_types = dataset['loss_type'].unique()
    
    for loss_type1 in loss_types:
        for loss_type2 in loss_types:
            if loss_type1 == loss_type2:
                    continue
            u, p = wisconsin_test_and_cliffs_delta(dataset[dataset['loss_type'] == loss_type1], dataset[dataset['loss_type'] == loss_type2], 'res_avg', 'res_avg')
            print(f'{loss_type1} vs {loss_type2}: u={u}, p={p}')

    # 检验code_format
    # code_formats = dataset['code_format'].unique()

    # for code_format1 in code_formats:
    #     for code_format2 in code_formats:
    #         if code_format1 == code_format2:
    #                 continue
    #         u, p = wisconsin_test_and_cliffs_delta(dataset[dataset['code_format'] == code_format1], dataset[dataset['code_format'] == code_format2], 'res_avg', 'res_avg')
    #         print(f'{code_format1} vs {code_format2}: u={u}, p={p}')

    # 横坐标为_ , 纵坐标为res_avg_avg，这里的res_avg_avg是指的对于每个proj_name，res_avg的平均值
    plot_avg_avg(dataset, 'loss_type')
    plot_avg_avg(dataset, 'model_type')
    plot_avg_avg(dataset, 'code_format')

    # 确定提出模型： robera, SBCE, Back





