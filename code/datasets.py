import pandas as pd
import torch
from torch.utils.data import Dataset
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Text_Code_Dataset(Dataset):
    def __init__(self, data_path, pad_seq_len, use_AST=True, classify_btype=True):
        self.data_path = data_path
        self.pad_seq_len = pad_seq_len
        self.use_AST, self.classify_btype = use_AST, classify_btype
        # 读取Datafile, 返回删除格式错误的行，不用index列(不知为啥)，dtype=unicode，编码方式latin-1，low-memory避免数据类型不同的问题
        # sample(frac = 1)打乱取全部(不懂，后面都按CreateDate排序了)
        self.data_pd = pd.read_csv(data_path,
                                   error_bad_lines=False, index_col=False, dtype='unicode', encoding='latin-1',
                                   low_memory=False).sample(frac=1)
        self.y_dev, self.y_btype = list(self.data_pd['FixedByID'].split('|')), list(self.data_pd['Name'].split('|'))
        self.x_context, self.x_AST = list(self.data_pd['Title_Description']), list(self.data_pd['AST'])
        self.x_context, self.x_AST = [str(row).lower() for row in self.x_context], [str(row).lower() for row in self.x_AST]


    def tokenize_input(self, tokenizer_C: Tokenizer, tokenizer_A: Tokenizer):
        self.x_context = tokenizer_C.texts_to_sequences(self.x_context)
        self.x_AST = tokenizer_A.texts_to_sequences(self.x_AST)
        self.x_context = pad_sequences(self.x_context, maxlen=self.pad_seq_len, padding='post')
        self.x_AST = pad_sequences(self.x_AST, maxlen=self.pad_seq_len, padding='post')
        self.x_context = torch.from_numpy(self.x_context)
        self.x_AST = torch.from_numpy(self.x_AST)

    def map_output(self, map_d: dict, map_b: dict):
        tensor_d, tensor_b = torch.zeros((len(self.data_pd), len(map_d))), torch.zeros((len(self.data_pd), len(map_b)))
        # for i, d in enumerate(self.y_dev):


    def __getitem__(self, item):
        input = self.x_context, self.x_AST if self.use_AST else self.x_context
        output = self.y_dev, self.y_btype if self.classify_btype else self.y_dev
        return input, output

    def __len__(self):
        return len(self.data_pd)


def tokenize_dataset_input(train_dataset: Text_Code_Dataset, test_dataset: Text_Code_Dataset):
    tokenizer_C = Tokenizer(num_words=None, char_level=None, oov_token='Unknown',
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer_C.fit_on_texts(train_dataset.x_context + test_dataset.x_context)
    tokenizer_A = Tokenizer(num_words=None, char_level=None, oov_token='Unknown',
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer_A.fit_on_texts(train_dataset.x_AST + test_dataset.x_AST)

    train_dataset.tokenize_input(tokenizer_C, tokenizer_A)
    test_dataset.tokenize_input(tokenizer_C, tokenizer_A)

def map_dataset_output(train_dataset: Text_Code_Dataset, test_dataset: Text_Code_Dataset):
    set_d, set_b = set(), set()
    for i in range(len(test_dataset)):
        set_d.update(train_dataset.y_dev)
        set_b.update(train_dataset.y_btype)
    labels_d, labels_b = list(set_d), list(set_b)
    labels_d.sort(), labels_b.sort()
    map_d, map_b = {labels_d[i]: i for i in range(len(labels_d))}, {labels_b[i]: i for i in range(len(labels_b))}
    train_dataset.map_output(map_d, map_b)
    test_dataset.map_output(map_d, map_b)
    return labels_d, labels_b

if __name__ == '__main__':
    train_path = 'Data/powershell/C_uA_Train.csv'
    test_path = 'Data/powershell/C_uA_Test.csv'
    MAX_SEQ_LEN = 300

    train_dataset = Text_Code_Dataset(train_path, pad_seq_len=MAX_SEQ_LEN)
    test_dataset = Text_Code_Dataset(test_path, pad_seq_len=MAX_SEQ_LEN)

