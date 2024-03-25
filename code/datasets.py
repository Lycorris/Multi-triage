import pandas as pd
import torch
from torch.utils.data import Dataset
from keras_preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import trange

from embed import *


class TextCodeDataset(Dataset):
    def __init__(self, data_path, pad_seq_len, use_AST=True, classify_btype=True, from_emb=False):
        self.data_path = data_path
        self.pad_seq_len = pad_seq_len
        self.use_AST, self.classify_btype = use_AST, classify_btype
        self.from_emb = from_emb
        self.data_pd = pd.read_csv(data_path, index_col=False, dtype='unicode', encoding='latin-1',
                                   low_memory=False).sample(frac=1)
        self.y_dev, self.y_btype = list(self.data_pd['FixedByID']), list(self.data_pd['Name'])
        self.y_dev, self.y_btype = [str(x).split('|') for x in self.y_dev], [str(x).split('|') for x in self.y_btype]

        self.x_context, self.x_AST = list(self.data_pd['Title_Description']), list(self.data_pd['AST'])
        self.x_context, self.x_AST = [str(x) for x in self.x_context], [str(x) for x in self.x_AST]
        self.x_context_emb, self.x_AST_emb = self.x_context, self.x_AST

    def get_embeded(self, tokenizer, device, max_seq_len):
        print('-' * 20 + 'embedding begin' + '-' * 20)
        emb_c, emb_a = torch.tensor([]).to(device), torch.tensor([]).to(device)

        for i in trange(0, len(self.x_context_emb), 64):
            batch_c = self.x_context_emb[i:i + 64]
            batch_a = self.x_AST_emb[i:i + 64]

            batch_emb_c = get_word_embedding(batch_c, tokenizer, device, max_seq_len)
            batch_emb_a = get_word_embedding(batch_a, tokenizer, device, max_seq_len)

            emb_c = torch.concat((emb_c, batch_emb_c))
            emb_a = torch.concat((emb_a, batch_emb_a))
            torch.cuda.empty_cache()

        self.x_context_emb, self.x_AST_emb = emb_c, emb_a
        # self.x_context_emb = get_word_embedding(self.x_context_emb, tokenizer, device,max_seq_len)
        # self.x_AST_emb = get_word_embedding(self.x_AST_emb, tokenizer, device,max_seq_len)
        print('-' * 20 + 'embedding over' + '-' * 20)

    def tokenize_input(self, tokenizer_C: Tokenizer, tokenizer_A: Tokenizer):
        self.x_context = tokenizer_C.texts_to_sequences(self.x_context)
        self.x_AST = tokenizer_A.texts_to_sequences(self.x_AST)
        self.x_context = pad_sequences(self.x_context, maxlen=self.pad_seq_len, padding='post')
        self.x_AST = pad_sequences(self.x_AST, maxlen=self.pad_seq_len, padding='post')
        self.x_context = torch.from_numpy(self.x_context)
        self.x_AST = torch.from_numpy(self.x_AST)

    def map_output(self, map_d: dict, map_b: dict):
        tensor_d, tensor_b = torch.zeros((len(self.data_pd), len(map_d))), torch.zeros((len(self.data_pd), len(map_b)))
        for i, ds in enumerate(self.y_dev):
            for d in ds:
                tensor_d[i][map_d[d]] = 1
        for i, bs in enumerate(self.y_btype):
            for b in bs:
                tensor_b[i][map_b[b]] = 1
        self.y_dev, self.y_btype = tensor_d, tensor_b

    def __getitem__(self, item):
        if not self.from_emb:
            input = self.x_context[item], self.x_AST[item] if self.use_AST else self.x_context[item]
        else:
            input = self.x_context_emb[item], self.x_AST_emb[item] if self.use_AST else self.x_context_emb[item]
        output = self.y_dev[item], self.y_btype[item] if self.classify_btype else self.y_dev[item]
        return input, output

    def __len__(self):
        return len(self.data_pd)


def tokenize_dataset_input(train_dataset: TextCodeDataset, test_dataset: TextCodeDataset):
    tokenizer_C = Tokenizer()
    tokenizer_C.fit_on_texts(train_dataset.x_context + test_dataset.x_context)
    tokenizer_A = Tokenizer()
    tokenizer_A.fit_on_texts(train_dataset.x_AST + test_dataset.x_AST)
    train_dataset.tokenize_input(tokenizer_C, tokenizer_A)
    test_dataset.tokenize_input(tokenizer_C, tokenizer_A)
    return [len(tokenizer_C.word_index) + 100, len(tokenizer_A.word_index) + 100]


def map_dataset_output(train_dataset: TextCodeDataset, test_dataset: TextCodeDataset):
    set_d, set_b = set(), set()
    for i in range(len(train_dataset)):
        for d in train_dataset.y_dev:
            set_d.update(d)
        for b in train_dataset.y_btype:
            set_b.update(b)
    for i in range(len(test_dataset)):
        for d in test_dataset.y_dev:
            set_d.update(d)
        for b in test_dataset.y_btype:
            set_b.update(b)
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

    train_dataset = TextCodeDataset(train_path, pad_seq_len=MAX_SEQ_LEN)
    test_dataset = TextCodeDataset(test_path, pad_seq_len=MAX_SEQ_LEN)

    vocab_size = tokenize_dataset_input(train_dataset, test_dataset)
    idx2label = map_dataset_output(train_dataset, test_dataset)
    num_out = [len(x) for x in idx2label]

    # print(vocab_size)  # [12513, 5910]
    # print(num_out)  # [396, 204]
    print(len(train_dataset[0]))  # 2 表示输入和输出
    print(len(train_dataset[0][0]))  # 2 表示x_context和x_AST
    print(train_dataset[0][0])  # 两个tensor 分别对应x_context和x_AST
    print(train_dataset[0][1])  # 两个tensor 分别对应y_dev和y_btype
    # 检查x_context的形状
    print(train_dataset[0][0][0].shape)  # torch.Size([300])
    # 检查x_AST的形状
    print(train_dataset[0][0][1].shape)  # torch.Size([300])
    # 检查y_dev的形状
    print(train_dataset[0][1][0].shape)  # torch.Size([396])
    # 检查y_btype的形状
    print(train_dataset[0][1][1].shape)  # torch.Size([204])
    # train_dataset[0]
    # ((tensor([[100, 64, 383, ..., 0, 0, 0],
    #           [2035, 606, 105, ..., 0, 0, 0],
    #           [8, 3395, 174, ..., 0, 0, 0],
    #           ...,
    #           [110, 508, 43, ..., 0, 0, 0],
    #           [230, 14, 69, ..., 0, 0, 0],
    #           [23, 3739, 131, ..., 0, 0, 0]], dtype=torch.int32),
    #   tensor([[15, 0, 0, ..., 0, 0, 0],
    #           [15, 0, 0, ..., 0, 0, 0],
    #           [15, 0, 0, ..., 0, 0, 0],
    #           ...,
    #           [15, 0, 0, ..., 0, 0, 0],
    #           [15, 0, 0, ..., 0, 0, 0],
    #           [15, 0, 0, ..., 0, 0, 0]], dtype=torch.int32)),
    #  (tensor([[0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           ...,
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.]]),
    #   tensor([[0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           ...,
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.],
    #           [0., 0., 0., ..., 0., 0., 0.]])))
