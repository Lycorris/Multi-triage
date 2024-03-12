from datasets import *
from keras.preprocessing.text import Tokenizer

train_path = 'Data/powershell/C_uA_Train.csv'
test_path = 'Data/powershell/C_uA_Test.csv'
MAX_SEQ_LEN = 300


train_dataset = Text_Code_Dataset(train_path, pad_seq_len=MAX_SEQ_LEN)
test_dataset = Text_Code_Dataset(test_path, pad_seq_len=MAX_SEQ_LEN)




