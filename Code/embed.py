from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import torch


def get_tokenizer_models(pretrained_name, device):
    tokenizer, model = None, None
    # TODO: use AutoTokenizer.from_pretrained('path')
    if pretrained_name == "Albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2').to(device)
    elif pretrained_name == "Bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)
    elif pretrained_name == "Bert-L":
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        model = BertModel.from_pretrained('bert-large-uncased').to(device)
    elif pretrained_name == "RoBerta":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base').to(device)
    return tokenizer, model


def get_sents_token_emb(sentences: list, pretrained_name: str, device, max_seq_len=300, mode: str = "token"):
    """
    Args:
        mode: str, "token" or "emb"
        max_seq_len: int, padding len
        pretrained_name: str, the name of specified tokenizer model
        sentences: list of sentence, (n_sentence)
        device: 'cuda' or 'cpu' or 'mps'
    Returns:
        embedded sentence, (n_sentence, emb_dim)
    """
    tokenizer, model = get_tokenizer_models(pretrained_name, device)

    # tokenize & to_ids & pad
    tokenized_sents = tokenizer(sentences, padding='max_length',
                                truncation=True, max_length=max_seq_len, return_tensors="pt").to(device)
    if mode == "token":
        return tokenized_sents

    embedded_sentences = model(tokenized_sents['input_ids'], tokenized_sents['attention_mask'])[0]
    if mode == "emb":
        return embedded_sentences


if __name__ == '__main__':
    mySentences = ["Hello, my dog is cute", "Hello, my cat"]
    myTokenizer = "Albert"
    myDevice = 'cuda' if torch.cuda.is_available() else 'cpu'
    myMode = 'emb'
    # myDevice = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(get_sents_token_emb(mySentences, myTokenizer, myDevice, mode=myMode).shape)
    # Tokenized Inputs:
    # {
    #     input_ids:
    #         tensor([[    0,   100,   437,  2283,     7,  1532,    59, 30581,  3923, 12346,
    #          34379,   328,     2]])
    #     attention_mask:
    #         tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # }
    # torch.Size([2, 300, 768])
