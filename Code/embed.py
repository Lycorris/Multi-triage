from transformers import AlbertTokenizer, AlbertModel
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences


def get_word_embedding(sentences: list, tokenizer, device, max_seq_len=300):
    """
    Args:
        sentences: list of sentence, (n_sentence)
        tokenizer: the name of specified tokenizer model
        device: 'cuda' or 'cpu' or 'mps'
    Returns:
        embedded sentence, (n_sentence, emb_dim)
    """
    if tokenizer == "Albert":
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        model = AlbertModel.from_pretrained('albert-base-v2')
    elif tokenizer == "Bert":
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
    elif tokenizer == "RoBERTa":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')

    # tokenize, convert-to-id, pad
    for i, sentence in enumerate(sentences):
        sentence_tokenized = tokenizer.tokenize(sentence)
        sentence_tokenized_id = tokenizer.convert_tokens_to_ids(sentence_tokenized)
        sentences[i] = sentence_tokenized_id
    sentences_padded = pad_sequences(sentences, maxlen=max_seq_len, padding='post', value=0)
    # create attention mask
    attention_mask = [[float(i != 0) for i in seq] for seq in sentences_padded]
    sentences = torch.tensor(sentences_padded)
    # attention_mask = torch.where(sentences != 0, 1, 0)

    sentences.to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(sentences, attention_mask=attention_mask)
        embedded_sentences = outputs[0]

    return embedded_sentences


if __name__ == '__main__':
    mySentences = ["Hello, my dog is cute", "Hello, my cat"]
    myTokenizer = "Albert"
    myDevice = "cpu"
    print(get_word_embedding(mySentences, myTokenizer, myDevice))
    # torch.Size([2, 6, 768])
