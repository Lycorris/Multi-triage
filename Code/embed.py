from transformers import AlbertTokenizer, AlbertModel
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences
def get_word_embedding(sentences: list, tokenizer: str, device,max_seq_len=300):
    """
    Args:
        sentences: list of sentence, (n_sentence)
        tokenizer: the name of specified tokenizer model
        device: 'cuda' or 'cpu' or 'mps'
    Returns:
        embedded sentence, (n_sentence, emb_dim)
    """
    if tokenizer == "Albert":
        return albert_emb(sentences, device, max_seq_len)


def albert_emb(sentences, device, max_seq_len=300):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    # Tokenize input
    for i, sentence in enumerate(sentences):
        sentence_tokenized = tokenizer.tokenize(sentence)
        sentence_tokenized_id = tokenizer.convert_tokens_to_ids(sentence_tokenized)
        # print(sentence_tokenized_id)
        # print(sentences)
        sentences[i] = sentence_tokenized_id

    sentences_padded = pad_sequences(sentences, maxlen=max_seq_len, padding='post')
    sentences = torch.tensor(sentences_padded)
    sentences = sentences.to(device)

    # Load pre-trained model (weights)
    model = AlbertModel.from_pretrained('albert-base-v2')
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(sentences)
        embedded_sentences = outputs[0]
        # pooled_output = outputs[1]

    return embedded_sentences


if __name__ == '__main__':
    mySentences = ["Hello, my dog is cute", "Hello, my cat is cute"]
    myTokenizer = "Albert"
    myDevice = "cpu"
    print(get_word_embedding(mySentences, myTokenizer, myDevice).shape)
    # torch.Size([2, 6, 768])
