from transformers import AlbertTokenizer, AlbertModel, BertTokenizer, BertModel
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize input
    for i, sentence in enumerate(sentences):
        sentence_tokenized = tokenizer.tokenize(sentence)
        sentence_tokenized_id = tokenizer.convert_tokens_to_ids(sentence_tokenized)
        # print(sentence_tokenized_id)
        # print(sentences)
        sentences[i] = sentence_tokenized_id


    sentences_padded = pad_sequences(sentences, maxlen=max_seq_len, padding='post',value = 0)

    # Create attention mask
    attention_mask = [[float(i != 0) for i in seq] for seq in sentences_padded]

    sentences = torch.tensor(sentences_padded)
    attention_mask = torch.tensor(attention_mask)
    # print(attention_mask)
    sentences = sentences.to(device)

    # Load pre-trained model (weights)
    model = BertModel.from_pretrained('bert-base-uncased')
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(sentences)
        embedded_sentences = outputs[0]
        # pooled_output = outputs[1]

    return embedded_sentences


if __name__ == '__main__':
    mySentences = ["Hello, my dog is cute", "Hello, my cat"]
    myTokenizer = "Albert"
    myDevice = "cpu"
    print(get_word_embedding(mySentences, myTokenizer, myDevice).shape)
    # print(type(get_word_embedding(mySentences, myTokenizer, myDevice)))
    # torch.Size([2, 6, 768])
