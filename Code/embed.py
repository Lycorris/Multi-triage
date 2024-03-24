from transformers import AlbertTokenizer, AlbertModel
import torch


def get_word_embedding(sentences: list, tokenizer: str, device):
    """
    Args:
        sentences: list of sentence, (n_sentence)
        tokenizer: the name of specified tokenizer model
        device: 'cuda' or 'cpu' or 'mps'
    Returns:
        embedded sentence, (n_sentence, emb_dim)
    """
    if tokenizer == "Albert":
        return albert_emb(sentences, device)


def albert_emb(sentences, device):
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    # Tokenize input
    for i, sentence in enumerate(sentences):
        sentence_tokenized = tokenizer.tokenize(sentence)
        sentence_tokenized_id = tokenizer.convert_tokens_to_ids(sentence_tokenized)
        sentences[i] = sentence_tokenized_id

    sentences = torch.tensor(sentences)
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
    myDevice = "mps"
    print(get_word_embedding(mySentences, myTokenizer, myDevice).shape)
    # torch.Size([2, 6, 768])
