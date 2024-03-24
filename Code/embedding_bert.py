from transformers import AlbertTokenizer, AlbertModel
from torch import torch
# Load pre-trained model tokenizer (vocabulary)
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# Tokenize input
text = "Hello, my dog is cute"
tokenized_text = tokenizer.tokenize(text)

print(tokenized_text)

# Convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# Load pre-trained model (weights)
model = AlbertModel.from_pretrained('albert-base-v2')

# Set the model in evaluation mode to deactivate the DropOut modules
model.eval()

# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cpu')
model.to('cpu')

# Predict hidden states features for each layer
with torch.no_grad():
    outputs = model(tokens_tensor)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
    pooler_output = outputs[1]

# show output
print(len(outputs))
print(encoded_layers.shape)
print(pooler_output.shape)