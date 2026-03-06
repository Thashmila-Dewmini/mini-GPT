import torch
import torch.nn.functional as F
from model import MiniGPT

# Hyperparameters 
batch_size=32
block_size=256
max_iters=10000
eval_interval=500
learning_rate=3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Dataset
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
# set(text) -> unique characters
# list() -> convert set into list
# sorted() -> sort alphabetically
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)} # character into ID
# EX: 'a' -> 0, 'b' -> 1, 'c' -> 2
# stoi = string to integer
itos = {i: ch for i, ch in enumerate(chars)} # integer into character
# EX: 0 -> 'a', 1 -> 'b'
# itos = integer to string

# encoding function
# convert text into numbers
def encode(s):
    return [stoi[c] for c in s]
    # ex: "cat" becomes [2,0,19]

# decoding function
# convert numbers into text
def decode(l):
    return ''.join([itos[i] for i in l])
    # ex: [2,0,19] becomes "cat"

data = torch.tensor(encode(text), dtype=torch.long) # convert dataset to tensor
# text -> encode -> numbers -> tensor

# train / validation split
n = int(0.9 * len(data)) # 90% for training

train_data = data[:n] # first 90% for training 
val_data = data[n:]

# Batch Loader - create training batches
def get_batch(split):
    data_source = train_data if split == "train" else val_data  # select dataset
    ix = torch.randint(len(data_source) - block_size, (batch_size,)) # randomly pick batch_size starting points
    x = torch.stack([data_source[i:i + block_size] for i in ix]) # create input batch
    # for each starting index take 256 characters 
    # shape: (32,256)
    y = torch.stack([data_source[i + 1:i + block_size + 1] for i in ix]) # create target
    # targets are shifted by one token
    # ex: input=> hello worl, target=> ello world, model learns next character predication

    return x.to(device), y.to(device)

# Model
model = MiniGPT(
    vocab_size=vocab_size,
    d_model=256,
    num_head=8,
    d_ff=256,
    num_layers=6,
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # updates weights

# Training Loop
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits = model(xb)  # for every token position predict next character
    # shape: (32,256,vocab_size)

    # loss calculation
    # flatten tensors
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        yb.view(-1)
    )
    # from (32,256,vocab_size) to (8192,vocab_size) , target(8192)
    # computing cross entropy across all tokens

    optimizer.zero_grad() # clear previous gradients
    loss.backward() # compute gradient using backpropagation
    optimizer.step() # update model weights

    # Every 500 steps show training progress
    if iter % eval_interval == 0:
        print("step", iter, "loss", loss.item())  

# Text Generation
model.eval()  # switch to evaluation mode and disables dropout and training behaviors
context = torch.zeros((1, 1), dtype=torch.long, device=device) # start context
# initial token: [[0]], this is starting prompt
generated = model.generate(context, max_new_tokens=500)  # generate 500 characters
print("\nGenerated Text:\n")
print(decode(generated[0].tolist())) # decode tokens
# tensor -> list -> decode -> text

