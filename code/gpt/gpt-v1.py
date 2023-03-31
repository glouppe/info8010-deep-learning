# v1: Simple bigram language model

import torch
import torch.nn as nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(100)

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
learning_rate = 1e-3 
eval_every = 250
eval_iters = 200

# Data
with open("harry-potter.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text))
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]

# Data loader
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Bigram language model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, x): # (B, T)
        logits = self.token_embedding_table(x) # (B, T, V)
        return logits

    def loss(self, x, targets):
        logits = self(x) 
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return loss
    
    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(x) # (B, T, V)
            logits = logits[:, -1, :] # (B, V)
            probs = F.softmax(logits, dim=1) # (B, V)
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat([x, x_next], dim=1)
            
        return x  

lm = BigramLanguageModel(vocab_size)
lm = lm.to(device)

# Training loop
@torch.no_grad()
def evaluate(lm):
    out =  {}
    lm.eval()

    for split, data in [("train", train_data), ("validation", validation_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data)
            loss = lm.loss(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    lm.train()
    return out

optimizer = torch.optim.Adam(lm.parameters(), lr=learning_rate)

for i in range(max_iters):
    if i % eval_every == 0:
        losses = evaluate(lm)
        print(f"step {i}: train loss {losses['train']:.4f}, validation loss {losses['validation']:.4f}")

    optimizer.zero_grad()
    x, y = get_batch(train_data)
    loss = lm.loss(x, y)
    loss.backward()
    optimizer.step()

# Generate text
x = torch.tensor([[stoi["H"]]]).to(device)
y = lm.generate(x, 100)
print(decode(y[0].tolist()))