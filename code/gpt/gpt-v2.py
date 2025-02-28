# v2: v1 + the following changes:
# - remove vocab_size in __init__
# - add token embedding 
# - add position embedding
# - add number of parameters
# - add weight and biases
# - add checkpoint

import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(100)

# Hyperparameters
batch_size = 32
block_size = 8
n_embedding = 32
max_iters = 5000
learning_rate = 1e-3
eval_every = 500
eval_iters = 200

model_name = "bigram-v2"
directory = "/scratch/users/glouppe/gpt"
checkpoint = f"{directory}/{model_name}.pt"

# WandB
wandb.init(
    project="nano-gpt",
    config={
        "model": model_name,
        "batch_size": batch_size,
        "block_size": block_size,
        "n_embedding": n_embedding,
        "max_iters": max_iters,
        "learning_rate": learning_rate,
        "eval_every": eval_every,
        "eval_iters": eval_iters,
    }
)

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
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embedding)
        self.position_embedding_table = nn.Embedding(block_size, n_embedding)
        self.lm_head = nn.Linear(n_embedding, vocab_size)
        
    def forward(self, x): # (B, T)
        B, T = x.shape

        tok_emb = self.token_embedding_table(x) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        h = tok_emb + pos_emb
        logits = self.lm_head(h) # (B, T, V)
        
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
            x_crop = x[:, -block_size:]
            logits = self(x_crop) # (B, T, V)
            logits = logits[:, -1, :] # (B, V)
            probs = F.softmax(logits, dim=1) # (B, V)
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            x = torch.cat([x, x_next], dim=1)
            
        return x  

lm = BigramLanguageModel()
lm = lm.to(device)
print("Parameters:", sum(p.numel() for p in lm.parameters())/1e3, 'K parameters')

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

for i in range(max_iters+1):
    if i % eval_every == 0:
        losses = evaluate(lm)
        print(f"step {i}: train loss {losses['train']:.4f}, validation loss {losses['validation']:.4f}")

        prompts = torch.tensor([encode(p) for p in ["Har", "Her", "Ron", "Dum", "Hag"]]).to(device)
        outputs = lm.generate(prompts, 100)
        outputs = [[decode(output.tolist())] for output in outputs]
        wandb.log({
            "step": i,
            "train loss": losses["train"],
            "validation loss": losses["validation"],
            "samples": wandb.Table(columns=["samples"], data=outputs)
        })

    optimizer.zero_grad()
    x, y = get_batch(train_data)
    loss = lm.loss(x, y)
    loss.backward()
    optimizer.step()

# Save model
torch.save(lm.state_dict(), checkpoint)