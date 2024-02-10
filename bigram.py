import torch
import torch.nn as nn
from torch.nn import functional as F


# Hyperparameters.
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------

torch.manual_seed(1337)

# Loading text.
with open("input.txt", "r") as f:
    text = f.read()

# Creating the vocabulary.
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("".join(chars))
print(vocab_size)

# Make encoder and decoder.
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Encode the whole dataset.
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)


# Split train/val.
train_ratio = 0.9
n = int(len(data) * train_ratio)
train_data = data[:n]
val_data = data[n:]


# Data loader.
def get_batch(split):
    # generate a small batch of data of inputs x and targets y.
    data = train_data if split == "train" else val_data
    # Make sure that we can sample block_size character from the index. Reshape the output to :(batch_size,)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):

    def __init__(self, n_emdb):
        super().__init__()
        self.net = nn.Sequential(
            # Try to reproduce what is in the official paper for the linear layer. In the paper
            # feed-forward linear dimesion was 2048 and  n_embd was 512.
            # So feed-forward dimension = 4 * n_embd = 4 * 512 = 2048
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # No parameter, so we register it as a buffer.
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # C is the n_embd
        k = self.key(x)  # --> (B,T,C) # What I contain.
        q = self.query(x)  # --> (B,T,C) # What I'm I looking for ?

        wei = (
            q @ k.transpose(-2, -1) * C**0.5
        )  # (B,T,C) @ (B,C,T) --> (B,T,T) | Affinity matrix

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v  # If you find me interresting, here what I can give.
        return out


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()

        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x + are the residual connections.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Simple bigram model.
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)],
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # ix : (B,T).
        # targets : (B,T).

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C).
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # --> (T,C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            # Conserve only last block_size token.
            idx_bond = idx[:, -block_size:]

            # Get predictions
            logits, loss = self(idx_bond)
            # Focus on the last time step only.
            logits = logits[:, -1, :]  # --> (B,C).
            # Apply softmax to get probabilities.
            probs = F.softmax(logits, dim=-1)  # --> (B,C).
            # Sample from the distribution.
            idx_next = torch.multinomial(probs, num_samples=1)  # --> (B, 1)
            # Add to next character to the end of the sequence.
            idx = torch.cat((idx, idx_next), dim=1)  # --> (B, T+1)

        return idx


# Instantiate the model.
model = BigramLanguageModel()
m = model.to(device)

# Create the otimizer.
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training Loop.
for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4}"
        )
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
