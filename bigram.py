import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# Hyperparameters.
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
# --------------

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


# Simple bigram model.
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # ix : (B,T).
        # targets : (B,T).

        logits = self.token_embedding_table(idx)  # (B,T,C).

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
            # Get predictions
            logits, loss = self(idx)
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
model = BigramLanguageModel(vocab_size)
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
