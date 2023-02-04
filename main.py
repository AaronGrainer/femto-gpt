import typer
import config
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


app = typer.Typer()

# Set seed for reproducibility
torch.manual_seed(config.SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Load text
with open(Path(config.DATASET_DIR, "input.txt")) as f:
    text = f.read()

# Encode text
tokens = sorted(list(set(text)))
text_to_encode = {v: i for i, v in enumerate(tokens)}
encode_to_text = {i: v for i, v in enumerate(tokens)}


def encoder(text):
    # Encode text to integers
    return [text_to_encode[i] for i in text]


def decoder(encoded_text):
    # Decode integers to text
    return "".join([encode_to_text[i] for i in encoded_text])


def get_batch(input_text):
    # Randomly select a starting point and retrieve a batch of x and y text
    start = torch.randint(len(input_text) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([input_text[i : i + config.BLOCK_SIZE] for i in start])
    y = torch.stack([input_text[i + 1 : i + config.BLOCK_SIZE + 1] for i in start])
    x, y = x.to(DEVICE), y.to(DEVICE)
    
    return x, y


@torch.no_grad()
def estimate_loss(train_text, test_text, model):
    # Estimate loss on a batch of x and y text
    model.eval()

    losses = {"train": torch.zeros(config.EVAL_ITER), "test": torch.zeros(config.EVAL_ITER)}
    for i in tqdm(range(config.EVAL_ITER)):
        train_x, train_y = get_batch(train_text)
        _, train_loss = model(train_x, train_y)
        
        test_x, test_y = get_batch(test_text)
        _, test_loss = model(test_x, test_y)

        losses["train"][i] = train_loss.item()
        losses["test"][i] = test_loss.item()

    print(f"Training Loss: {torch.mean(losses['train']):.4f}, Validation Loss: {torch.mean(losses['test']):.4f}")
    model.train()


class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Linear(config.N_EMBED, config.HEAD_SIZE, bias=False)
        self.query = nn.Linear(config.N_EMBED, config.HEAD_SIZE, bias=False)
        self.value = nn.Linear(config.N_EMBED, config.HEAD_SIZE, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE)))

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) / (C ** 0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        bow = wei @ v

        return bow


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList([Head() for _ in range(config.N_HEAD)])
        self.linear = nn.Linear(config.N_EMBED, config.N_EMBED)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.linear(out))

        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            *[
                nn.Linear(config.N_EMBED, config.N_EMBED * 4),
                nn.ReLU(),
                nn.Linear(config.N_EMBED * 4, config.N_EMBED),
                nn.Dropout(config.DROPOUT)
            ]
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()

        self.heads = MultiHeadAttention()
        self.linear = FeedForward()
        self.layer_norm1 = nn.LayerNorm(config.N_EMBED)
        self.layer_norm2 = nn.LayerNorm(config.N_EMBED)

    def forward(self, x):
        x = x + self.heads(self.layer_norm1(x))
        x = x + self.linear(self.layer_norm2(x))

        return x


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embed = nn.Embedding(len(tokens), config.N_EMBED)
        self.pos_embed = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.blocks = nn.Sequential(*[Block() for _ in range(config.N_LAYER)])
        self.layer_norm = nn.LayerNorm(config.N_EMBED)
        self.lm_head = nn.Linear(config.N_EMBED, len(tokens))

    def forward(self, x, target=None):
        B, T = x.shape

        logits = self.token_embed(x)
        logits += self.pos_embed(torch.arange(T, device=DEVICE))
        logits = self.blocks(logits)
        logits = self.layer_norm(logits)
        logits = self.lm_head(logits)

        if target is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_tokens=100):
        for _ in tqdm(range(max_tokens)):
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx


@app.command()
def train():
    # Prepare dataset
    split = int(len(text) * config.TRAIN_TEST_SPLIT)
    train_text = torch.tensor(encoder(text[:split]), dtype=torch.long)
    test_text = torch.tensor(encoder(text[split:]), dtype=torch.long)

    # Initialize model
    model = BigramModel()
    model = model.to(DEVICE)

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
  
    # Train model
    for epoch in tqdm(range(config.EPOCHS + 1)):
        x, y = get_batch(train_text)

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % config.EVAL_INTERVAL == 0:
            estimate_loss(train_text, test_text, model)

    # Save model
    torch.save(model.state_dict(), Path(config.CHECKPOINT_DIR, "model.pt"))


@app.command()
def generate():
    # Load model
    model = BigramModel()
    model.load_state_dict(torch.load(Path(config.CHECKPOINT_DIR, "model.pt")))
    model = model.to(DEVICE)

    # Generate text
    start_idx = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    generated_response = decoder(model.generate(start_idx, max_tokens=500)[0].tolist())
    print('generated_response: ', generated_response)


@app.command()
def tril():
    B, T, C = config.BATCH_SIZE, config.BLOCK_SIZE, config.CHANNEL_SIZE

    x = torch.rand((B, T, C))

    key = nn.Linear(C, config.HEAD_SIZE, bias=False)
    query = nn.Linear(C, config.HEAD_SIZE, bias=False)
    value = nn.Linear(C, config.HEAD_SIZE, bias=False)

    k = key(x)
    q = query(x)

    wei = q @ k.transpose(-2, -1) / (config.HEAD_SIZE ** 0.5)

    tril = torch.tril(torch.ones((T, T), dtype=torch.long))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    
    v = value(x)
    bow = wei @ v

    print('tril: ', tril)
    print('wei: ', wei)
    print('bow: ', bow)
    


if __name__ == "__main__":
    app()

