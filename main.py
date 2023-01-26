from pathlib import Path

import torch
import torch.nn as nn
import typer
from torch.nn import functional as F
from tqdm import tqdm

import config
from config import logger

app = typer.Typer()


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Read input dataset
with open(Path(config.DATASET_DIR, "input.txt"), encoding="utf-8") as f:
    input_text = f.read()

encoding = sorted(list({x for x in input_text}))
text_encode_mapping = {value: index for index, value in enumerate(encoding)}
encode_text_mapping = {index: value for index, value in enumerate(encoding)}


def encode(text):
    return [text_encode_mapping[x] for x in text]


def decode(text):
    return "".join([encode_text_mapping[x] for x in text])


def get_batch(data):
    ix = torch.randint(len(data) - config.BLOCK_SIZE, (config.BATCH_SIZE,))
    x = torch.stack([data[i : i + config.BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + config.BLOCK_SIZE + 1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_set, test_set):
    model.eval()
    out = {}

    for mode, data in zip(["train", "test"], [train_set, test_set]):
        losses = torch.zeros(config.EVAL_ITER)
        for i in range(config.EVAL_ITER):
            x, y = get_batch(test_set)
            logits, loss = model(x, y)
            losses[i] = loss.item()
        out[mode] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.key = nn.Linear(config.N_EMBED, config.HEAD_SIZE, bias=False)
        self.query = nn.Linear(config.N_EMBED, config.HEAD_SIZE, bias=False)
        self.value = nn.Linear(config.N_EMBED, config.HEAD_SIZE, bias=False)

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.BLOCK_SIZE, config.BLOCK_SIZE))
        )

        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) * C**-0.5
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
        out = torch.cat([h(x) for h in self.heads], dim=-1)
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
                nn.Dropout(config.DROPOUT),
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
        x = x + self.linear(self.layer_norm1(x))

        return x


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, config.N_EMBED)
        self.token_pos_table = nn.Embedding(config.BLOCK_SIZE, config.N_EMBED)
        self.blocks = nn.Sequential(*[Block() for _ in range(config.N_LAYER)])
        self.layer_norm = nn.LayerNorm(config.N_EMBED)
        self.lm_head = nn.Linear(config.N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_embed = self.token_embedding_table(idx)
        pos_embed = self.token_pos_table(torch.arange(T, device=DEVICE))
        x = token_embed + pos_embed
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in tqdm(range(max_new_tokens)):
            idx_cond = idx[:, -config.BLOCK_SIZE :]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@app.command()
def train_bigram():
    # Prepare dataset
    data = torch.tensor(encode(input_text), dtype=torch.long)
    n = int(len(data) * config.TRAIN_TEST_SPLIT)
    train_set = data[:n]
    test_set = data[n:]

    # Train model
    torch.manual_seed(config.SEED)

    model = BigramLanguageModel(len(encoding))
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    for i in tqdm(range(config.EPOCHS + 1)):
        x, y = get_batch(train_set)

        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % config.EVAL_INTERVAL == 0:
            out = estimate_loss(model, train_set, test_set)
            logger.info(
                f"Step {i}, Training Loss: {out['train']:.4f}, Testing Loss: {out['test']:.4f}"
            )

    # Generate text
    start_token = torch.zeros([1, 1], dtype=torch.long, device=DEVICE)
    generated_text = decode(model.generate(start_token, max_new_tokens=500)[0].tolist())
    print("generated_text: ", generated_text)


@app.command()
def tril():
    B, T, C = config.BATCH_SIZE, config.BLOCK_SIZE, config.CHANNEL_SIZE

    x = torch.randn((B, T, C))

    key = nn.Linear(C, config.HEAD_SIZE, bias=False)
    query = nn.Linear(C, config.HEAD_SIZE, bias=False)
    value = nn.Linear(C, config.HEAD_SIZE, bias=False)
    k = key(x)
    q = query(x)
    wei = q @ k.transpose(-2, -1)

    tril = torch.tril(torch.ones(T, T))
    # wei = torch.zeros((T, T))
    wei = wei.masked_fill(tril == 0, float("-inf"))
    wei = F.softmax(wei, dim=-1)
    v = value(x)
    bow = wei @ v

    print("tril: ", tril)
    print("wei: ", wei)
    print("bow: ", bow)


if __name__ == "__main__":
    app()
