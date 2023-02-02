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
    start = torch.randint(0, len(input_text) - config.BLOCK_SIZE - 1, (config.BATCH_SIZE,))
    x = torch.stack([input_text[i : i + config.BLOCK_SIZE] for i in start])
    y = torch.stack([input_text[i + 1 : i + config.BLOCK_SIZE + 1] for i in start])
    
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

    print(f"Training Loss: {torch.mean(losses['train'])}, Validation Loss: {torch.mean(losses['test'])}")
    model.train()


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(len(tokens), config.N_EMBED)
        self.lm_head = nn.Linear(config.N_EMBED, len(tokens))

    def forward(self, x, target=None):
        logits = self.embedding(x)
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
        for _ in range(max_tokens):
            idx_cond = idx[:, -config.BLOCK_SIZE:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)

            idx = torch.cat([idx, idx_next], dim=-1)

        return idx


@app.command()
def main():
    # Initialize model
    model = BigramModel()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    split = int(len(text) * config.TRAIN_TEST_SPLIT)
    train_text = torch.tensor(encoder(text[:split]), dtype=torch.long)
    test_text = torch.tensor(encoder(text[split:]), dtype=torch.long)

    start_idx = torch.zeros((1, 1), dtype=torch.long)    

    for epoch in tqdm(range(config.EPOCHS)):
        x, y = get_batch(train_text)

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % config.EVAL_INTERVAL == 0:
            estimate_loss(train_text, test_text, model)

    generated_response = decoder(model.generate(start_idx)[0].tolist())
    print('generated_response: ', generated_response)
        


if __name__ == "__main__":
    app()

