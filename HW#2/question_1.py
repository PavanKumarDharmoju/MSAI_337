import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import argparse
import numpy as np
from transformers import GPT2TokenizerFast
import logging
from typing import Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Embedder(nn.Module):
    """ Embedding layer for the transformer model. """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass for embeddings. """
        return self.embed(x)

class PositionalEncoder(nn.Module):
    """ Positional Encoder that adds positional information to token embeddings. """
    def __init__(self, d_model: int, max_seq_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention layer for model. """
    def __init__(self, heads: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_model // heads
        self.heads = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bs = q.size(0)
        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.heads, self.d_k).transpose(1,2)
        q = self.q_linear(q).view(bs, -1, self.heads, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.heads, self.d_k).transpose(1,2)
        # calculate attention using scaled dot product
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v).transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(output)

class FeedForward(nn.Module):
    """ FeedForward network for model, used after attention. """
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class DecoderLayer(nn.Module):
    """ Single layer of the GPT-2 style decoder. """
    def __init__(self, d_model: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout)
        self.ff = FeedForward(d_model, dropout)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = self.norm_1(x)
        x = x + self.attn(x2, x2, x2)
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x

class TransformerDecoder(nn.Module):
    """ Transformer decoder model for autoregressive tasks like language modeling. """
    def __init__(self, vocab_size: int, d_model: int, N: int, heads: int, dropout: float):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.out(x)

class TextDataset(Dataset):
    """ Dataset class for loading data. """
    def __init__(self, tokens: list, sequence_length: int):
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.tokens[index:index+self.sequence_length], dtype=torch.long),
            torch.tensor(self.tokens[index+1:index+self.sequence_length+1], dtype=torch.long)
        )

def train(model: nn.Module, data_loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """ Training loop for the model. """
    model.train()
    total_loss = 0
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs.transpose(1, 2), targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    logging.info(f"Average Training Loss: {average_loss}")

def validate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """ Validation loop for the model. """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.transpose(1, 2), targets)
            total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    logging.info(f"Average Validation Loss: {average_loss}")
    return average_loss

def save_model(model, path):
    """ Saves the model's state dictionary to the specified path. """
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GPT2-style autoregressive language model.")
    parser.add_argument('-epochs', type=int, default=20, help='Number of epochs to train.')
    parser.add_argument('-d_model', type=int, default=512, help='Dimension of the model.')
    parser.add_argument('-n_layers', type=int, default=6, help='Number of layers.')
    parser.add_argument('-heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('-batchsize', type=int, default=16, help='Batch size.')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('-seqlen', type=int, default=512, help='Sequence length.')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    try:
        train_tokens = tokenizer.encode(open('data/wiki.train.tokens').read())
        valid_tokens = tokenizer.encode(open('data/wiki.valid.tokens').read())
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    train_data = TextDataset(train_tokens, opt.seqlen)
    valid_data = TextDataset(valid_tokens, opt.seqlen)

    train_loader = DataLoader(train_data, batch_size=opt.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.batchsize, shuffle=False)

    model = TransformerDecoder(tokenizer.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout).to(device)
    optimizer = Adam(model.parameters(), lr=opt.lr)

    for epoch in range(opt.epochs):
        logging.info(f"Epoch {epoch+1}/{opt.epochs}")
        train(model, train_loader, optimizer, device)
        validate(model, valid_loader, device)
        # Save the model at the end of each epoch
        save_model(model, f"model_epoch_{epoch+1}.pth")
    # Optionally save the final model separately
    save_model(model, "models/final_model.pth")

if __name__ == "__main__":
    main()
