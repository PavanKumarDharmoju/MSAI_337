import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import argparse
import math
import numpy as np
from transformers import GPT2TokenizerFast
from scripts.starter import read_corpus

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
import argparse
import math
import numpy as np
import copy

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=4096, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        self.d_model = d_model  # Ensure this attribute is defined
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x * math.sqrt(self.d_model)  # Use self.d_model to ensure access to the class attribute
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1,2)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1,2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1,2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
    
    def forward(self, x):
        x2 = self.norm_1(x)
        x = x + self.attn(x2, x2, x2)
        x2 = self.norm_2(x)
        x = x + self.ff(x2)
        return x

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super(TransformerDecoder, self).__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)  # Ensure d_model is provided here
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(self.norm(x))


class TextDataset(Dataset):
    def __init__(self, tokens, sequence_length):
        self.tokens = tokens
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.tokens) - self.sequence_length
    
    def __getitem__(self, index):
        return (
            torch.tensor(self.tokens[index:index+self.sequence_length]),
            torch.tensor(self.tokens[index+1:index+self.sequence_length+1])
        )

def train(model, train_loader, valid_loader, test_loader, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.transpose(1, 2), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Training Loss {total_loss / len(train_loader)}")
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                loss = F.cross_entropy(outputs.transpose(1, 2), targets)
                valid_loss += loss.item()
        
        print(f"Epoch {epoch}: Validation Loss {valid_loss / len(valid_loader)}")
        
        # Optionally, you could add early stopping logic here

def test(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = F.cross_entropy(outputs.transpose(1, 2), targets)
            test_loss += loss.item()
    
    print(f"Test Loss: {test_loss / len(test_loader)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=16)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-seqlen', type=int, default=512)
    opt = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    train_text = read_corpus('data/wiki2.train.txt', tokenizer)
    valid_text = read_corpus("data/wiki2.valid.txt", tokenizer)
    test_text = read_corpus("data/wiki2.test.txt", tokenizer)
    
    train_data = TextDataset(train_text, opt.seqlen)
    valid_data = TextDataset(valid_text, opt.seqlen)
    test_data = TextDataset(test_text, opt.seqlen)

    train_loader = DataLoader(train_data, batch_size=opt.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.batchsize, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=opt.batchsize, shuffle=False)

    model = TransformerDecoder(tokenizer.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    train(model, train_loader, valid_loader, test_loader, optimizer, opt.epochs)
    test(model, test_loader)

if __name__ == "__main__":
    main()
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-batchsize', type=int, default=16)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-seqlen', type=int, default=512)
    opt = parser.parse_args()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    train_text = read_corpus('data/wiki2.train.txt',tokenizer)
    valid_text = read_corpus("data/wiki2.valid.txt", tokenizer)
    test_text = read_corpus("data/wiki2.test.txt", tokenizer)

    train_data = TextDataset(train_text, tokenizer, opt.seqlen)
    valid_data = TextDataset(valid_text, tokenizer, opt.seqlen)
    test_data = TextDataset(test_text, tokenizer, opt.seqlen)

    train_loader = DataLoader(train_data, batch_size=opt.batchsize, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=opt.batchsize, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=opt.batchsize, shuffle=False)

    model = TransformerDecoder(tokenizer.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    train(model, train_loader, valid_loader, test_loader, optimizer, opt.epochs)
    test(model, test_loader)

if __name__ == "__main__":
    main()
