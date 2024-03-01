import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import mmap
import random
import pickle
import os

def encode(input, chars):
    string_to_int = { ch:i for i, ch in enumerate(chars)} #Getting the str
    return [string_to_int[c] for c in input]  #Encoding from string to int
    
def decode(input, chars):
    int_to_string = { i:ch for i,ch in enumerate(chars)} #Getting the ints
    return ''.join([int_to_string[i] for i in input])  #Decoding from int to string

def get_random_chunk(split, batch_size, block_size, chars, train_file_path, valid_file_path):
    filename = train_file_path  if split == 'train' else valid_file_path

    #Creating decoders and encoders to easily convert data from String to int and int to string
    #string_to_int = { ch:i for i, ch in enumerate(chars)} #Getting the string
    int_to_string = { i:ch for i,ch in enumerate(chars)} #Getting the ints
    #encode = lambda s: [string_to_int[c] for c in s]  #Encoding from string to int
    decode = lambda l: ''.join([int_to_string[i] for i in l])  #Decoding from int to string
    
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            #Determine the file size and random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size*batch_size)

            #Seek to random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size*batch_size-1)

            #decode the block to a string
            decoded_block = block.decode('utf-8', errors='ignore').replace('\r','')

            #Splitting the dataset into train an test
            data = torch.tensor(encode(decoded_block, chars), dtype=torch.long)
    return data

def get_batch(split, batch_size, block_size, chars, train_file_path, valid_file_path, device):
    data = get_random_chunk(split, batch_size, block_size, chars, train_file_path, valid_file_path)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


#def estimate_loss(batch_size, block_size, eval_iters, chars, train_file_path, valid_file_path, model):
#    out = {}
#    model.eval()
#    for split in ['train', 'val']:
#        losses = torch.zeros(eval_iters)
#        for k in range(eval_iters):
#            X, Y = get_batch(split, batch_size, block_size, chars, train_file_path, valid_file_path)
#            logits, loss = model(X, Y)
#            losses[k] = loss.item()
#        out[split] = losses.mean()
#    model.train()
#    return out


class Head(nn.Module):
    def __init__(self,head_size, n_embd, dropout, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  #key transformation
        q = self.query(x)
        #Compute attention scores
        # Matrix coalculation - query by key (transposed)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 #(B, T, hs) @ (B, hs, T) -> B, T, T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
             nn.Linear(n_embd, 4 * n_embd),
             nn.ReLU(),
             nn.Linear(4 * n_embd, n_embd),
             nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, dropout, block_size)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)   #Self Attentiion
        x = self.ln1(x + y)  #Linear Normalization
        y = self.ffwd(x)  #Feed Forward
        x = self.ln2(x + y) #Linear Normalization
        return x

def tokenizer(vocab_file_path):
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            chars = sorted(list(set(text)))
        vocab_size = len(chars)
        return chars, vocab_size


class GptModel(nn.Module):
    def __init__(self, n_embd, n_head, n_layers, block_size, dropout, vocab_file_path, device):
        super().__init__()
        self.device = device #'cuda' if torch.cuda.is_available() else 'cpu'
        self.apply(self._init_weights)            
        #self.n_embd = n_embd
        self.chars = ""
        self.chars, self.vocab_size = tokenizer(vocab_file_path)
        
        #print(self.chars)
        #print(self.vocab_size)
        
        #self.vocab_size = len(self.chars)
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(self.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, n_embd)
        #Blocs
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, dropout=dropout,block_size=block_size) for _ in range(n_layers)])  #Creating the decoding blocks
        self.ln_f = nn.LayerNorm(n_embd)   #Final normalization Layer
        self.lm_head = nn.Linear(n_embd, self.vocab_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, index, targets=None):
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))  #(T, C)
        x = tok_emb + pos_emb    #(B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)        
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, index, max_new_tokens):
        #index is (B, T) array of indexes in the current context
        for _ in range(max_new_tokens):
            index_cond = index[:, -self.block_size:]
            #Getting predictions
            logits, loss = self.forward(index_cond)
            #Focus only on the last time step
            logits = logits[:, -1, :]
            #Apply softmax to get probs
            probs = F.softmax(logits, dim=-1)
            #Sample from distrib
            index_next = torch.multinomial(probs, num_samples=1)
            #Append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  #(B, T+1)
        return index
        
    def estimate_loss(self, batch_size, block_size, eval_iters, chars, train_file_path, valid_file_path):
        #@torch.no_grad()  #Reducing computation
        out = {}
        self.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split, batch_size, block_size, chars, train_file_path, valid_file_path, self.device)
                logits, loss = self.forward(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train()
        return out
        
    def train_model(self, max_iters, eval_iters, batch_size, train_file_path, valid_file_path, vocab_file_path):
        #Create a PyTorch Optimizer
        print(f'Params:\n Max Iterations: {max_iters}\n Eval Iterations: {eval_iters}\n batch Size: {batch_size}')
    
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        
        if(self.device == 'cuda'):
            print(f' CUDA Acceleration: {torch.version.cuda}\n')


        print('Starting Training...')
        for iter in range(max_iters):
            if iter % eval_iters == 0:                                       
                losses = self.estimate_loss( batch_size=batch_size, block_size=self.block_size, eval_iters=eval_iters, chars=self.chars, train_file_path=train_file_path, valid_file_path=valid_file_path)
                print(f" step: {iter}, train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")
            
            
            xb, yb = get_batch('train', batch_size, self.block_size, self.chars, train_file_path, valid_file_path, device=self.device)
        
            #evaluate
            logits, loss = self.forward(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        print(f' Final Loss: {loss.item()}')
    
    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)
        print('Model saved')

    def load_model(model_path):
        print("Loading model parameters...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("lodaded")
        return model





