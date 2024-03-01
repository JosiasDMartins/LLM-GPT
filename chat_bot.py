import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import mmap
import random
import pickle
import os
import argparse


import gpt_model_kit as GPT


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

model_filename = 'jdm_model.pkl'

#Definitions / hyperparameters
batch_size = 64
block_size = 128
max_iters = 200  #3000
learning_rate = 3e-4 #, 3e-4, 1e-3, 1e-4
eval_iters = 100
n_embd = 384
n_head = 8  #8
n_layers = 8  #8 #Normaly same number of decoders
dropout = 0.2

vocab_file_path = 'D:/AI_Datasets/LLM/openwebtext/vocab.txt'


# Recriate the model to configure all parameters of the object
model = GPT.GptModel(n_embd, n_head, n_layers, block_size, dropout, vocab_file_path, device)
m = model.to(device)

#Reloading the model
model = GPT.GptModel.load_model(model_filename)
m = model.to(device)


while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(GPT.encode(prompt, m.chars), dtype=torch.long, device=device)
    generated_chars = GPT.decode(m.generate(context.unsqueeze(0), max_new_tokens=150)[0].tolist(), m.chars)
    print(f'Completion:\n{generated_chars}')