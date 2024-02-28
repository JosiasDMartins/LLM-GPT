import lzma
import os

from tqdm import tqdm
import sys

def xz_files_in_dir(directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files


folder_path = "D:/UsersFiles/JosiasMartins/Downloads/openwebtext.tar/openwebtext"
output_file_train = "train_split.txt"
output_file_val = "val_split.txt"
vocab_file = "vocal.txt"
#split_files = int(input("How many files would you like to split this into?"))

files = xz_files_in_dir(folder_path)
total_files = len(files)

#Calculation the split indices
split_index = int(total_files * 0.9) #Training Dataset
files_train = files[:split_index]
files_val = files[split_index:]

#max_count = total_files // split_files if split_files != 0 else total_files

vocab = set()

#Processing taining files
with open(output_file_train, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

#Processing validation files
with open(output_file_val, "w", encoding="utf-8") as outfile:
    for filename in tqdm(files_train, total=len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt", encoding="utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')