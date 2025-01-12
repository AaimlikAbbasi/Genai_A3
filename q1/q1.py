import os
import re
import unicodedata
import pickle
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# === Step 1: Data Preprocessing === #
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sentence, lang):
    if lang == "en":
        sentence = sentence.lower().strip()
    sentence = unicode_to_ascii(sentence)
    sentence = re.sub(r"([?.!,¿'])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    return sentence.strip()

def load_data(file_en, file_ur):
    with open(file_en, 'r', encoding='utf-8') as f_en, open(file_ur, 'r', encoding='utf-8') as f_ur:
        en_sentences = f_en.readlines()
        ur_sentences = f_ur.readlines()
    assert len(en_sentences) == len(ur_sentences), "Mismatch in sentence count."
    en_sentences = [preprocess_sentence(sent, "en") for sent in en_sentences]
    ur_sentences = [preprocess_sentence(sent, "ur") for sent in ur_sentences]
    return en_sentences, ur_sentences

def save_preprocessed_data(sentences_en, sentences_ur, save_path_en, save_path_ur):
    with open(save_path_en, 'w', encoding='utf-8') as f_en, open(save_path_ur, 'w', encoding='utf-8') as f_ur:
        for en, ur in zip(sentences_en, sentences_ur):
            f_en.write(en + '\n')
            f_ur.write(ur + '\n')
    print(f"Data saved to {save_path_en} and {save_path_ur}")

# Paths
dataset_path = "D:\\Genai_a2\]q1\\umc005-corpus (1)"
bible_train_en = os.path.join(dataset_path, "D:\\Genai_a2\\q1\\umc005-corpus (1)\\bible\\train.en")
bible_train_ur = os.path.join(dataset_path, "D:\\Genai_a2\\q1\\umc005-corpus (1)\\bible\\train.ur")

# Load data
train_en, train_ur = load_data(bible_train_en, bible_train_ur)
train_en, test_en, train_ur, test_ur = train_test_split(train_en, train_ur, test_size=0.1, random_state=42)

# Save the preprocessed data
save_preprocessed_data(train_en, train_ur, 'train.en', 'train.ur')
save_preprocessed_data(test_en, test_ur, 'test.en', 'test.ur')

# === Step 2: Tokenization with SentencePiece === #
def train_sentencepiece(sentences, model_prefix, vocab_size=3000):
    input_file = f"{model_prefix}_input.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    spm.SentencePieceTrainer.train(
        input=input_file, model_prefix=model_prefix, vocab_size=vocab_size,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3
    )

# Train and load SentencePiece models
train_sentencepiece(train_en, "spm_en", 3000)
train_sentencepiece(train_ur, "spm_ur", 3000)

sp_en = spm.SentencePieceProcessor(model_file="spm_en.model")
sp_ur = spm.SentencePieceProcessor(model_file="spm_ur.model")

def save_tokenized_data(src_sentences, tgt_sentences, src_tokenizer, tgt_tokenizer, save_path):
    tokenized_data = []
    for src, tgt in zip(src_sentences, tgt_sentences):
        src_ids = src_tokenizer.encode(src)
        tgt_ids = tgt_tokenizer.encode(tgt)
        tokenized_data.append((src_ids, tgt_ids))
    with open(save_path, 'wb') as f:
        pickle.dump(tokenized_data, f)
    print(f"Tokenized data saved to {save_path}")

save_tokenized_data(train_en, train_ur, sp_en, sp_ur, 'tokenized_train.pkl')
save_tokenized_data(test_en, test_ur, sp_en, sp_ur, 'tokenized_test.pkl')

# === Step 3: Dataset Preparation === #
class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)

def collate_fn(batch):
    src, tgt = zip(*batch)
    src = torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=0)
    tgt = torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=0)
    return src, tgt

def load_tokenized_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

tokenized_train = load_tokenized_data('tokenized_train.pkl')
tokenized_test = load_tokenized_data('tokenized_test.pkl')

train_dataset = TranslationDataset(tokenized_train)
test_dataset = TranslationDataset(tokenized_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# === Step 4: Transformer Model === #
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dropout=dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt):
        src = self.positional_encoding(self.src_embedding(src))
        tgt = self.positional_encoding(self.tgt_embedding(tgt))
        output = self.transformer(src.transpose(0, 1), tgt.transpose(0, 1))
        return self.fc_out(output.transpose(0, 1))

# === Step 5: Training === #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(len(sp_en), len(sp_ur)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    epoch_loss = 0
    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")


# Directory to save results and model checkpoints
# Lists to store training and validation losses
training_losses = []
validation_losses = []

# Directory to save results and model checkpoints
os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# File to save training logs
log_file = "results/training_log.txt"

# Initialize log file
with open(log_file, "w") as f:
    f.write("Epoch, Training Loss, Validation Loss\n")

# Training loop with validation and saving
for epoch in range(10):
    # Training phase
    model.train()
    epoch_train_loss = 0

    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # Remove the last token in the target (teacher forcing)
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))  # Target shifted right
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    # Calculate average training loss
    avg_train_loss = epoch_train_loss / len(train_loader)
    training_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for src, tgt in tqdm(test_loader, desc=f"Epoch {epoch+1} Validation"):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            epoch_val_loss += loss.item()

    # Calculate average validation loss
    avg_val_loss = epoch_val_loss / len(test_loader)
    validation_losses.append(avg_val_loss)

    # Print losses for the epoch
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Save losses to log file
    with open(log_file, "a") as f:
        f.write(f"{epoch+1}, {avg_train_loss:.4f}, {avg_val_loss:.4f}\n")

    # Save model checkpoint
    checkpoint_path = f"checkpoints/transformer_epoch_{epoch+1}.pth"
    torch.save({
        "epoch": epoch+1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_loss": avg_train_loss,
        "validation_loss": avg_val_loss
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# === Plotting Training and Validation Loss === #
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss", marker="o")
plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss", marker="o")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.savefig("results/loss_curve.png")
plt.show()
