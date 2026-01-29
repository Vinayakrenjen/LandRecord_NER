import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import pickle
import os

from model import BiLSTM_CRF

# --- CONFIGURATION ---
BATCH_SIZE = 32
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
EPOCHS = 15 # Increased for diverse data
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Handle Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/land_records.csv")
GLOVE_PATH = os.path.join(BASE_DIR, "../data/glove/glove.6B.100d.txt")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../saved_models/bilstm_crf.pth")
VOCAB_SAVE_PATH = os.path.join(BASE_DIR, "../saved_models/vocab.pkl")

print(f"🚀 Training on: {DEVICE}")

# --- 1. PREPROCESSING FUNCTIONS ---
class LandRecordDataset(Dataset):
    def __init__(self, sentences, tags, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.tags = tags
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence_ids = [self.word_to_ix.get(w, self.word_to_ix["<UNK>"]) for w in self.sentences[idx]]
        tag_ids = [self.tag_to_ix[t] for t in self.tags[idx]]
        return torch.tensor(sentence_ids, dtype=torch.long), torch.tensor(tag_ids, dtype=torch.long)

def collate_fn(batch):
    sentences, tags = zip(*batch)
    lengths = [len(s) for s in sentences]
    max_len = max(lengths)
    mask = torch.zeros(len(sentences), max_len, dtype=torch.uint8)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
    tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
    return sentences_padded, tags_padded, mask

def load_glove_embeddings(path, word_to_ix, embedding_dim):
    print(f"🌍 Loading GloVe from {path}...")
    if not os.path.exists(path):
        print("⚠️ GloVe file not found! Training from scratch (random embeddings).")
        return None

    embeddings_index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            
    print(f"   Found {len(embeddings_index)} word vectors.")
    
    vocab_size = len(word_to_ix)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    hits = 0
    misses = 0
    
    for word, i in word_to_ix.items():
        if i == 0: continue # Skip padding
        
        # Try exact match, then lowercase
        vector = embeddings_index.get(word)
        if vector is None:
            vector = embeddings_index.get(word.lower())
            
        if vector is not None:
            embedding_matrix[i] = vector
            hits += 1
        else:
            misses += 1
            # Initialize unknown words with random normal distribution to break symmetry
            embedding_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim,))
            
    print(f"   ✅ Mapped {hits} words. Missed {misses} words.")
    return torch.tensor(embedding_matrix, dtype=torch.float)

# --- 2. LOAD & PREPARE DATA ---
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Data file not found at {DATA_PATH}")
    print("   Please run generate_data.py or provide a CSV first.")
    exit(1)

print("📂 Loading Data...")
df = pd.read_csv(DATA_PATH)

sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
tags = df.groupby("Sentence #")["Tag"].apply(list).tolist()

words = list(set(df["Word"].values))
tags_unique = list(set(df["Tag"].values))

word_to_ix = {word: i + 2 for i, word in enumerate(words)}
word_to_ix["<PAD>"] = 0
word_to_ix["<UNK>"] = 1

tag_to_ix = {tag: i for i, tag in enumerate(tags_unique)}

train_sents, test_sents, train_tags, test_tags = train_test_split(sentences, tags, test_size=0.1, random_state=42)
train_data = LandRecordDataset(train_sents, train_tags, word_to_ix, tag_to_ix)
test_data = LandRecordDataset(test_sents, test_tags, word_to_ix, tag_to_ix)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 3. INITIALIZE MODEL ---
embedding_matrix = load_glove_embeddings(GLOVE_PATH, word_to_ix, EMBEDDING_DIM)

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, dropout=0.5, pretrained_embeddings=embedding_matrix).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- 4. EVALUATION ---
def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for sentence_in, targets, mask in loader:
            sentence_in, mask = sentence_in.to(DEVICE), mask.to(DEVICE)
            predictions = model.predict(sentence_in, mask)
            for i, pred_seq in enumerate(predictions):
                true_seq = targets[i][:len(pred_seq)].tolist()
                all_preds.extend(pred_seq)
                all_targets.extend(true_seq)
    
    labels_to_check = [ix for tag, ix in tag_to_ix.items() if tag != "O"]
    return f1_score(all_targets, all_preds, labels=labels_to_check, average='macro', zero_division=0)

# --- 5. TRAINING LOOP ---
print("🔥 Starting Training...")
best_f1 = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for sentence_in, targets, mask in progress_bar:
        sentence_in, targets, mask = sentence_in.to(DEVICE), targets.to(DEVICE), mask.to(DEVICE)
        model.zero_grad()
        loss = model(sentence_in, targets, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': total_loss / len(train_loader)})
    
    val_f1 = evaluate(model, test_loader)
    print(f"   Epoch {epoch+1} | Loss: {total_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f}")
    
    if val_f1 >= best_f1:
        best_f1 = val_f1
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        with open(VOCAB_SAVE_PATH, "wb") as f:
            pickle.dump({'word_to_ix': word_to_ix, 'tag_to_ix': tag_to_ix}, f)
        print(f"   🎉 Saved Best Model (F1: {best_f1:.4f})")

print("✅ Training Complete.")
