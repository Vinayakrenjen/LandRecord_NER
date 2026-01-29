import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from model import BiLSTM_CRF
from train import LandRecordDataset, collate_fn
import os

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/bilstm_crf.pth")
VOCAB_PATH = os.path.join(BASE_DIR, "../saved_models/vocab.pkl")
DATA_PATH = os.path.join(BASE_DIR, "../data/land_records.csv")

def evaluate_current_model():
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
        print("❌ Model or Vocab not found!")
        return

    # 1. Load Resources
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    word_to_ix = vocab['word_to_ix']
    tag_to_ix = vocab['tag_to_ix']
    
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, 100, 256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 2. Load Data
    df = pd.read_csv(DATA_PATH)
    sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
    tags = df.groupby("Sentence #")["Tag"].apply(list).tolist()
    
    # We'll evaluate on a subset (last 10% similar to test split)
    split_idx = int(len(sentences) * 0.9)
    test_sents = sentences[split_idx:]
    test_tags = tags[split_idx:]
    
    dataset = LandRecordDataset(test_sents, test_tags, word_to_ix, tag_to_ix)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 3. Predict
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

    # 4. Calculate F1
    labels_to_check = [ix for tag, ix in tag_to_ix.items() if tag != "O"]
    score = f1_score(all_targets, all_preds, labels=labels_to_check, average='macro', zero_division=0)
    print(f"\n📊 Current Model F1 Score: {score:.4f}")

if __name__ == "__main__":
    evaluate_current_model()
