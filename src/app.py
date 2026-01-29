import streamlit as st
import torch
import pickle
from model import BiLSTM_CRF
import os

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/bilstm_crf.pth")
VOCAB_PATH = os.path.join(BASE_DIR, "../saved_models/vocab.pkl")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(MODEL_PATH):
        return None, None, None

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    
    word_to_ix = vocab['word_to_ix']
    tag_to_ix = vocab['tag_to_ix']
    ix_to_tag = {v: k for k, v in tag_to_ix.items()}
    
    # Initialize model (embedding matrix will be loaded from state dict)
    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, 100, 256)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model, word_to_ix, ix_to_tag

model, word_to_ix, ix_to_tag = load_resources()

# --- SMART EXTRACTION LOGIC (PURE ML) ---
def extract_info(text):
    if not model: return [], []

    # 1. Cleaning
    clean_text = text.replace(",", " , ").replace(".", " . ").replace("-", " - ").replace(":", " : ")
    words = clean_text.split()
    
    # 2. Prediction
    idxs = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in words]
    if not idxs: return [], []
    
    tensor_in = torch.tensor(idxs, dtype=torch.long).unsqueeze(0).to(DEVICE)
    mask = torch.ones(1, len(idxs), dtype=torch.uint8).to(DEVICE)
    
    with torch.no_grad():
        tag_ids = model.predict(tensor_in, mask)[0]
    
    tags = [ix_to_tag[i] for i in tag_ids]
    
    # --- NO HARDCODED LOGIC ENGINE ---
    # We trust the model.
    return words, tags

# --- USER INTERFACE ---
st.set_page_config(page_title="Advanced Land Record NER", layout="wide")
st.title("🧠 Pure BiLSTM-CRF (Advanced Data)")
st.info("This model runs without rule-based overrides. It relies entirely on the quality of training data.")

if not model:
    st.error("⚠️ Model not found!")
    
    # --- DEBUGGING INFO ---
    st.write("### 🕵️ Troubleshooting Info")
    st.write(f"**Current Script Location:** `{BASE_DIR}`")
    st.write(f"**Listing parent folder (`../`):**")
    try:
        parent_dir = os.path.dirname(BASE_DIR)
        files = os.listdir(parent_dir)
        st.code("\n".join(files))
        
        saved_models_path = os.path.join(parent_dir, "saved_models")
        if os.path.exists(saved_models_path):
             st.write(f"**Listing `saved_models` folder:**")
             st.code("\n".join(os.listdir(saved_models_path)))
        else:
             st.error("❌ `saved_models` folder DOES NOT EXIST here.")
    except Exception as e:
        st.error(f"Error listing files: {e}")
        
    st.stop()

input_text = st.text_area("Enter Legal Text:", height=100)

if st.button("Extract Entities"):
    if input_text:
        words, tags = extract_info(input_text)
        
        # 1. Visuals
        st.subheader("Document Analysis")
        html_code = ""
        for w, t in zip(words, tags):
            color = "#f0f2f6"
            label = ""
            
            if "SEL" in t: color = "#ffcccc"; label = "SELLER"
            elif "BUY" in t: color = "#ccffcc"; label = "BUYER"
            elif "SRV" in t: color = "#ccccff"; label = "SURVEY"
            elif "LOC" in t: color = "#e6ccff"; label = "LOC"
            elif "AMT" in t: color = "#ffedcc"; label = "AMT"
            elif "AREA" in t: color = "#d9f2d9"; label = "AREA"
            elif "DATE" in t: color = "#ffffcc"; label = "DATE"
            
            if t != "O":
                html_code += f"<span style='background-color:{color}; padding:4px; border-radius:4px; margin:2px; display:inline-block'><b>{w}</b> <span style='font-size:0.7em; opacity:0.6'>[{label}]</span></span> "
            else:
                html_code += f"{w} "
        
        st.markdown(f"<div style='line-height:2.5; font-size:1.1em'>{html_code}</div>", unsafe_allow_html=True)
        
        # 2. Database Record
        st.subheader("Extracted Database Record")
        entities = {"Seller": [], "Buyer": [], "Survey No": [], "Location": [], "Cost": [], "Area": [], "Date": []}
        
        for w, t in zip(words, tags):
            if "SEL" in t: entities["Seller"].append(w)
            if "BUY" in t: entities["Buyer"].append(w)
            if "SRV" in t: entities["Survey No"].append(w)
            if "LOC" in t: entities["Location"].append(w)
            if "AMT" in t: entities["Cost"].append(w)
            if "AREA" in t: entities["Area"].append(w)
            if "DATE" in t: entities["Date"].append(w)

        clean_data = {k: " ".join(v) for k, v in entities.items() if v}
        st.table(clean_data)
