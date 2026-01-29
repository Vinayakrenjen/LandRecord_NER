import pandas as pd
import random
from faker import Faker
from tqdm import tqdm

fake = Faker('en_IN')

import os

# --- CONFIGURATION ---
NUM_SAMPLES = 5000
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(BASE_DIR, "../data/land_records.csv")

# --- 1. VOCABULARY BANK (The Building Blocks) ---
verbs_active = ["sold", "conveyed", "transferred", "assigned", "gifted", "handed over", "alienated", "leased", "executed"]
verbs_passive = ["was sold", "was conveyed", "was transferred", "has been assigned", "is gifted", "stands transferred", "is executed", "was executed"]

prepositions_seller = ["by", "from", "through", "on behalf of"]
prepositions_buyer = ["to", "in favour of", "in favor of", "unto", "for the benefit of"]

intros = [
    "This Deed of Sale witnesses that", "It is hereby agreed that", "The property described below", 
    "Whereas the Vendor is the absolute owner,", "This Indenture made on this day states", 
    "Know all men by these presents that", "This deed", ""
]

# --- 2. ENTITY GENERATORS ---
def get_seller():
    # Mix of simple names, Mr/Mrs, and Organizations
    if random.random() < 0.1:
        return f"M/s {fake.company()}"
    prefix = random.choice(["Mr.", "Mrs.", "Shri", "Smt.", ""])
    return f"{prefix} {fake.first_name()} {fake.last_name()}".strip()

def get_buyer():
    if random.random() < 0.1:
        return f"M/s {fake.company()}"
    prefix = random.choice(["Mr.", "Mrs.", "Shri", "Smt.", ""])
    return f"{prefix} {fake.first_name()} {fake.last_name()}".strip()

def get_survey():
    # 123, 123/A, 123/1B, Gat No 45
    base = str(random.randint(1, 999))
    suffix = random.choice(["", "/A", "/B", "/1", "/2", "/1A"])
    prefix = random.choice(["Survey No", "Gat No", "Hissa No", "Plot No", "S.No"])
    return f"{prefix} {base}{suffix}"

def get_location():
    # Village Wagholi, Taluka Haveli, Pune
    loc_type = random.choice(["Village", "M.", "Mouje", "at", "situated at"])
    return f"{loc_type} {fake.city()}"

def get_area():
    # 1500 Sqft, 2 Hectares
    val = random.randint(500, 10000)
    unit = random.choice(["Sq.Ft", "sqft", "Square Feet", "Sq Mtrs", "Gunthas", "Hectares"])
    return f"{val} {unit}"

def get_amount():
    # Rs. 50 Lakhs, 5,00,000/-, Rupees Fifty Thousand
    if random.random() < 0.5:
        return f"Rs. {random.randint(1, 99)} Lakhs"
    else:
        return f"Rs. {random.randint(10, 99)},000/-"

def get_date():
    return fake.date(pattern="%d-%m-%Y")

# --- 3. GRAMMAR ENGINE (The Logic) ---
def generate_complex_sentence(sentence_id):
    # Strategy: Build a sentence by assembling random grammatical components
    # We track (Word, Tag) pairs for every component added.
    
    tokens = [] # List of (Word, Tag) tuples
    
    # helper to add a phrase with a specific tag
    def add_phrase(text, tag="O"):
        # CRITICAL FIX: Align tokenization with app.py
        # app.py does: .replace(",", " , ").replace(".", " . ").replace("-", " - ").replace(":", " : ")
        clean_text = text.replace(",", " , ").replace(".", " . ").replace("-", " - ").replace(":", " : ")
        words = clean_text.split()
        
        for i, w in enumerate(words):
            if tag == "O":
                tokens.append((w, "O"))
            else:
                prefix = "B-" if i == 0 else "I-"
                tokens.append((w, f"{prefix}{tag}"))

    # --- PATTERN SELECTION ---
    pattern_type = random.choice([1, 2, 3, 4, 5])
    
    if pattern_type == 1: 
        # Passive: "Property was sold by Seller to Buyer"
        add_phrase(random.choice(intros))
        add_phrase("the property")
        add_phrase(random.choice(verbs_passive))
        add_phrase(random.choice(prepositions_seller))
        add_phrase(get_seller(), "SEL")
        add_phrase(random.choice(prepositions_buyer))
        add_phrase(get_buyer(), "BUY")
        add_phrase("for a consideration of")
        add_phrase(get_amount(), "AMT")
        add_phrase(".")

    elif pattern_type == 2:
        # Active: "Seller sold property to Buyer"
        add_phrase(get_seller(), "SEL")
        add_phrase("has")
        add_phrase(random.choice(verbs_active))
        add_phrase("the land bearing")
        add_phrase(get_survey(), "SRV")
        add_phrase("to")
        add_phrase(get_buyer(), "BUY")
        add_phrase("on")
        add_phrase(get_date(), "DATE")
        add_phrase(".")

    elif pattern_type == 3:
        # Location Focus: "Land at Loc measuring Area was transferred..."
        add_phrase("All that piece of land at")
        add_phrase(get_location(), "LOC")
        add_phrase("measuring")
        add_phrase(get_area(), "AREA")
        add_phrase("was")
        add_phrase(random.choice(verbs_active))
        add_phrase("from")
        add_phrase(get_seller(), "SEL")
        add_phrase("to")
        add_phrase(get_buyer(), "BUY")
        add_phrase(".")
        
    elif pattern_type == 4:
        # Payment Focus: "Amount was paid by Buyer to Seller"
        add_phrase("An amount of")
        add_phrase(get_amount(), "AMT")
        add_phrase("was paid by")
        add_phrase(get_buyer(), "BUY")
        add_phrase("to")
        add_phrase(get_seller(), "SEL")
        add_phrase("towards purchase of land at")
        add_phrase(get_location(), "LOC")
        add_phrase(".")

    elif pattern_type == 6:
        # Explicit Role Reversal: "Purchased by Buyer from Seller"
        add_phrase("The")
        add_phrase("property")
        add_phrase("was purchased by") # Passive 'by' usually implies Agent (Buyer in this verb context)
        add_phrase(get_buyer(), "BUY")
        add_phrase("from")
        add_phrase(get_seller(), "SEL")
        add_phrase(".")
        
    elif pattern_type == 7:
        # Title Separation: "The Vendor Mr. X sold to the Vendee Mrs. Y"
        add_phrase("The Vendor")
        add_phrase(get_seller(), "SEL")
        add_phrase("sold")
        add_phrase("the")
        add_phrase("land")
        add_phrase("bearing")
        add_phrase(get_survey(), "SRV")
        add_phrase("situated") # Explicitly add 'situated' as O tag to fix bleeding
        add_phrase("at")
        add_phrase(get_location(), "LOC")
        add_phrase("to the Vendee")
        add_phrase(get_buyer(), "BUY")
        add_phrase(".")

    elif pattern_type == 8:
        # Complex "situated" boundary check
        add_phrase("Land")
        add_phrase("bearing")
        add_phrase(get_survey(), "SRV")
        add_phrase("situated", "O") # Force 'situated' to always be O
        add_phrase("at")
        add_phrase(get_location(), "LOC")
        add_phrase("was")
        add_phrase(random.choice(verbs_passive))
        add_phrase("by")
        add_phrase(get_seller(), "SEL")
        add_phrase(".")

    return tokens

 

# RE-WRITING generate_complex_sentence TO INCLUDE NEW PATTERNS
def generate_complex_sentence(sentence_id):
    tokens = []
    
    def add_phrase(text, tag="O"):
        clean_text = text.replace(",", " , ").replace(".", " . ").replace("-", " - ").replace(":", " : ")
        words = clean_text.split()
        for i, w in enumerate(words):
            if tag == "O": tokens.append((w, "O"))
            else:
                prefix = "B-" if i == 0 else "I-"
                tokens.append((w, f"{prefix}{tag}"))

    pattern_type = random.randint(1, 11) # Expanded range to include new patterns 9-11

    if pattern_type == 1: 
        # Passive: "Property was sold by Seller to Buyer"
        add_phrase(random.choice(intros))
        add_phrase("the property")
        add_phrase(random.choice(verbs_passive))
        add_phrase(random.choice(prepositions_seller))
        add_phrase(get_seller(), "SEL")
        add_phrase(random.choice(prepositions_buyer))
        add_phrase(get_buyer(), "BUY")
        add_phrase("for a consideration of")
        add_phrase(get_amount(), "AMT")
        add_phrase(".")

    elif pattern_type == 2:
        # Active: "Seller sold property to Buyer"
        add_phrase(get_seller(), "SEL")
        add_phrase("has")
        add_phrase(random.choice(verbs_active))
        add_phrase("the land bearing")
        add_phrase(get_survey(), "SRV")
        add_phrase("to")
        add_phrase(get_buyer(), "BUY")
        add_phrase("on")
        add_phrase(get_date(), "DATE")
        add_phrase(".")

    elif pattern_type == 3:
        # Location Focus
        add_phrase("All that piece of land at")
        add_phrase(get_location(), "LOC")
        add_phrase("measuring")
        add_phrase(get_area(), "AREA")
        add_phrase("was")
        add_phrase(random.choice(verbs_active))
        add_phrase("from")
        add_phrase(get_seller(), "SEL")
        add_phrase("to")
        add_phrase(get_buyer(), "BUY")
        add_phrase(".")
        
    elif pattern_type == 4:
        # Payment Focus
        add_phrase("An amount of")
        add_phrase(get_amount(), "AMT")
        add_phrase("was paid by")
        add_phrase(get_buyer(), "BUY")
        add_phrase("to")
        add_phrase(get_seller(), "SEL")
        add_phrase("towards purchase of land at")
        add_phrase(get_location(), "LOC")
        add_phrase(".")

    elif pattern_type == 5:
        # Definition Style
        add_phrase("The Transferor is")
        add_phrase(get_seller(), "SEL")
        add_phrase("and the Transferee is")
        add_phrase(get_buyer(), "BUY")
        add_phrase(".")
        add_phrase("The property is")
        add_phrase(get_survey(), "SRV")
        add_phrase("located at")
        add_phrase(get_location(), "LOC")
        add_phrase(".")
        
    elif pattern_type == 6:
        # Explicit Role Reversal: "Purchased by Buyer from Seller"
        add_phrase("The property was purchased by") 
        add_phrase(get_buyer(), "BUY")
        add_phrase("from")
        add_phrase(get_seller(), "SEL")
        add_phrase("for")
        add_phrase(get_amount(), "AMT")
        add_phrase(".")
        
    elif pattern_type == 7:
        # Title Separation: "The Vendor Mr. X sold to the Vendee Mrs. Y"
        add_phrase("The Vendor")
        add_phrase(get_seller(), "SEL")
        add_phrase("sold")
        add_phrase("the land bearing")
        add_phrase(get_survey(), "SRV")
        add_phrase("situated", "O")
        add_phrase("at")
        add_phrase(get_location(), "LOC")
        add_phrase("to the Vendee")
        add_phrase(get_buyer(), "BUY")
        add_phrase(".")

    elif pattern_type == 8:
        # Complex "situated" boundary check
        add_phrase("Land bearing")
        add_phrase(get_survey(), "SRV")
        add_phrase("situated in")
        add_phrase(get_location(), "LOC")
        add_phrase("was sold by")
        add_phrase(get_seller(), "SEL")
        add_phrase(".")

    elif pattern_type == 9:
        # Date vs Amount Disambiguation (Hard Mode)
        # "Sold for Rs. 50 Lakhs on 12-12-2022"
        add_phrase("Sold for")
        add_phrase(get_amount(), "AMT")
        add_phrase("on")
        add_phrase("dated", "O") # Teach that 'dated' is not a date
        add_phrase(get_date(), "DATE")
        add_phrase("during the transaction") # filler
        add_phrase(".")

    elif pattern_type == 10:
        # Start/End Negatives (Anti-Hallucination)
        # "Executed on..." , "Sold today."
        add_phrase("Executed", "O") # Explicit O tag
        add_phrase("on")
        add_phrase(get_date(), "DATE")
        add_phrase("at")
        add_phrase(get_location(), "LOC")
        add_phrase("between")
        add_phrase(get_seller(), "SEL")
        add_phrase("and")
        add_phrase(get_buyer(), "BUY")
        add_phrase(".")
        
    elif pattern_type == 11:
        # Adverb Confusion
        # "Property sold today"
        add_phrase("This property was sold")
        add_phrase("today", "O") # Explicit O tag
        add_phrase("by")
        add_phrase(get_seller(), "SEL")
        add_phrase("to")
        add_phrase(get_buyer(), "BUY")
        add_phrase(".")

    return tokens

# --- 4. EXECUTION ---
if __name__ == "__main__":
    NUM_SAMPLES = 15000 # Increased from 5000
    print(f"🚀 Generating {NUM_SAMPLES} complex legal sentences...")
    data_rows = []

    for i in tqdm(range(NUM_SAMPLES)):
        sentence_tokens = generate_complex_sentence(i)
        for w, t in sentence_tokens:
            data_rows.append([f"Sentence: {i}", w, t])

    df = pd.DataFrame(data_rows, columns=["Sentence #", "Word", "Tag"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Saved to {OUTPUT_FILE}")
