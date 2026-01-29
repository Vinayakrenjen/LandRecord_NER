import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/land_records.csv")

def fix_csv():
    print(f"🔧 Fixing CSV: {DATA_PATH}")
    cleaned_rows = []
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split(",")
        
        # Header or valid line
        if len(parts) == 3:
            cleaned_rows.append(parts)
        elif len(parts) > 3:
            # We have extra commas. Format is ID, Word, Tag.
            # So everything between index 1 and -1 is the "Word".
            sent_id = parts[0]
            tag = parts[-1]
            word = ",".join(parts[1:-1]).strip('"') # Join back and strip existing quotes if any
            
            # Reconstruct with quotes
            cleaned_rows.append([sent_id, word, tag])
        else:
            print(f"⚠️ Skipping weird line: {line}")

    # Write back
    with open(DATA_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cleaned_rows)
        
    print(f"✅ Fixed {len(cleaned_rows)} rows.")

if __name__ == "__main__":
    fix_csv()
