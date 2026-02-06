import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import pickle
import json
from collections import OrderedDict
import ssl 
import sys
import sqlite3
import pytesseract

# --- 1. SSL FIX FOR MAC (Prevents download errors) ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- 2. HEIC SUPPORT ---
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# --- CONFIGURATION ---
# "ViT-L/14" is the Advanced Model (Smarter but slightly slower)
MODEL_NAME = "ViT-L/14" 
BATCH_SIZE = 64  
EMBED_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBED_FOLDER, "faiss.index")
MAPPING_FILE = os.path.join(EMBED_FOLDER, "mapping.pkl")
KEYWORDS_FILE = os.path.join(EMBED_FOLDER, "keywords.json")
OCR_DB_FILE = os.path.join(EMBED_FOLDER, "ocr.db")
SETTINGS_FILE = "settings.json"

if torch.backends.mps.is_available():
    DEVICE = "mps" 
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"Running on: {DEVICE} | Model: {MODEL_NAME}")

os.makedirs(EMBED_FOLDER, exist_ok=True)

def load_settings():
    try:
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f).get("image_folder_path", None)
    except:
        return None

def init_ocr_db():
    """Initializes the SQLite database with FTS5 for full-text search."""
    conn = sqlite3.connect(OCR_DB_FILE)
    c = conn.cursor()
    # Create virtual table for FTS
    c.execute("CREATE VIRTUAL TABLE IF NOT EXISTS ocr_data USING fts5(filename, text_content)")
    conn.commit()
    return conn

def extract_text_from_image(img):
    """Extracts text from a PIL Image using Tesseract OCR."""
    try:
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"OCR Warning: {e}")
        return ""


def build_index(image_folder, model, preprocess, device):
    if not image_folder or not os.path.isdir(image_folder):
        print("Error: Invalid image folder.")
        return 0

    if not model:
        print(f"Loading {MODEL_NAME}...")
        model, preprocess = clip.load(MODEL_NAME, device=device)

    print(f"Indexing photos in: {image_folder}")
    
    # Initialize OCR DB
    ocr_conn = init_ocr_db()
    ocr_cursor = ocr_conn.cursor()
    # Clear existing data to avoid duplicates (optional, or we could upsert)
    ocr_cursor.execute("DELETE FROM ocr_data") 
    ocr_conn.commit()

    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".HEIC")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_exts)]
    total_images = len(image_files)
    filenames_map = OrderedDict() 
    embeddings_list = []

    # --- AUTO-TAGGING SETUP ---
    categories = ["animal", "bird", "city", "building", "landscape", "food", "people", 
                  "car", "beach", "mountain", "sunset", "party", "night", "flower", 
                  "water", "snow", "forest", "indoor", "outdoor", "sky", "sea", "street",
                  "dog", "cat", "architecture", "nature", "portrait", "vehicle", "dance", "drive"]
    
    text_inputs = clip.tokenize(categories).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    category_counts = np.zeros(len(categories))

    print(f"Processing {total_images} images...")

    # Ensure thumbnail folder exists
    THUMBNAIL_FOLDER = ".cache/thumbnails"
    os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)

    for i in range(0, total_images, BATCH_SIZE):
        batch_files = image_files[i:i + BATCH_SIZE]
        image_batch = []
        
        for filename in batch_files:
            if filename.startswith("._"): continue # Skip Mac ghost files

            path = os.path.join(image_folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                
                # --- THUMBNAIL GENERATION ---
                thumb_path = os.path.join(THUMBNAIL_FOLDER, filename)
                # Only save if it doesn't exist to save time on re-runs (optional, but good for speed)
                # For now, let's overwrite to ensure it's up to date or if logic changes
                img.copy().thumbnail((250, 250)) 
                # thumbnail() modifies in place, so we need to be careful not to affect the main img for CLIP
                # actually CLIP preprocess usually does a resize anyway, but let's be safe:
                
                thumb = img.copy()
                thumb.thumbnail((250, 250))
                thumb.save(thumb_path)
                
                image_batch.append(preprocess(img))
                filenames_map[filename] = len(filenames_map) 

                # --- OCR EXTRACTION ---
                extracted_text = extract_text_from_image(img)
                if extracted_text:
                    ocr_cursor.execute("INSERT INTO ocr_data (filename, text_content) VALUES (?, ?)", (filename, extracted_text))
                    
            except Exception as e:
                print(f"Skipping {filename}: {e}")

        if image_batch:
            image_tensor = torch.stack(image_batch).to(device)
            with torch.no_grad():
                batch_embeds = model.encode_image(image_tensor)
                batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)
                
                # --- UPDATE CATEGORY COUNTS ---
                similarity = (100.0 * batch_embeds @ text_features.T).softmax(dim=-1)
                top_indices = similarity.argmax(dim=-1).cpu().numpy()
                for idx in top_indices:
                    category_counts[idx] += 1

            for embed in batch_embeds:
                embeddings_list.append(embed.cpu().numpy().flatten())
            
            print(f"  -> Processed {len(filenames_map)} / {total_images}...")
            
    if not embeddings_list:
        print("No valid images found.")
        return 0

    # Save FAISS Index
    D = embeddings_list[0].shape[0] 
    embeddings_matrix = np.stack(embeddings_list).astype('float32')
    index = faiss.IndexFlatL2(D)
    index.add(embeddings_matrix)
    faiss.write_index(index, INDEX_FILE)

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(list(filenames_map.keys()), f)

    # --- SAVE TOP KEYWORDS ---
    top_indices = np.argsort(category_counts)[::-1][:12] # Get top 12
    top_keywords = [categories[i] for i in top_indices if category_counts[i] > 0]
    
    with open(KEYWORDS_FILE, "w") as f:
        json.dump(top_keywords, f)

    print(f"\nSUCCESS: Indexed {len(filenames_map)} images with {MODEL_NAME}.")
    ocr_conn.commit()
    ocr_conn.close()
    return len(filenames_map)

if __name__ == '__main__':
    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    folder = load_settings() or input("Enter folder path: ").strip().strip('"')
    if os.path.isdir(folder):
        build_index(folder, model, preprocess, DEVICE)