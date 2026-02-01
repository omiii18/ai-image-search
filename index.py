import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import pickle
import json
from collections import OrderedDict
import tkinter as tk
from tkinter import filedialog

# --- 1. ENABLE HEIC SUPPORT (Crucial for iPhone Photos) ---
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    print("WARNING: pillow_heif not installed. HEIC photos will not work.")
    print("Run: pip install pillow-heif")

# --- CONFIGURATION ---
BATCH_SIZE = 64  # Increased for speed on Mac
EMBED_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBED_FOLDER, "faiss.index")
MAPPING_FILE = os.path.join(EMBED_FOLDER, "mapping.pkl")
SETTINGS_FILE = "settings.json"

# Automatically select the fastest available hardware
if torch.backends.mps.is_available():
    DEVICE = "mps"    # Use Mac GPU (FASTEST)
elif torch.cuda.is_available():
    DEVICE = "cuda"   # Use NVIDIA GPU (Windows/Linux)
else:
    DEVICE = "cpu"    # Fallback (Slowest)

print(f"Running on: {DEVICE}")

os.makedirs(EMBED_FOLDER, exist_ok=True)

def load_settings():
    """Loads persistent settings to get the image folder path."""
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            return settings.get("image_folder_path", None)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def build_index(image_folder, model, preprocess, device):
    """Encodes images in batches and builds the FAISS index."""
    
    if not image_folder or not os.path.isdir(image_folder):
        print("Error: Image folder path is invalid or not set.")
        return 0

    if not model:
        print(f"Loading CLIP AI Model on {device}...")
        model, preprocess = clip.load("ViT-B/32", device=device)

    print(f"Indexing photos in: {image_folder}")
    
    # 1. Gather all valid image files (Including HEIC)
    valid_exts = (".png", ".jpg", ".jpeg", ".webp", ".heic", ".HEIC")
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_exts)]
    total_images = len(image_files)

    # Use OrderedDict for consistent mapping
    filenames_map = OrderedDict() 
    embeddings_list = []

    print(f"Generating embeddings for {total_images} images (Batch Size: {BATCH_SIZE})...")

    # 2. Process images in batches
    for i in range(0, total_images, BATCH_SIZE):
        batch_files = image_files[i:i + BATCH_SIZE]
        image_batch = []
        
        for filename in batch_files:
            # Skip hidden metadata files created by macOS on external drives
            if filename.startswith("._"):
                continue 

            path = os.path.join(image_folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                image_batch.append(preprocess(img))
                filenames_map[filename] = len(filenames_map) 
            except Exception as e:
                # Just print a small warning and keep going
                print(f"Skipping {filename}: {e}")
                continue

        # If we have valid images in this batch, process them
        if image_batch:
            image_tensor = torch.stack(image_batch).to(device)
            
            with torch.no_grad():
                batch_embeds = model.encode_image(image_tensor)
                batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)

            for embed in batch_embeds:
                embeddings_list.append(embed.cpu().numpy().flatten())
            
            print(f"  -> Processed {len(filenames_map)} / {total_images} images...")
            
    # --- FAISS INDEXING ---
    if not embeddings_list:
        print("No valid images found to process. Exiting.")
        return 0

    # Convert list of arrays to a big matrix
    D = embeddings_list[0].shape[0] 
    embeddings_matrix = np.stack(embeddings_list)
    embeddings_matrix = embeddings_matrix.astype('float32')

    # Create and save FAISS index
    index = faiss.IndexFlatL2(D)
    index.add(embeddings_matrix)

    faiss.write_index(index, INDEX_FILE)

    # Save filename mapping
    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(list(filenames_map.keys()), f)

    print(f"\nSUCCESS: Indexed {len(filenames_map)} images.")
    print(f"Index saved to: {INDEX_FILE}")
    return len(filenames_map)


if __name__ == '__main__':
    # Standalone testing mode
    print("--- Running Indexer in Standalone Mode ---")
    
    # 1. Load CLIP Model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    # 2. Get Path (Settings or Manual Input)
    image_folder = load_settings()
    
    if not image_folder:
        print("\nWARNING: 'settings.json' not found.")
        image_folder = input("Paste the full path to your images folder: ").strip().strip('"')
    
    # 3. Run
    if os.path.isdir(image_folder):
        build_index(image_folder, model, preprocess, DEVICE)
    else:
        print(f"\nERROR: The folder path '{image_folder}' does not exist.")