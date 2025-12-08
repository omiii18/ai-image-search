import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import pickle
import json
from collections import OrderedDict

# --- CONFIGURATION ---
BATCH_SIZE = 32 
DEVICE = "cpu" 

# Folder paths
EMBED_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBED_FOLDER, "faiss.index")
MAPPING_FILE = os.path.join(EMBED_FOLDER, "mapping.pkl")
SETTINGS_FILE = "settings.json"

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

    # Model should already be loaded in the main app, but load if needed:
    if not model:
        print(f"Loading CLIP AI Model on {device}...")
        model, preprocess = clip.load("ViT-B/32", device=device)

    print(f"Indexing photos in: {image_folder}")
    
    # Get list of files
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    total_images = len(image_files)

    # Use OrderedDict for consistent mapping
    filenames_map = OrderedDict() 
    embeddings_list = []

    print(f"Generating embeddings for {total_images} images with batch size {BATCH_SIZE}...")

    # Process images in batches
    for i in range(0, total_images, BATCH_SIZE):
        batch_files = image_files[i:i + BATCH_SIZE]
        image_batch = []
        
        for filename in batch_files:
            path = os.path.join(image_folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                image_batch.append(preprocess(img))
                filenames_map[filename] = len(filenames_map) 
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                continue

        if image_batch:
            image_tensor = torch.stack(image_batch).to(device)
            
            with torch.no_grad():
                batch_embeds = model.encode_image(image_tensor)
                batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)

            for embed in batch_embeds:
                embeddings_list.append(embed.cpu().numpy().flatten())
            
            print(f"  -> Processed {len(filenames_map)} / {total_images} images.")
            
    # --- FAISS INDEXING ---
    if not embeddings_list:
        print("No valid images found to process. Exiting.")
        return 0

    D = embeddings_list[0].shape[0] 
    embeddings_matrix = np.stack(embeddings_list)
    embeddings_matrix = embeddings_matrix.astype('float32')

    index = faiss.IndexFlatL2(D)
    index.add(embeddings_matrix)

    faiss.write_index(index, INDEX_FILE)

    with open(MAPPING_FILE, "wb") as f:
        pickle.dump(list(filenames_map.keys()), f)

    print(f"\nSUCCESS: FAISS index saved to: {INDEX_FILE}")
    return len(filenames_map)


if __name__ == '__main__':
    # This block is for manual testing of the indexer only
    print("Running Indexer in standalone mode...")
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    image_folder = load_settings()
    
    if not image_folder:
        print("ERROR: Image folder not configured.")
    else:
        build_index(image_folder, model, preprocess, DEVICE)