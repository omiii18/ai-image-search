import os
import torch
import clip
from PIL import Image
import numpy as np
import faiss
import pickle

# --- CONFIGURATION ---
BATCH_SIZE = 32 # Process 32 images at once for speed
device = "cpu" 
print("Using device:", device)

# Load CLIP Model
model, preprocess = clip.load("ViT-B/32", device=device)

# Folder paths
IMAGE_FOLDER = "images"
EMBED_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBED_FOLDER, "faiss.index")
MAPPING_FILE = os.path.join(EMBED_FOLDER, "mapping.pkl")

os.makedirs(EMBED_FOLDER, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
total_images = len(image_files)

embeddings_list = []
filenames = []
image_batch = []
files_in_batch = []

print(f"Generating embeddings for {total_images} images with batch size {BATCH_SIZE}...")

# Process images in batches
for i, filename in enumerate(image_files):
    path = os.path.join(IMAGE_FOLDER, filename)
    
    # Load and preprocess image
    img = Image.open(path).convert("RGB")
    image_batch.append(preprocess(img))
    files_in_batch.append(filename)

    # Check if the batch is full or we reached the last image
    if len(image_batch) == BATCH_SIZE or i == total_images - 1:
        
        # Stack the preprocessed images into a single tensor
        image_tensor = torch.stack(image_batch).to(device)
        
        with torch.no_grad():
            # Encode the entire batch at once
            batch_embeds = model.encode_image(image_tensor)
            # Normalize embeddings
            batch_embeds = batch_embeds / batch_embeds.norm(dim=-1, keepdim=True)

        # Store results
        for j in range(batch_embeds.shape[0]):
            embeddings_list.append(batch_embeds[j].cpu().numpy().flatten())
            filenames.append(files_in_batch[j])

        # Clear the batch lists
        image_batch = []
        files_in_batch = []
        print(f"  -> Processed {len(filenames)} / {total_images} images.")

# --- FAISS INDEXING ---
if not embeddings_list:
    print("No images found to process. Exiting.")
    exit()

D = embeddings_list[0].shape[0] 
embeddings_matrix = np.stack(embeddings_list)
embeddings_matrix = embeddings_matrix.astype('float32')

index = faiss.IndexFlatL2(D)
index.add(embeddings_matrix)

faiss.write_index(index, INDEX_FILE)

with open(MAPPING_FILE, "wb") as f:
    pickle.dump(filenames, f)

print(f"\nSUCCESS: FAISS index saved to: {INDEX_FILE}")
print(f"Indexed {len(filenames)} images in total.")
# --------------------- END OF SCRIPT ---------------------