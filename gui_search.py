import pickle
import torch
import clip
import os
import faiss
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# --- ENVIRONMENT FIX (OMP Error) ---
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# --- 1. SETUP AI & LOAD DATA ---
device = "cpu" 
print(f"Loading AI model on {device}... please wait.")

# Load CLIP Model
model, preprocess = clip.load("ViT-B/32", device=device)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "embeddings/faiss.index")
MAPPING_FILE = os.path.join(BASE_DIR, "embeddings/mapping.pkl")
IMAGE_DIR = os.path.join(BASE_DIR, "images")

# Load FAISS Index and Filename Mapping
try:
    index = faiss.read_index(INDEX_FILE)
    with open(MAPPING_FILE, "rb") as f:
        filenames = pickle.load(f)
    print(f"Loaded FAISS index with {index.ntotal} images.")
except FileNotFoundError:
    messagebox.showerror("Error", "FAISS index files not found. Run index.py first!")
    exit()

# --- KEYWORDS FOR AUTO-SUGGESTION ---
# Expand this list based on your indexed image content!
SUGGESTION_KEYWORDS = [
    "dog", "cat", "car", "person", "beach", "sunset", "tree", "food",
    "building", "mountain", "water", "yellow", "red", "motorcycle", "newborn"
]

# --- 2. THE SEARCH LOGIC ---
def search_images(query, k=10): # k is set to 10 for 8-10 matches
    # 1. Convert text query to vector
    text_token = clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_vector = model.encode_text(text_token)
        # Normalize the query vector
        query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)
        
    # Convert to float32 NumPy array for FAISS
    query_vector_np = query_vector.cpu().numpy().astype('float32')

    # 2. Perform FAISS Search (Instantaneous!)
    D, I = index.search(query_vector_np, k)
    
    # 3. Format results
    results = []
    for distance, faiss_id in zip(D[0], I[0]):
        filename = filenames[faiss_id]
        
        # FAISS uses L2 distance; convert to similarity score (0 to 1)
        # Clamp distance at 0 to prevent numerical errors
        distance = max(0, distance)
        similarity_score = 1 - (distance / 2) # Correct formula for normalized vectors
        
        results.append((filename, similarity_score))

    return results

# --- 3. THE USER INTERFACE (GUI) ---
# NOTE: Tkinter widgets (root, results_frame, etc.) must be created before use.
# They are created in the START THE APP section.

def on_search():
    global results_frame, status_label, root # Ensure access to global GUI objects
    
    query = search_entry.get()
    if not query:
        messagebox.showwarning("Warning", "Please enter a search term.")
        return

    status_label.config(text=f"Searching for '{query}'...")
    root.update()

    # The new search call
    results = search_images(query)
    
    # --- CLEAR PREVIOUS RESULTS ---
    # Delete all widgets inside the results_frame
    for widget in results_frame.winfo_children():
        widget.destroy()

    if not results:
        status_label.config(text="No matches found.")
        return

    # --- DISPLAY NEW RESULTS IN 4 COLUMNS ---
    COLUMNS = 4 
    current_row_frame = None

    # Loop through all results (up to k=10)
    for index, (filename, score) in enumerate(results):

        # Start a new row frame every 4 items
        if index % COLUMNS == 0:
            current_row_frame = tk.Frame(results_frame)
            # pack the new row frame
            current_row_frame.pack(fill='x', pady=5) 

        path = os.path.join(IMAGE_DIR, filename)

        # Create a "Card" for the result, placed in the current row frame
        card = tk.Frame(current_row_frame, bd=1, relief="solid", padx=5, pady=5)
        # pack horizontally, expanding to fill space
        card.pack(side="left", padx=5, expand=True, fill='x')

        try:
            # Load and show image
            img = Image.open(path)
            img.thumbnail((150, 150))  # Resize for 4 columns
            # Convert to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(img)

            img_label = tk.Label(card, image=photo)
            img_label.image = photo # Keep reference
            img_label.pack()

            # Show details
            tk.Label(card, text=filename, font=("Arial", 9, "bold")).pack(pady=3)
            # Display score correctly formatted (e.g., 92.50%)
            tk.Label(card, text=f"Match: {score:.2%}", fg="green").pack() 

        except Exception as e:
            tk.Label(card, text=f"Error loading {filename}").pack()
            print(f"Could not load image {filename}: {e}")

    status_label.config(text=f"Done. Displaying top {len(results)} results.")

# --- AUTOCOMPLETE LOGIC ---
def show_suggestions(event):
    """Called whenever a key is released in the search entry."""
    # Ensure the listbox exists and the entry is active
    if not hasattr(root, 'suggestions_listbox'):
        return

    # Get the current text
    prefix = search_entry.get().lower()

    # Clear previous data
    root.suggestions_listbox.delete(0, tk.END)

    if not prefix:
        root.suggestions_listbox.place_forget() # Hide the listbox
        return

    # Filter keywords that start with the prefix
    suggestions = [
        word for word in SUGGESTION_KEYWORDS 
        if word.lower().startswith(prefix)
    ]

    if suggestions:
        # Show the listbox
        root.suggestions_listbox.place(x=search_entry.winfo_x(), 
                                       y=search_entry.winfo_y() + search_entry.winfo_height(),
                                       relwidth=search_entry.winfo_width())

        # Populate the listbox
        for suggestion in suggestions:
            root.suggestions_listbox.insert(tk.END, suggestion)

        # Bind the click event to select the word
        root.suggestions_listbox.bind("<<ListboxSelect>>", select_suggestion)

    else:
        root.suggestions_listbox.place_forget() # Hide if no suggestions

def select_suggestion(event):
    """Called when a suggestion is clicked in the listbox."""
    if root.suggestions_listbox.curselection():
        # Get the selected item's text
        selected_word = root.suggestions_listbox.get(root.suggestions_listbox.curselection())

        # Set the selected word as the entry content
        search_entry.delete(0, tk.END)
        search_entry.insert(0, selected_word)

        # Hide the listbox
        root.suggestions_listbox.place_forget()

        # Optional: Immediately perform search
        # on_search()

# --- 4. START THE APP (Global Variables Defined Here) ---
root = tk.Tk()
root.title("My Research AI Search Tool")
# Increase window size to accommodate 4 columns better
root.geometry("1000x500") 

# 1. Top Search Bar
top_frame = tk.Frame(root, pady=20)
top_frame.pack()

tk.Label(top_frame, text="Search your Photos:", font=("Arial", 14)).pack(side="left", padx=5)
search_entry = tk.Entry(top_frame, width=30, font=("Arial", 14))
search_entry.pack(side="left", padx=5)

# *** ADD BINDING ***
# Bind the KeyRelease event to the show_suggestions function
search_entry.bind("<KeyRelease>", show_suggestions) 

btn = tk.Button(top_frame, text="Search", command=on_search, bg="blue", fg="white")
btn.pack(side="left", padx=5)

# *** ADD LISTBOX FOR SUGGESTIONS ***
# Initialize the Listbox, but do NOT pack/grid it yet; we will use .place() to overlay it.
root.suggestions_listbox = tk.Listbox(root, height=5, selectmode=tk.SINGLE, 
                                      font=("Arial", 12), relief=tk.RIDGE, bd=1,
                                      background="white", highlightthickness=0)
root.suggestions_listbox.place_forget() # Start hidden

# 2. Status Bar
status_label = tk.Label(root, text="Ready", fg="gray")
status_label.pack()

# 3. Results Area (This will hold the row frames)
results_frame = tk.Frame(root, pady=20)
results_frame.pack(fill='both', expand=True)

root.mainloop()