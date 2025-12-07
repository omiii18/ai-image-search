import pickle
import torch
import clip
import os
import faiss
import numpy as np
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog 
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
def search_images(query_text=None, query_image_path=None, k=10):
    
    query_vector = None
    
    if query_image_path:
        # --- IMAGE QUERY MODE ---
        try:
            img = Image.open(query_image_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                query_vector = model.encode_image(img_tensor)
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image: {e}")
            return []
            
    elif query_text:
        # --- TEXT QUERY MODE (Original Logic) ---
        text_token = clip.tokenize([query_text]).to(device)
        with torch.no_grad():
            query_vector = model.encode_text(text_token)

    else:
        # No input provided
        return []

    # Normalize the query vector (CRITICAL for FAISS/L2 search)
    query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)
    
    # Convert to float32 NumPy array for FAISS
    query_vector_np = query_vector.cpu().numpy().astype('float32')

    # 2. Perform FAISS Search (Instantaneous!)
    D, I = index.search(query_vector_np, k)
    
    # 3. Format results (same as before)
    results = []
    for distance, faiss_id in zip(D[0], I[0]):
        filename = filenames[faiss_id]
        
        # Correct similarity score formula for normalized vectors
        distance = max(0, distance)
        similarity_score = 1 - (distance / 2)
        
        results.append((filename, similarity_score))

    return results

# --- 3. THE USER INTERFACE (GUI) ---
# NOTE: Tkinter widgets (root, results_frame, etc.) must be created before use.
# They are created in the START THE APP section.

# --- IMAGE SEARCH HANDLERS ---

def select_image():
    """Opens file dialog to select an image for query."""
    filepath = filedialog.askopenfilename(
        title="Select Query Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")]
    )
    if filepath:
        query_image_path.set(filepath)
        # Clear the text box when an image is selected
        search_entry.delete(0, tk.END)
        # Immediately run search on the selected image
        on_search() 

def clear_image_query():
    """Clears the selected image path."""
    query_image_path.set("")
    # Clear previous results display (optional)
    for widget in results_frame.winfo_children():
        widget.destroy()
    status_label.config(text="Ready")

def on_search():
    global results_frame, status_label, root 
    
    text_query = search_entry.get()
    image_query = query_image_path.get()
    
    if not text_query and not image_query:
        messagebox.showwarning("Warning", "Please enter a search term OR select an image.")
        return

    # Determine display text based on active search type
    display_query = f"Image: {os.path.basename(image_query)}" if image_query else f"Text: '{text_query}'"
    status_label.config(text=f"Searching for {display_query}...")
    root.update()

    # Dispatch to the core search function
    results = search_images(query_text=text_query, query_image_path=image_query)

    # --- REST OF THE DISPLAY LOGIC REMAINS THE SAME ---
    
    # Clear old results
    for widget in results_frame.winfo_children():
        widget.destroy()

    if not results:
        status_label.config(text="No matches found.")
        return
        
    # --- Loop through results and display them in columns --- 
    # (The rest of the function remains the same as your previous working version)
    # ... (the loop logic for displaying 4 columns of images)
    
    # ... (The loop logic you previously confirmed to be working)

    # Configuration for columns
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
root.title("My Research AI Search Tool (Text & Image Search)")
root.geometry("1000x600") # Slightly larger to fit new elements

# --- New Global State for Image Search ---
query_image_path = tk.StringVar(value="") 

# 1. Top Search Bar & Buttons
top_frame = tk.Frame(root, pady=20)
top_frame.pack()

# Labels and Entry (Text Input)
tk.Label(top_frame, text="Search by Text:", font=("Arial", 14)).grid(row=0, column=0, padx=5, sticky='w')
search_entry = tk.Entry(top_frame, width=30, font=("Arial", 14))
search_entry.grid(row=0, column=1, padx=5, sticky='w')
search_entry.bind("<KeyRelease>", show_suggestions) 

btn = tk.Button(top_frame, text="Search", command=on_search, bg="blue", fg="white")
btn.grid(row=0, column=2, padx=5)

# --- Image Input Area ---
tk.Label(top_frame, text="OR by Image:", font=("Arial", 14)).grid(row=1, column=0, padx=5, sticky='w')

browse_btn = tk.Button(top_frame, text="Browse Image", command=lambda: select_image(), bg="green", fg="white")
browse_btn.grid(row=1, column=1, padx=5, sticky='w')

# Display selected file path
image_path_label = tk.Label(top_frame, textvariable=query_image_path, anchor="w", fg="gray", wraplength=350)
image_path_label.grid(row=1, column=2, columnspan=2, padx=5, sticky='w')

# Clear button for image search
clear_img_btn = tk.Button(top_frame, text="Clear Image", command=lambda: clear_image_query(), fg="red")
clear_img_btn.grid(row=1, column=3, padx=5, sticky='w') 

# 2. Status Bar and Results Area remain the same...

# 2. Status Bar
status_label = tk.Label(root, text="Ready", fg="gray")
status_label.pack()

# 3. Results Area (This will hold the row frames)
results_frame = tk.Frame(root, pady=20)
results_frame.pack(fill='both', expand=True)

root.mainloop()