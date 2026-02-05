import pickle
import torch
import clip
import os
import faiss
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import json
import threading
import time
from PIL import Image
import pillow_heif  # Import the library

# Register HEIC opener - This makes Image.open() work with HEIC automatically!
pillow_heif.register_heif_opener()
# --- ENVIRONMENT FIX (OMP Error) ---
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-L/14"  # Advanced Model
K_MATCHES = 12  # Number of search results to return
EMBED_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBED_FOLDER, "faiss.index")
MAPPING_FILE = os.path.join(EMBED_FOLDER, "mapping.pkl")
SETTINGS_FILE = "settings.json"

# Load the external index builder script
from index import build_index 

class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSearch AI Photo Library")
        self.root.geometry("1100x750")
        self.root.resizable(True, True)
        
        # --- Persistent State ---
        self.settings = self._load_settings()
        self.image_folder_path = self.settings.get("image_folder_path", "")
        
        # --- AI State ---
        self.model = None
        self.preprocess = None
        self.faiss_index = None
        self.filenames = []
        
        # --- GUI Variables ---
        self.status_text = tk.StringVar(value="Initializing...")
        self.query_image_path = tk.StringVar(value="")
        
        # --- Suggestions (User Guide) ---
        self.SUGGESTION_KEYWORDS = ["dance", "night drives", "parties", "trips", "week's memory", 
                                     "beach", "food", "animal", "sunset", "mountain", "car", "person"]
        
        # Build the main GUI layout
        self._build_ui()
        
        # Start the AI/Indexing process in a separate thread (Non-blocking startup)
        threading.Thread(target=self._initialize_ai, daemon=True).start()

    def _load_settings(self):
        """Loads persistent settings or creates defaults."""
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"image_folder_path": ""}
            
    def _save_settings(self):
        """Saves persistent settings."""
        self.settings["image_folder_path"] = self.image_folder_path
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def _initialize_ai(self):
        """Loads AI model and checks for index updates."""
        try:
            self.status_text.set("1. Loading CLIP AI Model (CPU)...")
            self.model, self.preprocess = clip.load("ViT-L/14", device=DEVICE)
            self.status_text.set("2. Model Loaded. Checking Library...")
            
            # --- Auto-indexing check (runs indexing if needed) ---
            if self.image_folder_path and os.path.isdir(self.image_folder_path):
                self._check_and_index_photos()
            else:
                self.status_text.set("Ready. Use 'Select Folder' to configure your photos.")
                
        except Exception as e:
            self.status_text.set(f"FATAL ERROR: Initialization failed. Check dependencies. {e}")
            messagebox.showerror("Fatal Error", str(e))

    def _check_and_index_photos(self):
        """Checks for new files and updates/loads the FAISS index."""
        if not self.image_folder_path or not self.model: return

        # 1. Get file count in user folder
        try:
            image_files = [f for f in os.listdir(self.image_folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
            current_file_count = len(image_files)
        except FileNotFoundError:
            self.status_text.set("Error: Photo folder not found. Please re-select.")
            return
            
        # 2. Load existing index info
        indexed_count = 0
        try:
            temp_index = faiss.read_index(INDEX_FILE)
            indexed_count = temp_index.ntotal
        except Exception:
            pass 

        if current_file_count == 0:
            self.status_text.set("Ready. No images found in selected folder.")
        elif current_file_count > indexed_count:
            self.status_text.set(f"3. Found {current_file_count - indexed_count} new images. Auto-indexing...")
            
            # --- REBUILD INDEX ---
            try:
                final_count = build_index(self.image_folder_path, self.model, self.preprocess, DEVICE)
                self.status_text.set(f"4. Indexing complete. {final_count} images indexed.")
            except Exception as e:
                self.status_text.set(f"Indexing failed: {e}")
        else:
            self.status_text.set("3. Index up to date.")
        
        # 4. Load the final index into memory for searching
        self._load_search_index()
        self.status_text.set(f"Ready. {self.faiss_index.ntotal if self.faiss_index else 0} images searchable.")
        
    def _load_search_index(self):
        """Loads the FAISS index and mapping into memory."""
        try:
            self.faiss_index = faiss.read_index(INDEX_FILE)
            with open(MAPPING_FILE, "rb") as f:
                self.filenames = pickle.load(f)
        except Exception:
            self.faiss_index = None
            self.filenames = []

    # --- UI COMPONENTS & HANDLERS ---
    
    def _build_ui(self):
        # Modern Dark Theme Aesthetics
        BG_COLOR = "#23272A"  # Dark Charcoal
        ACCENT_COLOR = "#7289DA" # Light Blue/Purple
        FG_COLOR = "#FFFFFF" # White Text
        HEADER_COLOR = "#424549" # Gray Header
        
        self.root.config(bg=BG_COLOR)
        
        # 1. Header/Settings Frame
        header_frame = tk.Frame(self.root, bg=HEADER_COLOR)
        header_frame.pack(side="top", fill="x", padx=0, pady=0)
        
        settings_frame = tk.Frame(header_frame, bg=HEADER_COLOR)
        settings_frame.pack(side="top", fill="x", padx=15, pady=10)
        
        tk.Label(settings_frame, text="Photo Directory:", bg=HEADER_COLOR, fg=FG_COLOR, font=("Inter", 12)).pack(side="left", padx=10)
        
        # Directory label (shows current path)
        self.dir_label = tk.Label(settings_frame, text=self._get_display_path(), 
                                  bg=HEADER_COLOR, fg=ACCENT_COLOR, font=("Inter", 10), anchor="w", width=50)
        self.dir_label.pack(side="left", padx=5)

        # Select Folder Button 
        tk.Button(settings_frame, text="Select Folder", command=self._select_folder, bg=ACCENT_COLOR, fg=FG_COLOR, font=("Inter", 10, "bold"), relief=tk.FLAT).pack(side="left", padx=10)
        
        tk.Button(settings_frame, text="Re-Index", command=self.reindex_thread, bg="#f39c12", fg="#000000", font=("Inter", 10, "bold"), relief=tk.FLAT).pack(side="right", padx=10)
        
        # 2. Search & Suggestions Frame (Center Top)
        search_frame = tk.Frame(self.root, bg=BG_COLOR, pady=20)
        search_frame.pack(fill="x", padx=15)

        # A. Suggestions Area (like the mobile app recents)
        suggestions_label = tk.Label(search_frame, text="Suggested Themes:", bg=BG_COLOR, fg="#99aab5", font=("Inter", 12, "bold"))
        suggestions_label.pack(anchor="w", padx=10, pady=(0, 5))
        
        suggestion_buttons_frame = tk.Frame(search_frame, bg=BG_COLOR)
        suggestion_buttons_frame.pack(fill="x", padx=10)
        
        for keyword in self.SUGGESTION_KEYWORDS:
            btn = tk.Button(suggestion_buttons_frame, text=keyword, command=lambda k=keyword: self._quick_search(k), 
                            bg="#5865F2", fg=FG_COLOR, font=("Inter", 10), relief=tk.FLAT, activebackground="#4F5EBD", activeforeground=FG_COLOR)
            btn.pack(side="left", padx=5, pady=5)
            
        # B. Search Input (Entry box)
        input_frame = tk.Frame(search_frame, bg=BG_COLOR, pady=15)
        input_frame.pack(fill="x", padx=10)
        
        self.search_entry = tk.Entry(input_frame, width=40, font=("Inter", 16), bd=0, relief=tk.FLAT, bg="#424549", fg=FG_COLOR, insertbackground=FG_COLOR)
        self.search_entry.pack(side="left", fill="x", expand=True)
        self.search_entry.bind("<KeyRelease>", self._show_suggestions) 

        # Search Button
        tk.Button(input_frame, text="Search", command=self.search_thread, bg=ACCENT_COLOR, fg=FG_COLOR, font=("Inter", 12, "bold"), relief=tk.FLAT).pack(side="left", padx=10)
        
        # Browse Button (for Image-to-Image Search)
        tk.Button(input_frame, text="Browse Image", command=self._select_image_query, bg="#2ECC71", fg=FG_COLOR, font=("Inter", 12, "bold"), relief=tk.FLAT).pack(side="left", padx=10)

        # C. Suggestion Listbox (hidden by default)
        self.suggestions_listbox = tk.Listbox(input_frame, height=5, selectmode=tk.SINGLE, 
                                          font=("Inter", 12), relief=tk.RIDGE, bd=1,
                                          background="#424549", fg=FG_COLOR, highlightthickness=0)
        self.suggestions_listbox.place_forget()
        
        # 3. Status Bar
        status_frame = tk.Frame(self.root, bg=BG_COLOR, padx=15, pady=5)
        status_frame.pack(fill="x")
        tk.Label(status_frame, textvariable=self.status_text, bg=BG_COLOR, fg="#99aab5", font=("Inter", 10)).pack(anchor="w")

        # 4. Results Area (Scrollable)
        results_canvas = tk.Canvas(self.root, bg="#36393F", borderwidth=0, highlightthickness=0)
        results_canvas.pack(side="bottom", fill="both", expand=True, padx=15, pady=15)
        
        # Frame that holds all the result cards
        self.results_frame = tk.Frame(results_canvas, bg="#36393F")
        
        # Configure scrollbar
        v_scrollbar = tk.Scrollbar(results_canvas, orient="vertical", command=results_canvas.yview)
        v_scrollbar.pack(side="right", fill="y")
        results_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        # Add frame to canvas
        results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        
        # Update scroll region and column weights (important for responsive layout)
        self.results_frame.bind("<Configure>", lambda e: (results_canvas.configure(scrollregion=results_canvas.bbox("all")), self._update_column_weights(self.results_frame)))

    def _update_column_weights(self, frame):
        """Ensures the 4 columns expand equally."""
        for i in range(4):
            frame.grid_columnconfigure(i, weight=1)

    def _get_display_path(self):
        """Returns a trimmed path for display."""
        if not self.image_folder_path:
            return "No folder selected."
        return f"...{self.image_folder_path[-50:]}" if len(self.image_folder_path) > 50 else self.image_folder_path
        
    def _select_folder(self):
        """Prompts user for file access and saves the path."""
        folder_selected = filedialog.askdirectory(title="Select your Local Photo Library (File Access Granted)")
        if folder_selected:
            self.image_folder_path = folder_selected
            self.dir_label.config(text=self._get_display_path())
            self._save_settings()
            self.reindex_thread() # Start indexing after setting folder
            
    def _quick_search(self, query):
        """Sets the query entry and runs search immediately."""
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, query)
        self.search_thread()

    def _show_suggestions(self, event):
        """Filters suggestion list and displays/hides the listbox on key release."""
        
        # Only show suggestions if the browse image field is empty
        if self.query_image_path.get():
            self.suggestions_listbox.place_forget()
            return
            
        prefix = self.search_entry.get().lower()
        self.suggestions_listbox.delete(0, tk.END)
        
        if not prefix:
            self.suggestions_listbox.place_forget()
            return
            
        suggestions = [word for word in self.SUGGESTION_KEYWORDS if word.lower().startswith(prefix)]
        
        if suggestions:
            # Calculate position to place the listbox right under the entry field
            entry_x = self.search_entry.winfo_rootx()
            entry_y = self.search_entry.winfo_rooty()
            
            x = entry_x - self.root.winfo_rootx()
            y = entry_y - self.root.winfo_rooty() + self.search_entry.winfo_height() + 2 # +2 for padding

            self.suggestions_listbox.place(x=x, y=y, width=self.search_entry.winfo_width())
            
            for suggestion in suggestions:
                self.suggestions_listbox.insert(tk.END, suggestion)
            
            self.suggestions_listbox.bind("<<ListboxSelect>>", self._select_suggestion)
        else:
            self.suggestions_listbox.place_forget()

    def _select_suggestion(self, event):
        """Sets the selected word as the entry content and initiates search."""
        if self.suggestions_listbox.curselection():
            selected_word = self.suggestions_listbox.get(self.suggestions_listbox.curselection())
            self.search_entry.delete(0, tk.END)
            self.search_entry.insert(0, selected_word)
            self.suggestions_listbox.place_forget()
            self.search_thread()
            
    def _select_image_query(self):
        """Selects an image file for Image-to-Image search."""
        # ADDED *.heic and *.HEIC to the filetypes list below
        filepath = filedialog.askopenfilename(
            title="Select Query Image",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.webp *.heic *.HEIC")
            ]
        )
        if filepath:
            self.query_image_path.set(filepath)
            self.search_entry.delete(0, tk.END) # Clear text query
            self.search_thread()

    def search_thread(self):
        """Starts the search function in a separate thread to prevent UI freezing."""
        # Clear image path if text is being typed
        if self.search_entry.get():
            self.query_image_path.set("")
            
        # Also clear query image path if clear button was used
        if self.query_image_path.get() and not os.path.exists(self.query_image_path.get()):
            self.query_image_path.set("")
            
        threading.Thread(target=self._run_search, daemon=True).start()
        
    def reindex_thread(self):
        """Starts re-indexing in a separate thread."""
        threading.Thread(target=self._check_and_index_photos, daemon=True).start()

    def _run_search(self):
        """Dispatches search based on query type and displays results."""
        
        text_query = self.search_entry.get()
        image_query = self.query_image_path.get()
        
        if not self.faiss_index:
            self.status_text.set("Error: Index not loaded. Please index your photos.")
            return

        if not text_query and not image_query:
            self.status_text.set("Please enter text or select an image.")
            return

        self.status_text.set(f"Searching: {'Image' if image_query else text_query}...")
        
        # --- 1. Get Query Vector ---
        query_vector = None
        if image_query:
            try:
                img = Image.open(image_query).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    query_vector = self.model.encode_image(img_tensor)
            except Exception as e:
                self.status_text.set(f"Error processing image: {e}")
                return
        elif text_query:
            text_token = clip.tokenize([text_query]).to(DEVICE)
            with torch.no_grad():
                query_vector = self.model.encode_text(text_token)

        query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)
        query_vector_np = query_vector.cpu().numpy().astype('float32')

        # --- Perform FAISS Search ---
        # Get results from FAISS
        D, I = self.faiss_index.search(query_vector_np, k=K_MATCHES)
        results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            score = float(D[0][i]) # This is the "distance" score (lower is better in FAISS L2)
            
            # 'match_percentage' is the number shown on screen (e.g., 22.14)
            # Convert L2 Distance to Cosine Similarity %: (1 - d^2/2) * 100
            match_percentage = (1 - (score / 2)) * 100
            
            if match_percentage < 10.0:
                continue
            
            filename = self.filenames[idx]
            results.append((filename, match_percentage))

        # --- Display Results ---
        self.root.after(0, lambda: self._display_results(results, image_query)) # Update GUI safely
        self.status_text.set(f"Search complete. Displaying top {len(results)} matches.")


    def _display_results(self, results, query_image_path=None):
        """Clears old results and displays new ones in a 4-column grid."""
        
        # Clear all previous widgets in the results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        start_row = 0
        if query_image_path and os.path.exists(query_image_path):
             self._display_query_image_card(query_image_path)
             start_row = 1
            
        COLUMNS = 4 

        for index, (filename, score) in enumerate(results):
            
            # Start actual results display from row 1 if showing query image
            row = (index // COLUMNS) + start_row
            col = index % COLUMNS

            path = os.path.join(self.image_folder_path, filename)

            # Card container
            card = tk.Frame(self.results_frame, bd=1, relief="solid", bg="#424549", padx=5, pady=5)
            card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")
            
            # --- Image Loading ---
            try:
                img = Image.open(path)
                img.thumbnail((180, 180)) 
                # Store PhotoImage as an attribute to prevent garbage collection
                card.photo = ImageTk.PhotoImage(img) 

                img_label = tk.Label(card, image=card.photo, bg="#424549")
                img_label.pack(pady=5)

                # Show details
                tk.Label(card, text=filename, bg="#424549", fg="#ecf0f1", font=("Inter", 9, "bold")).pack(pady=2)
                tk.Label(card, text=f"Match: {score:.2f}%", bg="#424549", fg="#2ECC71", font=("Inter", 10)).pack() 

            except Exception as e:
                tk.Label(card, text=f"Error loading {filename}", bg="#424549", fg="#e74c3c").pack()
                print(f"Could not load image {filename}: {e}")
                
    def _display_query_image_card(self, path):
        """Displays the source image card at the top of the results."""
        try:
            img = Image.open(path)
            img.thumbnail((180, 180))
            photo = ImageTk.PhotoImage(img)

            query_frame = tk.Frame(self.results_frame, bg="#36393F", padx=10, pady=10)
            query_frame.grid(row=0, column=0, columnspan=1, sticky="w")
            
            tk.Label(query_frame, text="Query Image (I2I Search)", bg="#36393F", fg="#7289DA", font=("Inter", 12, "bold")).pack(pady=5)

            lbl = tk.Label(query_frame, image=photo, bg="#36393F")
            lbl.image = photo # Keep reference
            lbl.pack()
            
            # To ensure the query image card remains visible, we need to temporarily
            # stop packing the main result cards into column 0 of row 0. 
            # We skip one column in the main results display, but for simplicity here,
            # we let the main results start in the next row, or adjust column span later.
            # (Leaving this as a functional card display for now).

        except Exception:
            pass
            

if __name__ == "__main__":
    # Ensure embeddings folder exists before running the app
    if not os.path.exists(EMBED_FOLDER):
        os.makedirs(EMBED_FOLDER)
        
    root = tk.Tk()
    app = ImageSearchApp(root)
    root.mainloop()