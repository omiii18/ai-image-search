import pickle
import torch
import clip
import os
import faiss
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import json
import threading
import time
import shutil
import pillow_heif  # Import the library
import sqlite3
import platform

# Register HEIC opener - This makes Image.open() work with HEIC automatically!
pillow_heif.register_heif_opener()
# --- ENVIRONMENT FIX (OMP Error) ---
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# --- CONFIGURATION ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"  # Changed to Base model for SPEED
K_MATCHES = 12  # Number of search results to return
EMBED_FOLDER = "embeddings"
INDEX_FILE = os.path.join(EMBED_FOLDER, "faiss.index")
MAPPING_FILE = os.path.join(EMBED_FOLDER, "mapping.pkl")
KEYWORDS_FILE = os.path.join(EMBED_FOLDER, "keywords.json")
OCR_DB_FILE = os.path.join(EMBED_FOLDER, "ocr.db")
SETTINGS_FILE = "settings.json"

# Load the external index builder script
try:
    from index import build_index 
except ImportError:
    # Fallback if index.py is missing or has issues (for robustness)
    def build_index(*args): return 0

class ModernButton(tk.Label):
    """
    A custom button using tk.Label to ensure consistent styling 
    across platforms (especially macOS where buttons are restrictive).
    """
    def __init__(self, parent, text, command, bg="#5865F2", fg="white", hover_bg="#4752C4", font=("Inter", 10, "bold"), padx=15, pady=8, min_width=0):
        self.bg_color = bg
        self.hover_bg = hover_bg
        self.command = command
        self.min_width = min_width
        
        super().__init__(parent, text=text, bg=bg, fg=fg, font=font, padx=padx, pady=pady, cursor="hand2")
        
        # Rounded corners visual trick (optional, keeping it simple for now with just colors)
        # To make it look more button-like:
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def on_enter(self, e):
        self.config(bg=self.hover_bg)

    def on_leave(self, e):
        self.config(bg=self.bg_color)

    def on_click(self, e):
        if self.command:
            self.command()

class ImageSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepSearch AI Photo Library")
        self.root.geometry("1100x800")
        self.root.eval('tk::PlaceWindow . center') # Center on screen

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
        self.search_text_var = tk.StringVar()
        
        # --- Suggestions (User Guide) ---
        self.DEFAULT_KEYWORDS = ["dance", "night drives", "parties", "trips", "week's memory", 
                                     "beach", "food", "animal", "sunset", "mountain", "car", "person"]
        self.SUGGESTION_KEYWORDS = self._load_keywords()
        
        # Build the main GUI layout
        self._build_ui()
        
        # Start the AI/Indexing process in a separate thread (Non-blocking startup)
        threading.Thread(target=self._initialize_ai, daemon=True).start()

    def _load_keywords(self):
        """Loads dynamic keywords from file or returns defaults."""
        if os.path.exists(KEYWORDS_FILE):
            try:
                with open(KEYWORDS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return self.DEFAULT_KEYWORDS

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
            self.status_text.set("1. Loading AI MODEL ...")
            self.model, self.preprocess = clip.load(MODEL_NAME, device=DEVICE)
            self.status_text.set("2. Model Loaded. Checking Library...")
            
            # --- Auto-indexing check (runs indexing if needed) ---
            if self.image_folder_path and os.path.isdir(self.image_folder_path):
                self._check_and_index_photos()
            else:
                self.status_text.set("Ready. Use 'Select Folder' to configure your photos.")
                
        except Exception as e:
            self.status_text.set(f"FATAL ERROR: Initialization failed. Check dependencies. {e}")
            messagebox.showerror("Fatal Error", str(e))

    def _update_progress(self, current, total, message):
        """Callback to update the UI status text from the indexer."""
        percent = int((current / total) * 100)
        self.status_text.set(f"{message} ({percent}%)")
        self.root.update_idletasks() # Force UI refresh

    def _check_and_index_photos(self):
        """Checks for new files and updates/loads the FAISS index."""
        if not self.image_folder_path or not self.model: return

        # 1. Get file count in user folder
        try:
            image_files = [f for f in os.listdir(self.image_folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".heic"))]
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
                # Adding progress callback connection here!
                final_count = build_index(
                    self.image_folder_path, 
                    self.model, 
                    self.preprocess, 
                    DEVICE,
                    progress_callback=self._update_progress
                )
                self.status_text.set(f"4. Indexing complete. {final_count} images indexed.")
                
                # Reload keywords and refresh UI
                self.SUGGESTION_KEYWORDS = self._load_keywords()
                self.root.after(0, self._refresh_suggestions)
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
            
            # --- CRITICAL FIX: Dimension Check ---
            # If the loaded index has a different dimension than the current model,
            # we MUST invalidate it to prevent crashing.
            # ViT-B/32 has 512 dimensions. ViT-L/14 has 768.
            expected_dim = 512 # ViT-B/32
            if MODEL_NAME == "ViT-L/14": expected_dim = 768
            
            if self.faiss_index.d != expected_dim:
                print(f"Dimension Mismatch! Index: {self.faiss_index.d}, Model: {expected_dim}. Re-indexing required.")
                self.faiss_index = None
                self.filenames = []
                self.status_text.set("Model Changed. Please CLICK 'Re-Index' to update.")
                return

            with open(MAPPING_FILE, "rb") as f:
                self.filenames = pickle.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")
            self.faiss_index = None
            self.filenames = []

    # --- UI COMPONENTS & HANDLERS ---
    
    def _build_ui(self):
        # Modern Dark Theme Aesthetics
        self.colors = {
            "bg": "#121212",        # Very dark grey (almost black)
            "header": "#1E1E1E",    # Slightly lighter grey
            "card": "#252525",      # Card background
            "accent": "#6C5CE7",    # Blurple/Soft Purple
            "text": "#E1E1E6",      # Off-white
            "subtext": "#A1A1AA",   # Light grey
            "danger": "#FF5252",
            "success": "#00E676",
            "input_bg": "#2C2C2C"
        }
        
        self.root.config(bg=self.colors["bg"])
        
        # Main Layout Container
        main_container = tk.Frame(self.root, bg=self.colors["bg"])
        main_container.pack(fill="both", expand=True)

        # 1. Header Section
        self._build_header(main_container)
        
        # 2. Search & Filter Section
        self._build_search_area(main_container)

        # 3. Status Bar
        status_frame = tk.Frame(main_container, bg=self.colors["bg"], padx=20, pady=5)
        status_frame.pack(fill="x")
        tk.Label(status_frame, textvariable=self.status_text, bg=self.colors["bg"], fg=self.colors["subtext"], font=("Inter", 10)).pack(anchor="w")

        # 4. Results Area (Scrollable)
        self._build_results_area(main_container)


    def _build_header(self, parent):
        header_frame = tk.Frame(parent, bg=self.colors["header"], pady=15, padx=20)
        header_frame.pack(side="top", fill="x")
        
        # Title
        title_label = tk.Label(header_frame, text="DeepSearch AI", bg=self.colors["header"], fg=self.colors["text"], font=("Outfit", 18, "bold"))
        title_label.pack(side="left")
        
        # Settings Group
        settings_frame = tk.Frame(header_frame, bg=self.colors["header"])
        settings_frame.pack(side="right")
        
        # Current Folder Display
        self.dir_label = tk.Label(settings_frame, text=self._get_display_path(), 
                                  bg=self.colors["header"], fg=self.colors["subtext"], font=("Inter", 9), anchor="e")
        self.dir_label.pack(side="left", padx=(0, 15))

        # Buttons
        ModernButton(settings_frame, "Change Folder", self._select_folder, bg="#2D3436", hover_bg="#636e72", font=("Inter", 9)).pack(side="left", padx=5)
        ModernButton(settings_frame, "Re-Index", self.reindex_thread, bg=self.colors["accent"], hover_bg="#584ab8", font=("Inter", 9, "bold")).pack(side="left", padx=5)
        ModernButton(settings_frame, "Clean", self._clear_cache, bg=self.colors["danger"], hover_bg="#d32f2f", font=("Inter", 9)).pack(side="left", padx=5)


    def _build_search_area(self, parent):
        search_frame = tk.Frame(parent, bg=self.colors["bg"], pady=20, padx=20)
        search_frame.pack(fill="x")
        
        # Search Box Container (Rounded look via Frame)
        input_container = tk.Frame(search_frame, bg=self.colors["input_bg"], padx=10, pady=5)
        input_container.pack(fill="x", expand=True)

        # Icon/Label
        tk.Label(input_container, text="🔍", bg=self.colors["input_bg"], fg=self.colors["subtext"], font=("Inter", 14)).pack(side="left", padx=5)

        # Text Input
        self.search_entry = tk.Entry(input_container, textvariable=self.search_text_var, 
                                     font=("Inter", 14), bd=0, relief=tk.FLAT, 
                                     bg=self.colors["input_bg"], fg=self.colors["text"], 
                                     insertbackground=self.colors["text"]) # Caret color
        self.search_entry.pack(side="left", fill="x", expand=True, padx=5)
        # self.search_entry.bind("<Return>", lambda e: self.search_thread())
        self.search_entry.bind("<KeyRelease>", self._show_suggestions) 

        # Browse Image Button
        ModernButton(input_container, "📷 Image", self._select_image_query, bg=self.colors["input_bg"], hover_bg="#3D3D3D", fg=self.colors["accent"]).pack(side="right")
        
        # Submit Button
        ModernButton(input_container, "Search", self.search_thread, bg=self.colors["accent"], hover_bg="#584ab8").pack(side="right", padx=10)

        # Suggestion Tags
        self.suggestion_buttons_frame = tk.Frame(search_frame, bg=self.colors["bg"], pady=10)
        self.suggestion_buttons_frame.pack(fill="x")
        self._refresh_suggestions()

        # Suggestion Listbox (Floating)
        self.suggestions_listbox = tk.Listbox(parent, height=5, selectmode=tk.SINGLE, 
                                          font=("Inter", 12), relief=tk.FLAT, bd=0,
                                          background="#333333", fg=self.colors["text"], highlightthickness=0)
        self.suggestions_listbox.place_forget()

    def _build_results_area(self, parent):
        # Canvas for scrolling
        self.results_canvas = tk.Canvas(parent, bg=self.colors["bg"], borderwidth=0, highlightthickness=0)
        self.results_canvas.pack(side="left", fill="both", expand=True, padx=20, pady=10)
        
        # Scrollbar
        v_scrollbar = tk.Scrollbar(parent, orient="vertical", command=self.results_canvas.yview)
        v_scrollbar.pack(side="right", fill="y")
        self.results_canvas.configure(yscrollcommand=v_scrollbar.set)
        
        # Content Frame
        self.results_frame = tk.Frame(self.results_canvas, bg=self.colors["bg"])
        self.canvas_window = self.results_canvas.create_window((0, 0), window=self.results_frame, anchor="nw")
        
        # BINDINGS FOR SCROLLING
        self._bind_mousewheel(self.results_canvas)
        self._bind_mousewheel(self.results_frame)
        self._bind_mousewheel(self.root) # Bind root to catch scroll anywhere

        # Layout management
        self.results_frame.bind("<Configure>", self._on_frame_configure)
        self.results_canvas.bind("<Configure>", self._on_canvas_configure)

    def _bind_mousewheel(self, widget):
        """Binds mousewheel events for Windows, Mac, and Linux."""
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel) # Linux Scroll Up
        widget.bind("<Button-5>", self._on_mousewheel) # Linux Scroll Down

    def _on_mousewheel(self, event):
        """Handles scroll events cross-platform."""
        if platform.system() == "Darwin": # macOS
             # Delta is usually larger on Mac, so divide
             self.results_canvas.yview_scroll(int(-1*(event.delta)), "units")
        elif event.num == 4: # Linux Up
            self.results_canvas.yview_scroll(-1, "units")
        elif event.num == 5: # Linux Down
            self.results_canvas.yview_scroll(1, "units")
        else: # Windows
            self.results_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_frame_configure(self, event):
        """Reset the scroll region to encompass the inner frame."""
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Expand the inner frame to fill the canvas width."""
        canvas_width = event.width
        self.results_canvas.itemconfig(self.canvas_window, width=canvas_width)

    def _refresh_suggestions(self):
        """Re-populates the suggestion buttons."""
        for widget in self.suggestion_buttons_frame.winfo_children():
            widget.destroy()
            
        tk.Label(self.suggestion_buttons_frame, text="Quick Tags:", bg=self.colors["bg"], fg=self.colors["subtext"], font=("Inter", 10, "bold")).pack(side="left", padx=(0, 10))
            
        for keyword in self.SUGGESTION_KEYWORDS[:8]: # Show top 8
            # Using ModernButton instead of tk.Button for visibility
            btn = ModernButton(self.suggestion_buttons_frame, text=keyword, command=lambda k=keyword: self._quick_search(k), 
                            bg="#2D2D2D", fg=self.colors["text"], hover_bg="#3D3D3D", padx=10, pady=4, font=("Inter", 9))
            btn.pack(side="left", padx=4)

    def _get_display_path(self):
        if not self.image_folder_path: return "No Folder Selected"
        return f"...{self.image_folder_path[-30:]}" if len(self.image_folder_path) > 30 else self.image_folder_path
        
    def _select_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Photo Library")
        if folder_selected:
            self.image_folder_path = folder_selected
            self.dir_label.config(text=self._get_display_path())
            self._save_settings()
            self.reindex_thread()
            
    def _clear_cache(self, silent=False):
        if not silent and not messagebox.askyesno("Clear Cache", "Delete all index data? You will need to re-index."):
            return
        try:
            if os.path.exists(EMBED_FOLDER): shutil.rmtree(EMBED_FOLDER)
            if os.path.exists(".cache"): shutil.rmtree(".cache")
            self.faiss_index = None
            self.filenames = []
            self.SUGGESTION_KEYWORDS = self.DEFAULT_KEYWORDS
            os.makedirs(EMBED_FOLDER, exist_ok=True)
            self._refresh_suggestions()
            self._display_results([])
            if not silent: self.status_text.set("Cache cleared.")
        except Exception as e:
            if not silent: messagebox.showerror("Error", str(e))
            
    def _quick_search(self, query):
        self.search_text_var.set(query)
        self.search_thread()

    def _show_suggestions(self, event):
        if self.query_image_path.get():
            self.suggestions_listbox.place_forget()
            return
        prefix = self.search_text_var.get().lower()
        self.suggestions_listbox.delete(0, tk.END)
        if not prefix:
            self.suggestions_listbox.place_forget()
            return
        suggestions = [word for word in self.SUGGESTION_KEYWORDS if word.lower().startswith(prefix)]
        if suggestions:
            # Position listbox
            entry_x = self.search_entry.winfo_rootx() - self.root.winfo_rootx()
            entry_y = self.search_entry.winfo_rooty() - self.root.winfo_rooty() + self.search_entry.winfo_height() + 5
            self.suggestions_listbox.place(x=entry_x, y=entry_y, width=self.search_entry.winfo_width())
            for s in suggestions: self.suggestions_listbox.insert(tk.END, s)
            self.suggestions_listbox.bind("<<ListboxSelect>>", self._select_suggestion)
        else:
            self.suggestions_listbox.place_forget()

    def _select_suggestion(self, event):
        if self.suggestions_listbox.curselection():
            self.search_text_var.set(self.suggestions_listbox.get(self.suggestions_listbox.curselection()))
            self.suggestions_listbox.place_forget()
            self.search_thread()
            
    def _select_image_query(self):
        filepath = filedialog.askopenfilename(title="Select Query Image", filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.heic")])
        if filepath:
            self.query_image_path.set(filepath)
            self.search_text_var.set("")
            self.search_thread()

    def search_thread(self):
        # Clear conflicts
        if self.search_text_var.get(): self.query_image_path.set("")
        threading.Thread(target=self._run_search, daemon=True).start()
        
    def reindex_thread(self):
        threading.Thread(target=self._check_and_index_photos, daemon=True).start()

    def _run_search(self):
        text_query = self.search_text_var.get()
        image_query = self.query_image_path.get()
        
        if not self.faiss_index:
            self.status_text.set("Error: Index not loaded.")
            return

        if not text_query and not image_query: return

        self.status_text.set(f"Searching...")
        
        # --- 1. Get Query Vector ---
        query_vector = None
        if image_query:
            try:
                img = Image.open(image_query).convert("RGB")
                img_tensor = self.preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    query_vector = self.model.encode_image(img_tensor)
            except Exception as e:
                self.status_text.set(f"Error: {e}")
                return
        elif text_query:
            text_token = clip.tokenize([text_query]).to(DEVICE)
            with torch.no_grad():
                query_vector = self.model.encode_text(text_token)

        query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)
        query_vector_np = query_vector.cpu().numpy().astype('float32')

        # --- OCR Search ---
        ocr_matches = {}
        if text_query and os.path.exists(OCR_DB_FILE):
             try:
                conn = sqlite3.connect(OCR_DB_FILE)
                c = conn.cursor()
                c.execute("SELECT filename FROM ocr_data WHERE text_content MATCH ?", (f"{text_query}",))
                for row in c.fetchall(): ocr_matches[row[0]] = 100.0
                conn.close()
             except: pass

        # --- FAISS Search ---
        D, I = self.faiss_index.search(query_vector_np, k=K_MATCHES)
        
        final_scores = ocr_matches.copy()
        for i in range(len(I[0])):
            idx = I[0][i]
            score = float(D[0][i]) 
            match_percentage = (1 - (score / 2.5)) * 100 # Adjusted scaling
            
            if match_percentage < 5.0: continue
            
            filename = self.filenames[idx]
            if filename not in final_scores:
                final_scores[filename] = match_percentage

        sorted_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:K_MATCHES]
        self.root.after(0, lambda: self._display_results(sorted_results, image_query))

    def _display_results(self, results, query_image_path=None):
        for widget in self.results_frame.winfo_children(): widget.destroy()

        columns = 4
        # Dynamic grid configuration based on width could go here, but fixed 4 is fine for now
        
        # If Query Image, show it
        row_offset = 0
        if query_image_path:
            self._render_card(query_image_path, "Query Image", "Source", 0, 0, highlight=True)
            row_offset = 1 # Push results down or shift them? Let's just create a separate section or reuse grid logic
             # Ideally we want the query image to "push" the flow, but grid makes that hard without advanced logic.
             # Simple approach: Standard results grid.
        
        for index, (filename, score) in enumerate(results):
            path = os.path.join(self.image_folder_path, filename)
            row = (index // columns) + row_offset
            col = index % columns
            self._render_card(path, filename, f"{score:.1f}% Match", row, col)
            
        # Update scroll area
        self.results_frame.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _render_card(self, path, title, subtitle, row, col, highlight=False):
        """Helper to create a unified result card."""
        bg_color = self.colors["card"] if not highlight else "#3D2C4D" # Dark purple if highlight
        
        card = tk.Frame(self.results_frame, bg=bg_color, padx=10, pady=10)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # Load Image
        try:
            THUMBNAIL_FOLDER = ".cache/thumbnails"
            os.makedirs(THUMBNAIL_FOLDER, exist_ok=True)
            thumb_path = os.path.join(THUMBNAIL_FOLDER, os.path.basename(path))

            img = None
            if os.path.exists(thumb_path):
                try: img = Image.open(thumb_path)
                except: pass
            
            if not img:
                img = Image.open(path)
                # Aspect Ratio Cover
                img = ImageOps.fit(img, (220, 160), Image.Resampling.LANCZOS)
                # Optional: Save thumbnail for next time
                # img.save(thumb_path) 

            # Create rounded mask (simulated by not drawing corners? No, standard PIL rect)
            photo = ImageTk.PhotoImage(img)
            
            lbl_img = tk.Label(card, image=photo, bg=bg_color)
            lbl_img.image = photo # Keep ref
            lbl_img.pack()
            
            tk.Label(card, text=title[:20], bg=bg_color, fg=self.colors["text"], font=("Inter", 9, "bold")).pack(pady=(5,0))
            tk.Label(card, text=subtitle, bg=bg_color, fg=self.colors["accent"] if not highlight else "#fff", font=("Inter", 9)).pack()
            
        except Exception as e:
            tk.Label(card, text="Error", bg=bg_color, fg="red").pack()

    def _on_closing(self):
        """Cleanup on exit."""
        try:
            # Auto-clear cache on exit as requested
            self._clear_cache(silent=True)
            print("Cache cleared on exit.")
        except:
            pass
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    if not os.path.exists(EMBED_FOLDER): os.makedirs(EMBED_FOLDER)
    root = tk.Tk()
    app = ImageSearchApp(root)
    # Bind close event
    root.protocol("WM_DELETE_WINDOW", app._on_closing)
    root.mainloop()