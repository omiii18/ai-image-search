import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import pickle
import torch
import clip
import faiss
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageDraw, ImageOps
import customtkinter as ctk
import json
import threading
import time
import shutil
import pillow_heif  # HEIC support
import sqlite3
import platform
import sys
import multiprocessing

def resource_path(relative_path):
    """Get absolute path for resources."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# --- Mac Bundle Fix ---
if getattr(sys, 'frozen', False):
    # Adjust CWD for bundled apps to save data locally
    # instead of system root
    bundle_dir = os.path.dirname(sys.executable)
    if ".app/Contents/MacOS" in bundle_dir:
        # Move up 3 levels to get out of the bundle to the folder it sits in
        os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(bundle_dir))))
    else:
        os.chdir(bundle_dir)

# Register HEIC opener
pillow_heif.register_heif_opener()

# --- Configuration ---
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B/32"  # Speed optimized
K_MATCHES = 12 # Number of results to display
SETTINGS_FILE = "settings.json"

# Import index builder
try:
    from index import build_index, get_catalog_paths
except ImportError:
    # Fallback if indexing fails
    def build_index(*args): return 0
    def get_catalog_paths(path): return path, path, path, path, path

class ModernButton(tk.Label):
    """Custom consistent button styling."""
    def __init__(self, parent, text, command, bg="#5865F2", fg="white", hover_bg="#4752C4", font=("Inter", 10, "bold"), padx=15, pady=8, min_width=0):
        self.bg_color = bg
        self.hover_bg = hover_bg
        self.command = command
        self.min_width = min_width
        
        super().__init__(parent, text=text, bg=bg, fg=fg, font=font, padx=padx, pady=pady, cursor="hand2")
        
        # Bind events
        
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
        self.root.geometry("1250x700")
        self.root.eval('tk::PlaceWindow . center') # Center window

        # --- State ---
        self.settings = self._load_settings()
        self.image_folder_path = ""

        
        # --- AI ---
        self.model = None
        self.preprocess = None
        self.faiss_index = None
        self.filenames = []
        
        # --- GUI ---
        self.status_text = tk.StringVar(value="Initializing...")
        self.query_image_path = tk.StringVar(value="")
        self.search_text_var = tk.StringVar()
        
        # --- Suggestions ---
        self.DEFAULT_KEYWORDS = ["dance", "night drives", "parties", "trips", "week's memory", 
                                     "beach", "food", "animal", "sunset", "mountain", "car", "person"]
        self.SUGGESTION_KEYWORDS = self._load_keywords()
        
        # Build the main GUI layout
        self._build_ui()
        
        self.create_dashboard_view()
        
        # Non-blocking AI init
        threading.Thread(target=self._initialize_ai, daemon=True).start()

    def _load_keywords(self):
        """Load keywords from file."""
        if self.image_folder_path:
            try:
                _, _, _, keywords_file, _ = get_catalog_paths(self.image_folder_path)
                if os.path.exists(keywords_file):
                    with open(keywords_file, 'r') as f:
                        return json.load(f)
            except:
                pass
        return self.DEFAULT_KEYWORDS

    def _load_settings(self):
        """Load settings."""
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {"image_folder_path": "", "recent_folders": []}
            
    def _save_settings(self):
        """Save settings."""
        self.settings["image_folder_path"] = self.image_folder_path
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(self.settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")

    def get_recent_folders(self):
        return self.settings.get("recent_folders", [])

    def save_recent_folder(self, new_path):
        recent = self.get_recent_folders()
        if new_path in recent:
            recent.remove(new_path)
        recent.insert(0, new_path)
        self.settings["recent_folders"] = recent[:6]
        self._save_settings()

    def _initialize_ai(self):
        """Load AI model."""
        try:
            self.status_text.set("1. Loading AI MODEL ...")
            self.model, self.preprocess = clip.load(MODEL_NAME, device=DEVICE)
            self.status_text.set("2. Model Loaded. Checking Library...")
            
            # --- Check/Index photos ---
            if self.image_folder_path and os.path.isdir(self.image_folder_path):
                self._check_and_index_photos()
            else:
                self.status_text.set("Ready. Use 'Select Folder' to configure your photos.")
                
        except Exception as e:
            self.status_text.set(f"FATAL ERROR: Initialization failed. Check dependencies. {e}")
            messagebox.showerror("Fatal Error", str(e))

    def _update_progress(self, current, total, message):
        """Update UI progress."""
        percent = int((current / total) * 100)
        self.status_text.set(f"{message} ({percent}%)")
        self.root.update_idletasks() # Force UI refresh

    def _check_and_index_photos(self):
        """Check files and update index."""
        if not self.image_folder_path or not self.model: return

        # 1. Get file count in user folder
        try:
            image_files = [f for f in os.listdir(self.image_folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".heic"))]
            current_file_count = len(image_files)
        except FileNotFoundError:
            self.status_text.set("Error: Photo folder not found. Please re-select.")
            return
            
        # 2. Load existing info
        indexed_count = 0
        try:
            _, index_file, _, _, _ = get_catalog_paths(self.image_folder_path)
            temp_index = faiss.read_index(index_file)
            indexed_count = temp_index.ntotal
        except Exception:
            pass 

        if current_file_count == 0:
            self.status_text.set("Ready. No images found in selected folder.")
        elif current_file_count > indexed_count:
            self.status_text.set(f"3. Found {current_file_count - indexed_count} new images. Auto-indexing...")
            
            # --- Rebuild Index ---
            try:
                # Progress callback
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
        
        # 4. Load final index
        self._load_search_index()
        self.status_text.set(f"Ready. {self.faiss_index.ntotal if self.faiss_index else 0} images searchable.")
        
    def _load_search_index(self):
        """Load FAISS index."""
        try:
            _, index_file, mapping_file, _, _ = get_catalog_paths(self.image_folder_path)
            self.faiss_index = faiss.read_index(index_file)
            
            # --- Dimension Check ---
            # ViT-B/32 has 512, ViT-L/14 has 768.
            expected_dim = 512 
            if MODEL_NAME == "ViT-L/14": expected_dim = 768
            
            if self.faiss_index.d != expected_dim:
                print(f"Dimension Mismatch! Index: {self.faiss_index.d}, Model: {expected_dim}. Re-indexing required.")
                self.faiss_index = None
                self.filenames = []
                self.status_text.set("Model Changed. Please CLICK 'Re-Index' to update.")
                return

            with open(mapping_file, "rb") as f:
                self.filenames = pickle.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")
            self.faiss_index = None
            self.filenames = []

    # --- Dashboard UI ---

    def create_dashboard_view(self):
        # Hide main interface
        if hasattr(self, 'main_container'):
            self.main_container.pack_forget()

        self.dashboard_frame = ctk.CTkFrame(self.root, fg_color=self.colors["bg"])
        self.dashboard_frame.pack(fill="both", expand=True)

        # Header
        header_label = ctk.CTkLabel(self.dashboard_frame, text="Welcome to DeepSearch AI", font=("Outfit", 28, "bold"), text_color=self.colors["text"])
        header_label.pack(pady=(60, 10))

        subtitle_label = ctk.CTkLabel(self.dashboard_frame, text="Select a recent folder or open a new one to start searching.", font=("Inter", 14), text_color=self.colors["subtext"])
        subtitle_label.pack(pady=(0, 40))

        # Grid Container
        grid_frame = ctk.CTkFrame(self.dashboard_frame, fg_color="transparent")
        grid_frame.pack(padx=40, pady=20)

        recent_folders = self.get_recent_folders()

        for index, folder_path in enumerate(recent_folders):
            row = index // 3
            col = index % 3

            folder_name = os.path.basename(os.path.normpath(folder_path))
            
            # Truncate path if too long
            display_path = folder_path if len(folder_path) < 35 else "..." + folder_path[-32:]
            display_text = f"{folder_name}\n\n{display_path}"

            btn = ctk.CTkButton(
                grid_frame, 
                text=display_text, 
                height=100,
                width=260,
                corner_radius=10, 
                fg_color="#333333", 
                hover_color="#444444",
                font=("Inter", 14, "bold"),
                command=lambda p=folder_path: self.load_selected_folder(p)
            )
            btn.grid(row=row, column=col, padx=15, pady=15, sticky="nsew")

        # "Browse for New Folder..." tile
        new_row = len(recent_folders) // 3
        new_col = len(recent_folders) % 3

        browse_btn = ctk.CTkButton(
            grid_frame, 
            text="➕ Browse for New Folder...", 
            height=100,
            width=260,
            corner_radius=10, 
            fg_color=self.colors["accent"], 
            hover_color="#584ab8",
            font=("Inter", 14, "bold"),
            command=self._select_folder_from_dashboard
        )
        browse_btn.grid(row=new_row, column=new_col, padx=15, pady=15, sticky="nsew")

    def _select_folder_from_dashboard(self):
        folder_selected = filedialog.askdirectory(title="Select Photo Library")
        if folder_selected:
            self.load_selected_folder(folder_selected)

    def load_selected_folder(self, folder_path):
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            messagebox.showerror("Error", "Folder no longer exists.")
            # Remove from recent folders and refresh
            recent = self.get_recent_folders()
            if folder_path in recent:
                recent.remove(folder_path)
                self.settings["recent_folders"] = recent
                self._save_settings()
                self.dashboard_frame.destroy()
                self.create_dashboard_view()
            return
            
        self.save_recent_folder(folder_path)
        self.image_folder_path = folder_path
        self.dir_label.config(text=self._get_display_path())
        self._save_settings()

        self.dashboard_frame.pack_forget()
        self.main_container.pack(fill="both", expand=True)

        self.reindex_thread()

    # --- Main UI ---
    
    def _build_ui(self):
        # Modern Dark Theme Aesthetics
        self.colors = {
            "bg": "#121212",
            "header": "#1E1E1E",
            "card": "#252525",
            "accent": "#6C5CE7",
            "text": "#E1E1E6",
            "subtext": "#A1A1AA",
            "danger": "#FF5252",
            "success": "#00E676",
            "input_bg": "#2C2C2C"
        }
        
        self.root.config(bg=self.colors["bg"])
        
        # Main Layout Container
        self.main_container = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_container.pack(fill="both", expand=True)

        # 1. Header Section
        self._build_header(self.main_container)
        
        # 2. Search & Filter Section
        self._build_search_area(self.main_container)

        # 3. Status Bar
        status_frame = tk.Frame(self.main_container, bg=self.colors["bg"], padx=20, pady=5)
        status_frame.pack(fill="x")
        tk.Label(status_frame, textvariable=self.status_text, bg=self.colors["bg"], fg=self.colors["subtext"], font=("Inter", 10)).pack(anchor="w")

        # 4. Results Area (Scrollable)
        self._build_results_area(self.main_container)


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
        
        # Search Box Container
        input_container = tk.Frame(search_frame, bg=self.colors["input_bg"], padx=10, pady=5)
        input_container.pack(fill="x", expand=True)

        # Icon/Label
        tk.Label(input_container, text="🔍", bg=self.colors["input_bg"], fg=self.colors["subtext"], font=("Inter", 14)).pack(side="left", padx=5)

        # Text Input
        self.search_entry = tk.Entry(input_container, textvariable=self.search_text_var, 
                                     font=("Inter", 14), bd=0, relief=tk.FLAT, 
                                     bg=self.colors["input_bg"], fg=self.colors["text"], 
                                     insertbackground=self.colors["text"])
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
        # Scrolling Canvas
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
        self._bind_mousewheel(self.root) # Global scroll

        # Layout management
        self.results_frame.bind("<Configure>", self._on_frame_configure)
        self.results_canvas.bind("<Configure>", self._on_canvas_configure)

    def _bind_mousewheel(self, widget):
        """Cross-platform scroll binding."""
        widget.bind("<MouseWheel>", self._on_mousewheel)
        widget.bind("<Button-4>", self._on_mousewheel) # Linux Scroll Up
        widget.bind("<Button-5>", self._on_mousewheel) # Linux Scroll Down

    def _on_mousewheel(self, event):
        """Handle scroll."""
        if platform.system() == "Darwin": # macOS
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
        """Refresh suggestion buttons."""
        for widget in self.suggestion_buttons_frame.winfo_children():
            widget.destroy()
            
        tk.Label(self.suggestion_buttons_frame, text="Quick Tags:", bg=self.colors["bg"], fg=self.colors["subtext"], font=("Inter", 10, "bold")).pack(side="left", padx=(0, 10))
        
        for keyword in self.SUGGESTION_KEYWORDS[:8]: # Show top 8 keywords
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
            self.save_recent_folder(folder_selected)
            self.dir_label.config(text=self._get_display_path())
            self._save_settings()
            self.reindex_thread()
            
    def _clear_cache(self, silent=False):
        if not silent and not messagebox.askyesno("Clear Cache", "Delete all index data for this folder? You will need to re-index."):
            return
        try:
            if self.image_folder_path:
                catalog_dir, _, _, _, _ = get_catalog_paths(self.image_folder_path)
                if os.path.exists(catalog_dir): shutil.rmtree(catalog_dir)
            
            self.faiss_index = None
            self.filenames = []
            self.SUGGESTION_KEYWORDS = self.DEFAULT_KEYWORDS
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
        
        # --- Get Query Vector ---
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
        if text_query and self.image_folder_path:
             try:
                _, _, _, _, ocr_db_file = get_catalog_paths(self.image_folder_path)
                if os.path.exists(ocr_db_file):
                    conn = sqlite3.connect(ocr_db_file)
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
        
        # If Query Image, show it
        row_offset = 0
        if query_image_path:
            self._render_card(query_image_path, "Query Image", "Source", 0, 0, highlight=True)
            row_offset = 1
        
        for index, (filename, score) in enumerate(results):
            path = os.path.join(self.image_folder_path, filename)
            row = (index // columns) + row_offset
            col = index % columns
            self._render_card(path, filename, f"{score:.1f}% Match", row, col)
            
        # Update scroll area
        self.results_frame.update_idletasks()
        self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _render_card(self, path, title, subtitle, row, col, highlight=False):
        """Create result card."""
        bg_color = self.colors["card"] if not highlight else "#3D2C4D"
        
        card = tk.Frame(self.results_frame, bg=bg_color, padx=10, pady=10)
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # Load Image
        try:
            catalog_dir, _, _, _, _ = get_catalog_paths(self.image_folder_path)
            THUMBNAIL_FOLDER = os.path.join(catalog_dir, "thumbnails")
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

            # Create rounded mask
            photo = ImageTk.PhotoImage(img)
            
            lbl_img = tk.Label(card, image=photo, bg=bg_color)
            lbl_img.image = photo # Retain reference
            lbl_img.pack()
            
            tk.Label(card, text=title[:20], bg=bg_color, fg=self.colors["text"], font=("Inter", 9, "bold")).pack(pady=(5,0))
            tk.Label(card, text=subtitle, bg=bg_color, fg=self.colors["accent"] if not highlight else "#fff", font=("Inter", 9)).pack()
            
        except Exception as e:
            tk.Label(card, text="Error", bg=bg_color, fg="red").pack()

    def _on_closing(self):
        """Exit cleanup."""
        try:
            # Reset folder path on exit
            self.image_folder_path = ""
            self._save_settings()
            
            # cache clear
            self._clear_cache(silent=True)
            print("Settings reset and cache cleared on exit.")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = ImageSearchApp(root)
    # Bind close event
    root.protocol("WM_DELETE_WINDOW", app._on_closing)
    root.mainloop()