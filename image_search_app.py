import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys
import subprocess
import pickle
import torch
import clip
import faiss
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk, ImageDraw, ImageOps
import customtkinter as ctk
import json
import threading
import time
import shutil
import pillow_heif  # HEIC support
import sqlite3
import platform
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
        self.root.title("DeepSearch AI")
        self.root.geometry("1100x700")

        # --- State ---
        self.settings = self._load_settings()
        self.image_folder_path = ""
        
        # --- AI ---
        self.model = None
        self.preprocess = None
        self.faiss_index = None
        self.filenames = []
        
        # --- GUI State ---
        self.status_text = tk.StringVar(value="Initializing...")
        self.query_image_path = tk.StringVar(value="")
        self.search_text_var = tk.StringVar()
        
        # Link status_text to the UI label
        self.status_text.trace_add("write", lambda *args: self._sync_status_label())

        # --- Dynamic Suggestions Vocabulary ---
        self.candidate_tags = [
            "people", "nature", "vehicles", "documents", "screenshots", 
            "food", "buildings", "animals", "text", "night", 
            "indoor", "outdoor", "portraits", "landscapes", "art",
            "sports", "water", "sky", "flowers", "clothing",
            "furniture", "electronics", "tools", "toys", "signs",
            "drawings", "paintings", "diagrams", "memes", "concerts",
            "weddings", "mountains", "beaches", "forests", "cities"
        ]
        self.current_top_tags = ["search", "indexing..."]

        # Custom Theme Colors (Light Theme)
        self.colors = {
            "main_bg": "#F8F9FA",
            "sidebar_bg": "#FFFFFF",
            "accent": "#EF4444",      # Red
            "active_bg": "#FEE2E2",   # Light Red
            "text": "#1F2937",        # Dark Grey
            "subtext": "#6B7280",     # Medium Grey
            "border": "#E5E7EB",
            "success": "#10B981"      # Green
        }

        ctk.set_appearance_mode("Light")
        
        # Build the main GUI layout
        self._build_ui()
        
        # Non-blocking AI init
        threading.Thread(target=self._initialize_ai, daemon=True).start()

    def _sync_status_label(self):
        """Sync the trace from status_text to the CTkLabel."""
        if hasattr(self, "status_label"):
            self.status_label.configure(text=self.status_text.get())

    def update_quick_tags_ui(self, top_tags):
        """Rebuild the UI tag buttons dynamically with the top identified categories."""
        if not hasattr(self, "tags_frame"): return
        
        # 1. Clear existing buttons
        for widget in self.tags_frame.winfo_children():
            # Keep the "Quick Tags:" label intact
            if isinstance(widget, ctk.CTkButton):
                widget.destroy()
                
        # 2. Rebuild tags
        self.current_top_tags = top_tags
        for keyword in top_tags:
            btn = ctk.CTkButton(self.tags_frame, text=keyword, 
                                command=lambda k=keyword: self._quick_search(k),
                                fg_color="#F3F4F6", text_color=self.colors["text"], hover_color="#E5E7EB",
                                height=28, corner_radius=14, font=("Inter", 11))
            btn.pack(side="left", padx=5)

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
        progress = current / total
        self.progress_bar.set(progress)
        percent = int(progress * 100)
        self.status_label.configure(text=f"Indexing: {percent}%")
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
                
            # --- Generate Dynamic Tags ---
            threading.Thread(target=self.generate_dynamic_tags, daemon=True).start()
            
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
        self._save_settings()

        self.dashboard_frame.pack_forget()
        self.main_container.pack(fill="both", expand=True)

        self.reindex_thread()

    # --- Main UI Overhaul ---
    
    def _build_ui(self):
        """Build the modernized two-column layout."""
        self.root.configure(bg=self.colors["main_bg"])
        
        # 1. Main Layout: Sidebar and Content
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True)

        # 1a. Left Sidebar
        self.sidebar = ctk.CTkFrame(self.main_container, width=220, fg_color=self.colors["sidebar_bg"], corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Logo
        logo_label = ctk.CTkLabel(self.sidebar, text="DeepSearch AI", font=("Outfit", 20, "bold"), text_color=self.colors["text"])
        logo_label.pack(pady=(30, 40), padx=20, anchor="w")

        # Menu Items
        self.photos_btn = ctk.CTkButton(self.sidebar, text="  Photos", image=None, # Add icons later if needed
                                        anchor="w", font=("Inter", 13, "bold"),
                                        height=40, corner_radius=8,
                                        fg_color=self.colors["active_bg"], text_color=self.colors["accent"],
                                        hover_color=self.colors["active_bg"])
        self.photos_btn.pack(fill="x", padx=15, pady=5)

        recent_folders = self.get_recent_folders()
        dropdown_vals = recent_folders if recent_folders else ["No recent folders"]
        self.folders_dropdown = ctk.CTkOptionMenu(
            self.sidebar, 
            values=dropdown_vals,
            command=self.load_selected_folder,
            font=("Inter", 13),
            dropdown_font=("Inter", 12),
            height=40, corner_radius=8,
            fg_color=self.colors["sidebar_bg"], text_color=self.colors["text"],
            button_color=self.colors["sidebar_bg"], button_hover_color="#F3F4F6", dropdown_hover_color="#FEE2E2"
        )
        self.folders_dropdown.set("Recent Folders")
        self.folders_dropdown.pack(fill="x", padx=15, pady=5)

        # Sidebar Bottom: Indexing Status
        status_container = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        status_container.pack(side="bottom", fill="x", padx=20, pady=30)

        status_header = ctk.CTkFrame(status_container, fg_color="transparent")
        status_header.pack(fill="x")

        ctk.CTkLabel(status_header, text="Indexing Status", font=("Inter", 11, "bold"), text_color=self.colors["text"]).pack(side="left")
        self.live_marker = ctk.CTkLabel(status_header, text="Live", font=("Inter", 10, "bold"), text_color=self.colors["success"])
        self.live_marker.pack(side="right")

        self.status_label = ctk.CTkLabel(status_container, text="Ready", font=("Inter", 11), text_color=self.colors["subtext"])
        self.status_label.pack(anchor="w", pady=(5, 10))

        self.progress_bar = ctk.CTkProgressBar(status_container, height=6, progress_color=self.colors["accent"])
        self.progress_bar.pack(fill="x")
        self.progress_bar.set(1.0) # Default to full

        # 1b. Right Content Area
        self.content_area = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_area.pack(side="right", fill="both", expand=True)

        # 2. Top Header (Search & Actions)
        self._build_header(self.content_area)

        # 3. Main Display Area
        self.display_container = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.display_container.pack(fill="both", expand=True, padx=30, pady=(10, 30))

        # Title & Subtitle
        self.results_title = ctk.CTkLabel(self.display_container, text="Recent Matches", font=("Outfit", 24, "bold"), text_color=self.colors["text"])
        self.results_title.pack(anchor="w")
        
        self.results_subtitle = ctk.CTkLabel(self.display_container, text="Showing 0 images...", font=("Inter", 13), text_color=self.colors["subtext"])
        self.results_subtitle.pack(anchor="w", pady=(0, 20))

        # Scrollable Grid
        self.scroll_frame = ctk.CTkScrollableFrame(self.display_container, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True)
        # Configure columns for grid
        for i in range(4): self.scroll_frame.grid_columnconfigure(i, weight=1, pad=15)

    def _build_header(self, parent):
        header_container = ctk.CTkFrame(parent, fg_color="transparent")
        header_container.pack(fill="x", padx=30, pady=(30, 0))

        # Top row: Search Box & Actions
        search_row = ctk.CTkFrame(header_container, fg_color="transparent", height=45)
        search_row.pack(fill="x", pady=(0, 15))

        # Search Box Entry
        self.search_entry = ctk.CTkEntry(search_row, placeholder_text="Search photos by content...",
                                         textvariable=self.search_text_var,
                                         font=("Inter", 14), height=45, corner_radius=10,
                                         fg_color="#FFFFFF", border_color=self.colors["border"],
                                         text_color=self.colors["text"])
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.search_entry.bind("<Return>", lambda e: self.search_thread())
        
        # Attach Drag and Drop to Entry and Main Window
        try:
            self.root.drop_target_register(DND_FILES)
            self.root.dnd_bind('<<Drop>>', self.handle_image_drop)
            self.search_entry.drop_target_register(DND_FILES)
            self.search_entry.dnd_bind('<<Drop>>', self.handle_image_drop)
        except Exception: pass

        # New Buttons
        self.search_btn = ctk.CTkButton(search_row, text="Search", width=80,
                                         fg_color=self.colors["text"], text_color="#FFFFFF",
                                         hover_color="#374151", height=45, corner_radius=10,
                                         font=("Inter", 13, "bold"), command=self.search_thread)
        self.search_btn.pack(side="left", padx=(0, 10))

        self.image_search_btn = ctk.CTkButton(search_row, text="Image", width=80,
                                               fg_color="#FFFFFF", text_color=self.colors["text"],
                                               hover_color="#F3F4F6", height=45, corner_radius=10,
                                               border_width=1, border_color=self.colors["border"],
                                               font=("Inter", 13, "bold"), command=self._select_image_query)
        self.image_search_btn.pack(side="left", padx=(0, 20))

        # Action Buttons
        self.change_folder_btn = ctk.CTkButton(search_row, text="Change Folder", width=120,
                                               fg_color="#F3F4F6", text_color=self.colors["text"],
                                               hover_color="#E5E7EB", height=45, corner_radius=10,
                                               font=("Inter", 13, "bold"), command=self._select_folder)
        self.change_folder_btn.pack(side="left", padx=(0, 10))

        self.reindex_btn = ctk.CTkButton(search_row, text="Re-Index", width=100,
                                         fg_color=self.colors["accent"], text_color="#FFFFFF",
                                         hover_color="#DC2626", height=45, corner_radius=10,
                                         font=("Inter", 13, "bold"), command=self.reindex_thread)
        self.reindex_btn.pack(side="left")

        # Bottom row: Quick Tags
        self.tags_frame = ctk.CTkScrollableFrame(header_container, fg_color="transparent", height=40, orientation="horizontal")
        self.tags_frame.pack(fill="x")
        
        ctk.CTkLabel(self.tags_frame, text="Quick Tags:", font=("Inter", 12, "bold"), text_color=self.colors["subtext"]).pack(side="left", padx=(0, 10))
        
        for keyword in self.current_top_tags[:12]:
            btn = ctk.CTkButton(self.tags_frame, text=keyword, 
                                command=lambda k=keyword: self._quick_search(k),
                                fg_color="#F3F4F6", text_color=self.colors["text"], hover_color="#E5E7EB",
                                height=28, corner_radius=14, font=("Inter", 11))
            btn.pack(side="left", padx=5)

    def load_selected_folder(self, choice):
        if choice and choice != "No recent folders":
            if os.path.exists(choice):
                self.image_folder_path = choice
                self.save_recent_folder(choice)
                self._save_settings()
                self.reindex_thread()
            else:
                self.status_text.set("Error: Folder not found.")

    def _get_display_path(self):
        if not self.image_folder_path: return "No Folder Selected"
        return f"...{self.image_folder_path[-30:]}" if len(self.image_folder_path) > 30 else self.image_folder_path
        
    def _select_folder(self):
        folder_selected = filedialog.askdirectory(title="Select Photo Library")
        if folder_selected:
            self.image_folder_path = folder_selected
            self.save_recent_folder(folder_selected)
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
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, query)
        self.search_thread()

    def _select_image_query(self):
        filepath = filedialog.askopenfilename(title="Select Query Image", filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.heic")])
        if filepath:
            self.search_by_image(filepath)
            
    def handle_image_drop(self, event):
        """Handle Drag and Drop image files."""
        # Remove curly braces on Mac/Windows paths
        filepath = event.data.strip('{}') 
        
        # Validation
        valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".heic")
        if not filepath.lower().endswith(valid_exts):
            self.status_text.set("Error: Unsupported file type dropped. Please use images only.")
            return
            
        self.search_by_image(filepath)
        
    def reveal_file_in_os(self, file_path):
        """Cross-platform method to highlight a file in the OS native file explorer."""
        if not os.path.exists(file_path):
            self.status_text.set(f"Error: File no longer exists: {os.path.basename(file_path)}")
            return
            
        try:
            if sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", "-R", file_path])
            elif sys.platform == "win32": # Windows
                subprocess.Popen(f'explorer /select,"{os.path.normpath(file_path)}"')
            else:                         # Linux
                subprocess.Popen(["xdg-open", os.path.dirname(file_path)])
        except Exception as e:
            self.status_text.set(f"Error opening file explorer: {str(e)}")

    def search_by_image(self, image_path):
        """Perform an isolated, direct FAISS image-to-image search."""
        if not self.faiss_index:
            self.status_text.set("Error: Index not loaded. Please select a folder.")
            return

        filename = os.path.basename(image_path)
        self.status_text.set(f"Searching by image: {filename}...")
        self.search_entry.delete(0, tk.END)
        self.search_entry.insert(0, f"[Searching by Image]") # requested logic
        self.query_image_path.set(image_path)
        
        # Isolate the search logic in a thread to prevent UI freezing
        threading.Thread(target=self._run_image_search_isolated, args=(image_path,), daemon=True).start()

    def _run_image_search_isolated(self, image_path):
        """The core internal logic strictly for full AI image-to-image search."""
        try:
            # 1. Load the image
            img = Image.open(image_path).convert("RGB")
            
            # 2. Preprocess for CLIP
            image_tensor = self.preprocess(img).unsqueeze(0).to(DEVICE)
            
            # 3. Get the vector
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                
            # 4. Convert to numpy & normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features_np = image_features.cpu().numpy().astype('float32')
            
            # 5. Search FAISS
            D, I = self.faiss_index.search(image_features_np, k=K_MATCHES)
            
            # 6. Map the indices back to file paths and scale match %
            results_map = {}
            for i in range(len(I[0])):
                idx = I[0][i]
                score = float(D[0][i])
                match_percentage = (1 - (score / 2.5)) * 100 # Original scaling metric
                
                if match_percentage < 5.0: continue
                
                mapped_filename = self.filenames[idx]
                results_map[mapped_filename] = match_percentage
                
            # 7. Sort
            sorted_results = sorted(results_map.items(), key=lambda x: x[1], reverse=True)[:K_MATCHES]
            
            # 8. Update the UI
            self.root.after(0, lambda: self.populate_grid(sorted_results, image_path))
            self.root.after(0, lambda: self.status_text.set(f"Found {len(sorted_results)} image matches."))
            
        except Exception as e:
            self.root.after(0, lambda err=e: self.status_text.set(f"Search Error: {err}"))

    def search_thread(self):
        # Do not use old text/image combined runner if image path is set
        if self.query_image_path.get() and "[Searching by Image]" in self.search_entry.get():
            return
            
        if self.search_text_var.get(): 
            self.query_image_path.set("")
            
        # self.search_entry.delete(0, tk.END) # Clear visual search bar on enterprise search
        threading.Thread(target=self._run_search, daemon=True).start()
        
    def generate_dynamic_tags(self):
        """Run Zero-Shot classification to detect prominent folder themes."""
        if not self.faiss_index or not self.model or self.faiss_index.ntotal == 0:
            return
            
        try:
            tag_scores = {}
            for tag in self.candidate_tags:
                # 1. Encode Word
                text_token = clip.tokenize([tag]).to(DEVICE)
                with torch.no_grad():
                    query_vector = self.model.encode_text(text_token)
                
                # Normalize & Numpy
                query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)
                query_vector_np = query_vector.cpu().numpy().astype('float32')
                
                # 2. Search FAISS for top 5
                D, _ = self.faiss_index.search(query_vector_np, k=5)
                
                # 3. Calculate Average distance (lower is better)
                avg_distance = float(np.mean(D[0]))
                tag_scores[tag] = avg_distance
                
            # 4. Sort and extract top 5-7
            # Lower L2 distance means higher similarity/relevance to the tag
            sorted_tags = sorted(tag_scores.items(), key=lambda x: x[1])
            top_tags = [t[0] for t in sorted_tags[:6]]
            
            # 5. Push UI update to main thread
            self.root.after(0, lambda: self.update_quick_tags_ui(top_tags))
            
        except Exception as e:
            print(f"Error generating context tags: {e}")
        
    def reindex_thread(self):
        threading.Thread(target=self._check_and_index_photos, daemon=True).start()

    def _run_search(self):
        text_query = self.search_text_var.get()
        
        if not self.faiss_index:
            self.status_text.set("Error: Index not loaded.")
            return

        if not text_query: return

        self.status_text.set(f"Searching...")
        
        # --- Get Query Vector ---
        # Extract query vector exclusively for TEXT in this function
        query_vector = None
        if text_query:
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
        self.root.after(0, lambda: self._display_results(sorted_results, None))

    # --- Results Rendering (Populate Grid) ---

    def _display_results(self, results, query_image_path=None):
        """Populate the grid with results."""
        self.populate_grid(results, query_image_path)

    def populate_grid(self, results_list, query_image_path=None):
        """Take results and place CTkFrame 'Cards' into the scrollable frame."""
        # Clear existing
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
            
        # Update Subtitle
        count = len(results_list)
        self.results_subtitle.configure(text=f"Showing {count} images...")

        columns = 4
        row_offset = 0
        
        # If Query Image, show it first
        if query_image_path:
            self._render_card(query_image_path, "Query Image", "Input Source", 0, 0, highlight=True)
            row_offset = 1
            
        for index, (filename, score) in enumerate(results_list):
            path = os.path.join(self.image_folder_path, filename)
            row = (index // columns) + row_offset
            col = index % columns
            self._render_card(path, filename, f"Match: {score:.1f}%", row, col)

    def _render_card(self, path, title, subtitle, row, col, highlight=False):
        """Create a modernized white card."""
        bg_color = self.colors["active_bg"] if highlight else self.colors["sidebar_bg"]
        
        card = ctk.CTkFrame(self.scroll_frame, fg_color=bg_color, corner_radius=12, border_width=1, border_color="#F3F4F6")
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # Load & Process Image for Card
        try:
            catalog_dir, _, _, _, _ = get_catalog_paths(self.image_folder_path)
            THUMBNAIL_FOLDER = os.path.join(catalog_dir, "thumbnails")
            thumb_path = os.path.join(THUMBNAIL_FOLDER, os.path.basename(path))

            # Try to get existing thumb or create one
            pil_img = None
            if os.path.exists(thumb_path):
                pil_img = Image.open(thumb_path)
            else:
                pil_img = Image.open(path)
                pil_img = ImageOps.pad(pil_img, (300, 300), color=bg_color)
            
            # Wrap in CTkImage
            ctk_img = ctk.CTkImage(light_image=pil_img, size=(160, 160))
            
            img_label = ctk.CTkLabel(card, image=ctk_img, text="", corner_radius=8, cursor="hand2")
            img_label.image = ctk_img # Ref
            img_label.pack(pady=(12, 5), padx=12)
            
            # Bind Click Event to Reveal File
            img_label.bind("<Button-1>", lambda event, p=path: self.reveal_file_in_os(p))
            
            # Metadata
            filename_label = ctk.CTkLabel(card, text=title[:18] + "..." if len(title) > 18 else title, 
                                          font=("Inter", 13, "bold"), text_color=self.colors["text"], cursor="hand2")
            filename_label.pack(pady=(5, 0), padx=12, anchor="w")
            filename_label.bind("<Button-1>", lambda event, p=path: self.reveal_file_in_os(p))
            
            # Bottom Info Row
            info_row = ctk.CTkFrame(card, fg_color="transparent")
            info_row.pack(fill="x", padx=12, pady=(2, 12))
            
            # Get file size if exists
            size_str = "Unknown"
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                size_str = f"{size_mb:.1f} MB"
            
            ctk.CTkLabel(info_row, text=size_str, font=("Inter", 11), text_color=self.colors["subtext"]).pack(side="left")
            
            match_color = self.colors["success"] if not highlight else self.colors["accent"]
            ctk.CTkLabel(info_row, text=subtitle, font=("Inter", 11, "bold"), text_color=match_color).pack(side="right")
            
        except Exception:
            ctk.CTkLabel(card, text="Image Error", text_color="red").pack(pady=40)

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
    try:
        # Initializing core OS Drag-and-Drop capability wrapper
        root = TkinterDnD.Tk()
    except Exception as e:
        print(f"TkinterDnD initialization failed, falling back to standard Tk. Error: {e}")
        root = tk.Tk()
        
    app = ImageSearchApp(root)
    # Bind close event
    root.protocol("WM_DELETE_WINDOW", app._on_closing)
    root.mainloop()