import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import sys

# Fix for running with pythonw.exe (no console) where sys.stdout/stderr are None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

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
from PIL.ExifTags import TAGS
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

def get_exif_metadata(image_path):
    """Safely extract basic EXIF data (like Date) for RAG context."""
    metadata = {}
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name in ['DateTime', 'DateTimeOriginal', 'Make', 'Model']:
                    metadata[tag_name] = str(value).strip('\x00')
    except Exception:
        pass
        
    if not metadata:
        try:
            mtime = os.path.getmtime(image_path)
            # Fallback to file modified time
            import time
            metadata['FileDate'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        except:
            pass
            
    return metadata




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
        self.auto_album_data = {}

        # --- Smart Auto-Album Definitions ---
        # Semantic anchor text for each virtual album archetype
        self.album_definitions = {
            'Portraits':  'a close-up portrait photograph of a person',
            'Night':      'dark night time photography with city lights',
            'Landscapes': 'a wide scenic landscape photograph of nature',
            'Documents':  'a scanned document or page of text',
            'Rainy':      'a photograph taken in the rain with wet surfaces',
            'Food':       'a photograph of food, meal, or dish on a table',
            'Pets':       'a photograph of a pet dog or cat',
        }

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

    def _generate_auto_albums(self):
        """Generate smart virtual albums using CLIP + FAISS with a dynamic threshold."""
        if not self.faiss_index or not self.model or not self.filenames:
            return
        
        total_images = self.faiss_index.ntotal
        if total_images == 0:
            return

        self.auto_album_data = {}

        # --- Determine dynamic threshold via semantic range ---
        # Sample a generic query to learn the distance distribution of this folder
        sample_token = clip.tokenize(["a photograph"]).to(DEVICE)
        with torch.no_grad():
            sample_vec = self.model.encode_text(sample_token)
        sample_vec = sample_vec / sample_vec.norm(dim=-1, keepdim=True)
        sample_np = sample_vec.cpu().numpy().astype('float32')
        
        k_sample = min(total_images, 50)
        D_sample, _ = self.faiss_index.search(sample_np, k=k_sample)
        
        # The top-20% boundary of this folder's distance range
        distances_sorted = np.sort(D_sample[0])
        cutoff_index = max(1, int(len(distances_sorted) * 0.20)) - 1
        distance_threshold = float(distances_sorted[cutoff_index])

        # --- Scan each archetype ---
        for album_name, anchor_text in self.album_definitions.items():
            text_tokens = clip.tokenize([anchor_text]).to(DEVICE)
            with torch.no_grad():
                query_vector = self.model.encode_text(text_tokens)
            query_vector = query_vector / query_vector.norm(dim=-1, keepdim=True)
            query_vector_np = query_vector.cpu().numpy().astype('float32')

            # Search top 20% of the folder or at least 20
            k_search = min(total_images, max(20, int(total_images * 0.20)))
            D, I = self.faiss_index.search(query_vector_np, k=k_search)

            valid_results = {}
            for i in range(len(I[0])):
                idx = I[0][i]
                if idx < 0 or idx >= len(self.filenames):
                    continue
                score = float(D[0][i])
                # Only accept images within the dynamic threshold
                if score > distance_threshold:
                    continue
                match_pct = max(0, (1 - (score / 2.5)) * 100)
                filename = self.filenames[idx]
                valid_results[filename] = match_pct

            # --- Dynamic Visibility: only keep if > 5 matches ---
            if len(valid_results) > 5:
                sorted_results = sorted(valid_results.items(), key=lambda x: x[1], reverse=True)[:K_MATCHES]
                self.auto_album_data[album_name] = sorted_results

        # Push UI update to main thread
        self.root.after(0, self._update_auto_albums_ui)

    def _update_auto_albums_ui(self):
        """Rebuild the Auto-Albums sidebar section."""
        for widget in self.auto_albums_frame.winfo_children():
            widget.destroy()

        if not self.auto_album_data:
            return

        # Section header
        ctk.CTkLabel(
            self.auto_albums_frame, text="Auto-Albums",
            font=("Inter", 11, "bold"), text_color=self.colors["subtext"]
        ).pack(anchor="w", pady=(0, 8))

        for album_name, results in self.auto_album_data.items():
            count = len(results)
            # Row frame for button + badge
            row = ctk.CTkFrame(self.auto_albums_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            album_btn = ctk.CTkButton(
                row,
                text=f"  {album_name}",
                anchor="w", font=("Inter", 12),
                height=32, corner_radius=6,
                fg_color="transparent", text_color=self.colors["text"],
                hover_color=self.colors["active_bg"],
                command=lambda name=album_name: self._on_album_click(name)
            )
            album_btn.pack(side="left", fill="x", expand=True)

            # Count badge
            badge = ctk.CTkLabel(
                row, text=str(count),
                font=("Inter", 10, "bold"), text_color="white",
                fg_color=self.colors["accent"], corner_radius=10,
                width=28, height=20
            )
            badge.pack(side="right", padx=(0, 5))

    def _on_album_click(self, album_name):
        """Filter the image grid to show only the selected auto-album."""
        if not self.faiss_index or album_name not in self.auto_album_data:
            return
        self.results_title.configure(text=f"Auto-Album: {album_name}")
        count = len(self.auto_album_data[album_name])
        self.results_subtitle.configure(text=f"Showing {count} images curated by semantic analysis.")
        self.scroll_frame._parent_canvas.yview_moveto(0)
        self.populate_grid(self.auto_album_data[album_name], None)

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
                                        hover_color=self.colors["active_bg"], command=self._show_photos_view)
        self.photos_btn.pack(fill="x", padx=15, pady=5)

        self.assistant_btn = ctk.CTkButton(self.sidebar, text="  Memory Assistant", image=None,
                                        anchor="w", font=("Inter", 13, "bold"),
                                        height=40, corner_radius=8,
                                        fg_color="transparent", text_color=self.colors["text"],
                                        hover_color=self.colors["active_bg"], command=self._show_assistant_view)
        self.assistant_btn.pack(fill="x", padx=15, pady=(0, 5))

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

        # Auto-Albums Section (scrollable for many albums)
        self.auto_albums_frame = ctk.CTkScrollableFrame(
            self.sidebar, fg_color="transparent", height=180,
            label_text="", scrollbar_button_color=self.colors["border"]
        )
        self.auto_albums_frame.pack(fill="x", padx=15, pady=(15, 5))

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

        # 1b. Right Content Area Container
        self.content_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_container.pack(side="right", fill="both", expand=True)

        self.photos_view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self.photos_view.pack(fill="both", expand=True)

        self.assistant_view = ctk.CTkFrame(self.content_container, fg_color="transparent")
        self._build_assistant_ui(self.assistant_view)

        # 2. Top Header (Search & Actions) inside photos_view
        self._build_header(self.photos_view)

        # 3. Main Display Area
        self.display_container = ctk.CTkFrame(self.photos_view, fg_color="transparent")
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

        # Smooth Scroll Binding
        self._setup_smooth_scroll()

        # 4. Search Footer
        self.footer_label = ctk.CTkLabel(self.display_container, 
                                        text="DeepSearch AI v1.1 | Developed with ❤️ by Omkar Karne", 
                                        font=("Inter", 10), text_color=self.colors["subtext"])
        self.footer_label.pack(side="bottom", pady=(10, 0))

    def _show_photos_view(self):
        self.assistant_btn.configure(fg_color="transparent", text_color=self.colors["text"])
        self.photos_btn.configure(fg_color=self.colors["active_bg"], text_color=self.colors["accent"])
        self.assistant_view.pack_forget()
        self.photos_view.pack(fill="both", expand=True)
        
    def _show_assistant_view(self):
        self.photos_btn.configure(fg_color="transparent", text_color=self.colors["text"])
        self.assistant_btn.configure(fg_color=self.colors["active_bg"], text_color=self.colors["accent"])
        self.photos_view.pack_forget()
        self.assistant_view.pack(fill="both", expand=True)

    def _build_assistant_ui(self, parent):
        title = ctk.CTkLabel(parent, text="Memory Assistant", font=("Outfit", 24, "bold"), text_color=self.colors["text"])
        title.pack(anchor="w", padx=30, pady=(30, 10))
        
        desc = ctk.CTkLabel(parent, text="Ask questions about your indexed photos. Powered by local LLM + EXIF data.", font=("Inter", 13), text_color=self.colors["subtext"])
        desc.pack(anchor="w", padx=30, pady=(0, 20))
        
        # Chat History Scrollable Frame
        self.chat_scroll_frame = ctk.CTkScrollableFrame(parent, fg_color="transparent", scrollbar_button_color=self.colors["border"])
        self.chat_scroll_frame.pack(fill="both", expand=True, padx=20, pady=(0, 5))
        self._bind_chat_scroll(self.chat_scroll_frame._parent_canvas)
        self._bind_chat_scroll(self.chat_scroll_frame._parent_frame)
        
        # Typing indicator (Hidden by default)
        self.typing_indicator = ctk.CTkLabel(parent, text="", font=("Inter", 12, "italic"), text_color=self.colors["subtext"])
        
        # Modern Input Bar
        input_frame = ctk.CTkFrame(parent, fg_color="transparent", height=50)
        input_frame.pack(fill="x", padx=30, pady=(0, 30))
        
        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text="e.g. When did I last go to the beach?", font=("Inter", 14), height=45, corner_radius=22, fg_color="#FFFFFF", border_color=self.colors["border"], border_width=1, text_color=self.colors["text"])
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.chat_input.bind("<Return>", lambda e: self._handle_chat_message())
        
        send_btn = ctk.CTkButton(input_frame, text="↑", width=45, fg_color=self.colors["text"], text_color="#FFFFFF", hover_color="#374151", height=45, corner_radius=22, font=("Inter", 18, "bold"), command=self._handle_chat_message)
        send_btn.pack(side="left")

    def _create_chat_bubble(self, text, role="User"):
        bubble_frame = ctk.CTkFrame(self.chat_scroll_frame, fg_color="transparent")
        bubble_frame.pack(fill="x", padx=10, pady=5)
        
        is_user = role == "User"
        bg_color = self.colors["accent"] if is_user else "#FFFFFF"
        text_color = "#FFFFFF" if is_user else self.colors["text"]
        border_color = self.colors["accent"] if is_user else self.colors["border"]
        border_width = 0 if is_user else 1
        
        inner_frame = ctk.CTkFrame(bubble_frame, fg_color=bg_color, corner_radius=15, border_width=border_width, border_color=border_color)
        inner_frame.pack(side="right" if is_user else "left", anchor="e" if is_user else "w", padx=5)
        
        msg_label = ctk.CTkLabel(inner_frame, text=text, font=("Inter", 13), text_color=text_color, justify="left", wraplength=450)
        msg_label.pack(padx=15, pady=10)
        
        self._bind_chat_scroll(bubble_frame)
        self._bind_chat_scroll(inner_frame)
        self._bind_chat_scroll(msg_label)
        
        self.root.after(50, lambda: self._scroll_chat_to_bottom())
        return msg_label, inner_frame

    def _on_chat_mousewheel(self, event):
        """Universal scroll handler — works on Mac (delta ±1) and Windows (delta ±120)."""
        try:
            if sys.platform == "darwin":
                scroll_units = int(-1 * event.delta)
            else:
                scroll_units = int(-1 * (event.delta / 120))
            self.chat_scroll_frame._parent_canvas.yview_scroll(scroll_units, "units")
        except Exception:
            pass

    def _bind_chat_scroll(self, widget):
        """Recursively bind mouse wheel to a widget AND every one of its children."""
        if sys.platform.startswith("win") or sys.platform == "darwin":
            widget.bind("<MouseWheel>", self._on_chat_mousewheel, add="+")
        else:
            widget.bind("<Button-4>", self._on_chat_mousewheel, add="+")
            widget.bind("<Button-5>", self._on_chat_mousewheel, add="+")
        # Walk ALL internal children (including Tkinter internals inside CTk widgets)
        for child in widget.winfo_children():
            self._bind_chat_scroll(child)

    def _scroll_chat_to_bottom(self):
        try:
            self.chat_scroll_frame.update_idletasks()
            self.chat_scroll_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass
            
    def _show_typing(self, message="Thinking..."):
        self.typing_indicator.configure(text=message)
        self.typing_indicator.pack(anchor="w", padx=45, pady=(0, 10))
        self._scroll_chat_to_bottom()
        
    def _hide_typing(self):
        self.typing_indicator.pack_forget()
        
    def _handle_chat_message(self):
        msg = self.chat_input.get().strip()
        if not msg: return
        
        self.chat_input.delete(0, tk.END)
        self._create_chat_bubble(msg, role="User")
        
        threading.Thread(target=self._process_assistant_query, args=(msg,), daemon=True).start()

    def _extract_subject_from_query(self, query):
        import urllib.request
        
        history_text = ""
        if hasattr(self, 'chat_memory') and self.chat_memory:
            history_text = "Chat History:\n" + "\n".join([f"{role}: {msg}" for role, msg in self.chat_memory[-2:]])
            
        prompt = f'''Analyze the user's latest question to find the core visual subject for an image search.
If the question refers to a previous topic (e.g., "where was that?"), use the chat history to resolve what "that" is.
Ignore generic conversational words like "where", "hi", "show me", or "what". Focus purely on the nouns and target objects.
If the subject is a single word like "Food", "Dog", or "Car", expand it into a descriptive embedding prompt (e.g., "A high-quality photo of a meal or food item").
Return ONLY the final search phrase, nothing else.

{history_text}

Latest Question: "{query}"
Search Phrase:'''
        try:
            data = json.dumps({
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": { "temperature": 0.0 }
            }).encode('utf-8')
            req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode('utf-8'))
                subject = result.get('response', '').strip()
                if subject.lower().startswith('search phrase:'):
                    subject = subject[14:].strip()
                elif subject.lower().startswith('subject:'):
                    subject = subject[8:].strip()
                return subject or query
        except Exception:
            return query

    def _build_rag_context(self, subject):
        """Takes the top 5 results from FAISS and extracts filename and Date Taken metadata."""
        text_tokens = clip.tokenize([subject]).to(DEVICE)
        with torch.no_grad():
            query_vector = self.model.encode_text(text_tokens)
        query_vector /= query_vector.norm(dim=-1, keepdim=True)
        query_vector_np = query_vector.cpu().numpy().astype('float32')
        
        D, I = self.faiss_index.search(query_vector_np, k=5)
        
        context_lines = []
        for i in range(len(I[0])):
            idx = I[0][i]
            if idx < 0 or idx >= len(self.filenames): continue
            score = float(D[0][i])
            match_pct = max(0, (1 - (score / 2.5)) * 100)
            
            # Semantic Debugging
            print(f"[DEBUG] Attribute Verification for '{subject}' -> File: {self.filenames[idx]} | Confidence: {match_pct:.1f}%")
            
            if match_pct < 5.0: continue
            
            filename = self.filenames[idx]
            filepath = os.path.join(self.image_folder_path, filename)
            folder_name = os.path.basename(self.image_folder_path)
            meta = get_exif_metadata(filepath)
            date_taken = meta.get('DateTimeOriginal') or meta.get('DateTime') or meta.get('FileDate') or "Unknown Date"
            context_string = f"- Photo: {filename}, Folder: {folder_name}, Path: {filepath}, Date: {date_taken}, Match: {match_pct:.1f}%"
            context_lines.append(context_string)
            print(f"[RAG CONTEXT SENT TO LLM]: {context_string}")
            
        return context_lines

    def _process_assistant_query(self, query):
        try:
            if not getattr(self, 'chat_memory', None):
                self.chat_memory = []

            if not self.faiss_index or not self.model:
                self.root.after(0, lambda: self._create_chat_bubble("Please configure and index a photo folder first.", role="Assistant"))
                return
                
            # --- Intent Classifier (Logic Gate) ---
            query_lower = query.strip().lower().replace("?", "").replace("!", "")
            social_greetings = {"hello", "how are you", "hi", "hey", "greetings", "what's up", "who are you"}
            
            # Fast logic gate without LLM overhead
            is_social = query_lower in social_greetings
            
            # If input contains search subjects, overrule social logic
            search_keywords = ["when", "where", "show me", "what", "find", "search", "who"]
            if any(k in query_lower for k in search_keywords):
                is_social = False
                
            if is_social:
                ans = "Hello! I am your DeepSearch AI Memory Assistant. How can I help you find or remember your photos today?"
                self.root.after(0, lambda: self._create_chat_bubble(ans, role="Assistant"))
                self.chat_memory.append(("User", query))
                self.chat_memory.append(("Assistant", ans))
                return

            self.root.after(0, lambda: self._show_typing("Understanding your question..."))
            
            # --- LLM Connection Health Check ---
            import urllib.request
            try:
                # A quick ping to Ollama's default port to verify it's active
                req = urllib.request.Request("http://127.0.0.1:11434/")
                with urllib.request.urlopen(req, timeout=5) as response:
                    pass
            except Exception:
                self.root.after(0, self._hide_typing)
                self.root.after(0, lambda: self._create_chat_bubble("Error: Local LLM server is not running or unreachable.", role="Assistant"))
                return
            
            # 1. Use local LLM to extract the core visual subject & query expansion
            subject = self._extract_subject_from_query(query)
            
            # Intent refinement and fast expansion
            try:
                expanded_subject = self.expand_query(subject)
                search_subject = expanded_subject if expanded_subject else subject
            except Exception:
                search_subject = subject

            self.root.after(0, lambda: self._show_typing(f"Searching memories for '{search_subject}'..."))
            
            # 2. Build RAG Context using FAISS & EXIF Date Extraction
            context_lines = self._build_rag_context(search_subject)
            context_text = "\n".join(context_lines) if context_lines else "No direct photo matches found."
            
            # 3. Construct System Prompt — separate history from fresh results
            history_text = "\n".join([f"{role}: {msg}" for role, msg in self.chat_memory[-2:]])
            system_prompt = f"""System: You are a helpful, conversational Memory Assistant. Speak casually like a friend.

Here is the brief chat history for context:
{history_text}

AND here are the NEW search results for the user's current question:
{context_text}

CRITICAL RULES:
1. Always base your answer on the NEW search results above. If the user asks about a new topic (like 'beach' or 'cars'), forget the old photos and describe the new ones naturally.
2. Only use the chat history if the user says 'yes', 'where is that', 'open it', or is clearly referring to a previous image.
3. ALWAYS mention the exact filename (e.g., IMG_0897.HEIC) in your answer so the app can create a button.
4. If asked 'where,' use the folder name or date from the search results to answer.
5. Never mention similarity scores or percentages. Describe photos like a human friend would."""
            
            full_prompt = f"{system_prompt}\n\nUser Question: {query}\nAnswer:"
            print(f"\n[DEBUG] LLM PROMPT:\n{full_prompt}\n")

            def prep_bubble():
                self._hide_typing()
                self._current_assistant_text = ""
                self._current_bubble_label, self._current_bubble_frame = self._create_chat_bubble("", role="Assistant")
            
            self.root.after(0, prep_bubble)
            time.sleep(0.05) # Yield briefly for UI creation

            # 4. Generate local response (Streaming)
            data = json.dumps({
                "model": "llama3.2", 
                "prompt": full_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3
                }
            }).encode('utf-8')
            
            req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=data, headers={'Content-Type': 'application/json'})
            full_answer = ""
            with urllib.request.urlopen(req, timeout=120) as response:
                for line in response:
                    if line:
                        result = json.loads(line.decode('utf-8'))
                        chunk = result.get('response', '')
                        if chunk:
                            full_answer += chunk
                            def update_bubble(c=chunk):
                                if hasattr(self, '_current_bubble_label'):
                                    self._current_assistant_text += c
                                    self._current_bubble_label.configure(text=self._current_assistant_text)
                            self.root.after(0, update_bubble)
                            
            # Final scroll and Save to memory
            self.root.after(0, self._scroll_chat_to_bottom)
            self.chat_memory.append(("User", query))
            self.chat_memory.append(("Assistant", full_answer))
            
            # --- "Open in Folder" dynamic button generation ---
            import re
            # Strict regex: captures filenames like IMG_0897.HEIC, 000000009914.jpg, IMG_0924 2.heic
            found_files = re.findall(r'(?<![a-zA-Z])([A-Za-z0-9_\-]+(?:\s\d+)?\.(?:heic|jpg|jpeg|png|webp))', full_answer, re.IGNORECASE)
            found_files = [f.strip().rstrip('.,;:)]}') for f in found_files]
            
            print(f"[DEBUG] Regex found filenames in response: {found_files}")
            print(f"[DEBUG] Indexed filenames sample: {self.filenames[:5]}")
            
            valid_files = [f for f in set(found_files) if f in self.filenames]
            print(f"[DEBUG] Valid files after index check: {valid_files}")
            
            if valid_files:
                def append_jump_buttons(files, frame):
                    for f in files:
                        f_path = os.path.join(self.image_folder_path, f)
                        btn = ctk.CTkButton(
                            frame, text=f"📁 Open in Folder ({f})",
                            command=lambda p=f_path: self.reveal_file_in_os(p),
                            height=28, corner_radius=8, font=("Inter", 11, "bold"),
                            fg_color=self.colors["accent"], text_color="#FFFFFF", hover_color="#584ab8"
                        )
                        btn.pack(padx=15, pady=(0, 10), anchor="w")
                        self._bind_chat_scroll(btn)
                    self.root.after(50, self._scroll_chat_to_bottom)
                self.root.after(100, lambda v=valid_files[:3], f=self._current_bubble_frame: append_jump_buttons(v, f))
                
        except Exception as e:
            # Trap silent crashes inside the thread and report them back to GUI
            self.root.after(0, self._hide_typing)
            self.root.after(0, lambda err=e: self._create_chat_bubble(f"System Error: {str(err)}", role="Assistant"))

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

    def _setup_smooth_scroll(self):
        """Setup smooth momentum scrolling for the main results grid."""
        self.scroll_canvas = self.scroll_frame._parent_canvas
        self.scroll_target = 0.0
        self.scroll_current = 0.0
        self.is_scrolling = False

        # Bind mouse wheel (OS dependent)
        bind_key = "<MouseWheel>" if sys.platform != "linux" else "<Button-4>"
        self.scroll_frame.bind_all(bind_key, self._on_smooth_mousewheel)
        if sys.platform == "linux":
            self.scroll_frame.bind_all("<Button-5>", self._on_smooth_mousewheel)

    def _on_smooth_mousewheel(self, event):
        """Intercept scroll and calculate smooth target."""
        # Normalized delta
        delta = event.delta if sys.platform != "darwin" else event.delta * 20
        if sys.platform == "linux":
            delta = 120 if event.num == 4 else -120

        pixels = -delta * 0.5
        
        # Pixel offset
        self.scroll_target += pixels
        
        # Clamp to bounds - recalculate max_scroll dynamically
        bbox = self.scroll_canvas.bbox("all")
        if not bbox: return "break"
        
        visible_h = self.scroll_canvas.winfo_height()
        content_h = bbox[3]
        max_scroll = max(0, content_h - visible_h)
        
        self.scroll_target = max(0, min(self.scroll_target, max_scroll))

        if not self.is_scrolling:
            self._scroll_animate()
        
        return "break" # Prevent double scrolling

    def _scroll_animate(self):
        """Interpolate towards the scroll target with a settling threshold."""
        self.is_scrolling = True
        diff = self.scroll_target - self.scroll_current
        
        if abs(diff) > 0.1: # Tighter threshold for settling
            # Power damping for smooth deceleration
            self.scroll_current += diff * 0.2 # Slightly faster damping
            
            # Snap to target if very close to avoid micro-jitter
            if abs(self.scroll_target - self.scroll_current) < 0.5:
                self.scroll_current = self.scroll_target

            # Convert to fraction for yview
            bbox = self.scroll_canvas.bbox("all")
            if bbox and bbox[3] > 0:
                fraction = self.scroll_current / bbox[3]
                # Ensure don't exceed the true scrollable range fraction
                visible_h = self.scroll_canvas.winfo_height()
                max_fraction = max(0, (bbox[3] - visible_h) / bbox[3])
                self.scroll_canvas.yview_moveto(min(fraction, max_fraction))
            
            self.root.after(8, self._scroll_animate) # Faster polling for smoothness
        else:
            self.scroll_current = self.scroll_target
            self.is_scrolling = False
        
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
            
            # 6. Chain: generate auto-albums right after tags
            self._generate_auto_albums()
            
        except Exception as e:
            print(f"Error generating context tags: {e}")
        
    def reindex_thread(self):
        threading.Thread(target=self._check_and_index_photos, daemon=True).start()

    def expand_query(self, user_query):
        """
        Refines raw natural language into a descriptive prompt for CLIP.
        Uses embedding strategies to improve match accuracy.
        """
        if not user_query:
            return ""
        
        query = user_query.lower().strip()
        
        # 1. Subject-Specific Expansion Rules
        expansions = {
            "food": "a high-quality photo of a meal, plate of food, or dining experience",
            "car": "a vehicle, automobile, or car parked or driving",
            "document": "a scanned paper, receipt, document with text, or official form",
            "beach": "a coastal scene with sand, ocean waves, and blue sky",
            "night": "a dark scene taken at night with artificial lights or moonlight",
            "nature": "a beautiful landscape, forest, mountains, or outdoor nature scene",
            "people": "a photo of a person, group of friends, or family",
            "work": "an office, desk, computer screen, or professional work environment",
            "travel": "a vacation photo, tourist landmark, travel destination, or sightseeing",
            "dog": "a cute dog, puppy, or canine pet",
            "cat": "a cute cat, kitten, or feline pet",
            "bike": "a bicycle, motorcycle, or bike on a road or path"
        }
        
        # 2. Check if a single word match exists
        if query in expansions:
            return expansions[query]
            
        # 3. Attribute verification (Color + Object)
        colors = ["red", "blue", "green", "yellow", "black", "white", "pink", "purple", "orange"]
        words = query.split()
        if len(words) == 2 and words[0] in colors:
            return f"A person wearing a vibrant {words[0]} {words[1]}"
            
        # 4. Positional Pattern Expansion
        positional = ["lineup", "row", "side-by-side", "arranged", "series"]
        if any(p in query for p in positional):
            # Extract the actual object they are looking for (e.g., "cars in a row" -> "cars")
            obj = query.replace("lineup", "").replace("row", "").replace("in a", "").replace("of", "").strip()
            if not obj: obj = "objects"
            return f"A high-quality photo of multiple {obj} parked or arranged side by side in a row or lineup."
        
        # 4. Handle short phrases by adding "a clear photo of..." (CLIP's favorite prefix)
        if len(words) <= 2:
            return f"a clear photo of {query}"
            
        return query # Return as-is if it's already a descriptive sentence

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
            try:
                expanded_query = self.expand_query(text_query)
                search_term = expanded_query if expanded_query else text_query
            except Exception:
                search_term = text_query

            text_token = clip.tokenize([search_term]).to(DEVICE)
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
        """Populate the grid with results using a staggered animation."""
        # Reset scroll position smoothly
        self.scroll_target = 0
        self.scroll_current = 0
        self.scroll_canvas.yview_moveto(0)

        # Clear existing with a slight delay to prevent flicker
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
            
        # Update Subtitle
        count = len(results)
        self.results_subtitle.configure(text=f"Showing {count} images...")

        columns = 4
        row_offset = 0
        
        # If Query Image, show it first
        if query_image_path:
            self._render_card(query_image_path, "Query Image", "Input Source", 0, 0, highlight=True)
            row_offset = 1
            
        def staggered_render(idx):
            if idx >= len(results): return
            
            filename, score = results[idx]
            path = os.path.join(self.image_folder_path, filename)
            row = (idx // columns) + row_offset
            col = idx % columns
            
            card = self._render_card(path, filename, f"Match: {score:.1f}%", row, col)
            
            # Animate card appearance (fade/slide simulation)
            card.configure(fg_color="#F3F4F6") # Start with neutral
            self.root.after(10, lambda: card.configure(fg_color=self.colors["sidebar_bg"]))
            
            # Next one after a small delay
            self.root.after(30, lambda: staggered_render(idx + 1))

        # Start staggered rendering
        staggered_render(0)

    def populate_grid(self, results_list, query_image_path=None):
        """Legacy method redirected to new animator."""
        self._display_results(results_list, query_image_path)

    def show_image_in_grid(self, filename):
        """Highlights a specific image in the grid and switches to the gallery tab."""
        self._show_photos_view()
        
        # Look through all currently rendered cards in the scroll_frame
        found = False
        widgets_to_search = self.scroll_frame.winfo_children()
        
        for card in widgets_to_search:
            # We hid the path inside the bind closure earlier, but we can verify via the title label text
            # A more robust way is traversing children and checking the label text.
            for child in card.winfo_children():
                if isinstance(child, ctk.CTkLabel) and child.cget("text").startswith(filename[:18]):
                    found = True
                    # Flash red border
                    original_border = card.cget("border_color")
                    original_width = card.cget("border_width")
                    
                    def reset_border():
                        try:
                            card.configure(border_color=original_border, border_width=original_width)
                        except: pass

                    card.configure(border_color=self.colors["accent"], border_width=3)
                    self.root.after(3000, reset_border)

                    # Scroll to this card safely by calculating its Y pos vs total height
                    self.scroll_frame.update_idletasks()
                    card_y = card.winfo_y()
                    total_h = self.scroll_frame.winfo_reqheight()
                    if total_h > 0:
                        fraction = card_y / total_h
                        self.scroll_canvas.yview_moveto(max(0, fraction - 0.1)) # Offset slightly to see the top
                    break
        
        if not found:
            self.status_text.set(f"Wait... searching for {filename} to highlight it.")
            # If not in the current view, we can force a generic search that includes it.
            self.search_text_var.set(filename.split('.')[0])
            self._run_search()

    def _render_card(self, path, title, subtitle, row, col, highlight=False):
        """Create a modernized white card."""
        bg_color = self.colors["active_bg"] if highlight else self.colors["sidebar_bg"]
        
        card = ctk.CTkFrame(self.scroll_frame, fg_color=bg_color, corner_radius=12, border_width=1, border_color="#F3F4F6")
        card.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

        # Metadata
        filename_label = ctk.CTkLabel(card, text=title[:18] + "..." if len(title) > 18 else title, 
                                      font=("Inter", 13, "bold"), text_color=self.colors["text"], cursor="hand2")
        filename_label.pack(pady=(5, 0), padx=12, anchor="w")
        filename_label.bind("<Button-1>", lambda event, p=path: self.reveal_file_in_os(p))
        
        # Bottom Info Row
        info_row = ctk.CTkFrame(card, fg_color="transparent")
        info_row.pack(fill="x", padx=12, pady=(2, 12))
        
        # Bottom Stats (Match Score)
        match_color = self.colors["success"] if not highlight else self.colors["accent"]
        score_label = ctk.CTkLabel(info_row, text=subtitle, font=("Inter", 11, "bold"), text_color=match_color)
        score_label.pack(side="right")
        
        # Placeholder for Image (Prevents Jitter)
        img_label = ctk.CTkLabel(card, text="⌛", font=("Inter", 24), text_color=self.colors["subtext"], 
                                 width=160, height=160, corner_radius=8, cursor="hand2")
        img_label.pack(pady=(12, 5), padx=12, before=filename_label)

        def _async_load():
            try:
                catalog_dir, _, _, _, _ = get_catalog_paths(self.image_folder_path)
                THUMBNAIL_FOLDER = os.path.join(catalog_dir, "thumbnails")
                thumb_path = os.path.join(THUMBNAIL_FOLDER, os.path.basename(path))

                if os.path.exists(thumb_path):
                    pil_img = Image.open(thumb_path)
                else:
                    pil_img = Image.open(path)
                    pil_img = ImageOps.pad(pil_img, (300, 300), color=bg_color)
                
                # Wrap in CTkImage
                ctk_img = ctk.CTkImage(light_image=pil_img, size=(160, 160))
                
                # Update on main thread
                if self.root.winfo_exists():
                    self.root.after(0, lambda: self._safe_update_image(img_label, ctk_img, path))
            except Exception:
                pass

        threading.Thread(target=_async_load, daemon=True).start()
        
        return card

    def _safe_update_image(self, label, image, path):
        """Safely update label with image after loading."""
        try:
            if label.winfo_exists():
                label.configure(image=image, text="")
                label.image = image # Ref
                label.bind("<Button-1>", lambda event, p=path: self.reveal_file_in_os(p))
        except Exception: pass

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