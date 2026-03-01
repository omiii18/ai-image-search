# 🧠 DeepSearch AI Photo Library (Desktop & Local-First)

**DeepSearch AI** is a powerful, privacy-focused desktop application that allows you to search through your local photo libraries using natural language or image inputs. Unlike traditional filename-based search, DeepSearch understands the *content* and *context* of your photos using advanced AI.

---

## 🚀 Key Features

*   **Semantic Text Search**: Find photos by describing them (e.g., "a golden retriever running on a beach", "birthday party at night"). The AI "sees" the image content.
*   **OCR Text Search**: Search for **text inside your images** (e.g., screenshots, documents, signs) using integrated Tesseract OCR and Full-Text Search.
*   **Image-to-Image Search**: Drag and drop an existing image or click the "Image" button to find visually similar photos in your library.
*   **Dynamic Contextual Tags**: Instead of static filters, the app performs a **Zero-Shot AI analysis** of your folder content to generate the most relevant "Quick Tags" automatically.
*   **Drag-and-Drop Interaction**: Instantly search by dropping any image file directly into the application window or search bar.
*   **Native OS Integration**: Click on any search result to **"Reveal in Finder/File Explorer"**, highlighting the exact file location on your system.
*   **Smart Auto-Albums**: Automatically groups photos into virtual albums (e.g., "Portraits", "Night", "Landscapes") using **dynamic semantic thresholding**.
*   **100% Local & Private**: Runs entirely on your machine. No data is ever sent to the cloud. Optimized for both **Windows** and **macOS** (Apple Silicon compatible).
*   **Modern Desktop GUI**: A sleek, premium light-themed interface built with **CustomTkinter**, featuring glassmorphism elements and responsive result grids.
*   **Broad Format Support**: Supports standard formats (JPG, PNG, WEBP) and **HEIC** (Apple High Efficiency Image Container).
*   **Auto-Indexing**: Automatically detects new images in your selected folder and updates the FAISS index with a real-time progress bar.

---

## 🛠️ How It Works

This project combines state-of-the-art Computer Vision and NLP models:

1.  **Embedding Generation (CLIP)**:
    *   The application uses OpenAI's **CLIP (Contrastive Language-Image Pre-training)** model (`ViT-L/14`).
    *   CLIP maps both **images** and **text** into the same high-dimensional vector space (embeddings). This means text and images that are semantically similar will be mathematically close to each other.

2.  **Vector Indexing (FAISS)**:
    *   The generated image embeddings are stored in a **FAISS** index.
    *   FAISS allows for extremely fast nearest-neighbor search, finding the vectors most similar to your query vector.

3.  **Text Extraction (OCR & FTS)**:
    *   The app uses **Tesseract OCR** to extract text from your images.
    *   This text is stored in a **SQLite** database with Full-Text Search (FTS5) enabled, allowing you to find document scans or screenshots by their written content alongside semantic meaning.

4.  **Retrieval & Ranking**:
    *   When you search, your query is processed both semantically (CLIP) and literally (OCR).
    *   The app combines cosine similarity scores and keyword matches to rank results instantaneously.

5.  **Smart Auto-Albums**: 
    *   After indexing, the app scans your library against predefined **archetypes** (Portraits, Night, Landscapes, Documents, Rainy, Food, Pets).
    *   It uses **Dynamic Thresholding**: instead of a fixed score, it samples your folder's semantic range to find the top-20% most relevant matches for each category.
    *   Albums only appear if they contain more than 5 high-quality matches, ensuring your sidebar stays clean and relevant.

---

## 📥 Download & Quick Start

### macOS
1.  Go to the [Latest Release](https://github.com/omiii18/ai-image-search/releases).
2.  Download the `DeepSearchAI_macOS.zip` file.
3.  Unzip the file and move `DeepSearchAI.app` to your `Applications` folder.
4.  **Right-click** the app and select **"Open"** (needed for the first time as it's not signed).

### Windows
1.  Go to the [Latest Release](https://github.com/omiii18/ai-image-search/releases).
2.  Download the `DeepSearchAI_windowsOS.zip` file.
3.  Extract the ZIP folder to a location of your choice.
4.  Run `DeepSearchAI.exe` to launch the application.

> [!NOTE]
> You still need to have [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) installed on your system for text search to work. For Windows, download the installer from [here](https://github.com/UB-Mannheim/tesseract/wiki).

---

## 📦 Development Setup

### Prerequisites
*   **Python 3.12** or higher.
*   **git** (for cloning).

### 1.1 Install System Dependencies (Tesseract OCR)
This project requires Tesseract for OCR text extraction.

**macOS (Homebrew):**
```bash
brew install tesseract
```
**Windows:**
Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install tesseract-ocr
```

### 2. Clone the Repository
```bash
git clone https://github.com/omiii18/ai-image-search.git
cd ai-image-search
```

### 3. Create a Virtual Environment
It is highly recommended to use a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 4. Install Dependencies
```bash
pip install torch torchvision torchaudio faiss-cpu numpy Pillow pillow-heif pytesseract tqdm git+https://github.com/openai/CLIP.git
```

*(Note: `pillow-heif` is required for HEIC support)*

---

## 🏃 Usage Guide

### 1. Run the Application
Start the desktop interface:

```bash
python3 image_search_app.py
```

*Note: If you encounter an OpenMP error on macOS, run this command before starting the app:*
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### 3. Configure Your Library
The first time you run the app, stick to the GUI flow:
1.  Click **"Select Folder"** and choose your photo directory.
2.  The app will automatically create a `settings.json` file.
3.  Indexing will start with a progress bar.

*(Advanced)*: You can rename `settings.example.json` to `settings.json` and edit it manually if preferred.

---

## 🗄️ Project Structure

*   `image_search_app.py`: GUI application entry point.
*   `index.py`: Core logic for CLIP embedding generation and FAISS indexing.
*   `embeddings/`: Stores the vector index (`faiss.index`), file mapping (`mapping.pkl`), and OCR data (`ocr.db`).
*   `.cache/`: Stores generated thumbnails for fast loading.
*   `settings.json`: User configuration (gitignored).
*   `settings.example.json`: Template for configuration.

---

## 💻 Tech Stack

*   **Core Logic**: Python 3.12+
*   **AI Model**: OpenAI CLIP (ViT-B/32)
*   **Vector Database**: FAISS (Facebook AI Similarity Search)
*   **OCR Engine**: Tesseract OCR + SQLite FTS5
*   **GUI Framework**: CustomTkinter (Premium Light Theme)
*   **Image Handling**: Pillow (PIL) + pillow-heif (HEIC)
*   **Drag & Drop**: TkinterDnD2

---

**Developed for College Project:**
*A implementation of "Local-First Semantic Search for Personal Archives"*
