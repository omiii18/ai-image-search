# 🧠 DeepSearch AI Photo Library (Desktop & Local-First)

**DeepSearch AI** is a powerful, privacy-focused desktop application that allows you to search through your local photo libraries using natural language or image inputs. Unlike traditional filename-based search, DeepSearch understands the *content* and *context* of your photos using advanced AI.

---

## 🚀 Key Features

*   **Semantic Text Search**: Find photos by describing them (e.g., "a golden retriever running on a beach", "birthday party at night"). The AI "sees" the image content.
*   **Image-to-Image Search**: Use an existing image to find visually similar photos in your library.
*   **100% Local & Private**: Runs entirely on your machine. No data is ever sent to the cloud. Optimized for macOS (Apple Silicon compatible).
*   **High Performance**: Powered by **FAISS** (Facebook AI Similarity Search) for millisecond-speed queries even with large libraries.
*   **Modern Desktop GUI**: A sleek dark-mode interface built with Tkinter, featuring responsive results and easy navigation.
*   **Broad Format Support**: Supports standard formats (JPG, PNG, WEBP) and **HEIC** (Apple High Efficiency Image Container).
*   **Auto-Indexing**: Automatically detects new images in your selected folder and updates the search index.

---

## 🛠️ How It Works

This project combines state-of-the-art Computer Vision and NLP models:

1.  **Embedding Generation (CLIP)**:
    *   The application uses OpenAI's **CLIP (Contrastive Language-Image Pre-training)** model (`ViT-L/14`).
    *   CLIP maps both **images** and **text** into the same high-dimensional vector space (embeddings). This means text and images that are semantically similar will be mathematically close to each other.

2.  **Vector Indexing (FAISS)**:
    *   The generated image embeddings are stored in a **FAISS** index.
    *   FAISS allows for extremely fast nearest-neighbor search, finding the vectors most similar to your query vector.

3.  **Retrieval & Ranking**:
    *   When you search, your query (text or image) is converted into a vector.
    *   The app calculates the **Cosine Similarity** between your query and your library.
    *   Results are ranked by match percentage and displayed instantenously.

---

## 📦 Installation & Setup

### Prerequisites
*   **Python 3.12** or higher.
*   **git** (for cloning).

### 1. Clone the Repository
```bash
git clone https://github.com/omiii18/ai-image-search.git
cd ai-image-search
```

### 2. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies.

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Activate it (Windows)
# .venv\Scripts\activate
```

### 3. Install Dependencies
Install the required libraries including PyTorch, FAISS, and CLIP.

```bash
pip install torch torchvision torchaudio faiss-cpu numpy Pillow pillow-heif git+https://github.com/openai/CLIP.git
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

### 2. Configure Your Library
1.  Click the **"Select Folder"** button in the top header.
2.  Choose the local directory containing your photos.
3.  The app will automatically start **indexing** your photos. This might take a few moments depending on the size of your library.

### 3. Search
*   **Text Search**: Type a description (e.g., "sunset over mountains") and press Enter or click Search.
*   **Image Search**: Click **"Browse Image"** to select a reference photo and find similar matches.
*   **Suggestions**: Use the suggested tags for quick discovery.

---

## 🗄️ Project Structure

*   `image_search_app.py`: The main entry point. Handles the GUI (Tkinter), application logic, and search workflow.
*   `index.py`: Contains the logic for processing images and building the FAISS index.
*   `embeddings/`: Directory where the vector index (`faiss.index`) and metadata (`mapping.pkl`) are stored.
*   `requirements.txt`: List of python dependencies.
*   `settings.json`: Stores user preferences (e.g., last selected folder).

---

## 💻 Tech Stack

*   **Core Logic**: Python 3.12
*   **AI Model**: OpenAI CLIP (ViT-L/14)
*   **Vector Database**: FAISS (CPU)
*   **GUI Framework**: Tkinter
*   **Image Processing**: Pillow (PIL)

---

**Developed for College Project:**
*A implementation of "Local-First Semantic Search for Personal Archives"*
