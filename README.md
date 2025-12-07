🧠 AI Semantic Image Search (Desktop & Local-First)

This project demonstrates a high-performance, server-free, local semantic search engine built for the final-year MCA research paper. It allows users to search large local photo libraries using natural language queries (e.g., "a golden retriever running on a beach").

🚀 Key Features

Local-First: Runs 100% locally on macOS (optimized for Apple Silicon/M4) without sending any data to the cloud.

Semantic Search: Uses the CLIP model to understand the meaning of the text query and the content of the images.

Hyper-Optimized Speed: Implements FAISS (Facebook AI Similarity Search) for instantaneous nearest-neighbor vector search.

Desktop GUI: Built with Tkinter for a simple, native desktop application experience (no web server needed).

🛠️ Setup and Installation

Prerequisites

Python: Python 3.12 (Installed via Homebrew is recommended on macOS).

Dependencies: Homebrew and Xcode Command Line Tools.

Installation Steps

Clone the Repository:

git clone [https://github.com/omiii18/ai-image-search.git](https://github.com/omiii18/ai-image-search.git)
cd ai-image-search


Create and Activate Virtual Environment:

# Create the environment using the specific Python version
/opt/homebrew/bin/python3.12 -m venv .venv
# Activate it
source .venv/bin/activate


Install Required Libraries:

# This installs all core components: PyTorch, FAISS, CLIP, and PIL.
pip install torch torchvision torchaudio faiss-cpu numpy Pillow git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)


🏃 Usage Guide

Add Images: Place your images into the ./images folder.

Build the AI Index (One-Time Setup): This process scans your images, generates the vector embeddings, and builds the FAISS index.

python3 index.py


Run the Desktop Application:

python3 gui_search.py


If the application does not launch, ensure you have resolved the OMP library conflict by setting the environment variable:

# Run this before python3 gui_search.py if needed:
export KMP_DUPLICATE_LIB_OK=TRUE


💻 Core Project Files

File

Purpose

gui_search.py

The main Python desktop application (Tkinter UI and Search Logic).

index.py

Utility script to encode all images and build the FAISS index files.

images/

Directory where user photos are stored.

.gitignore

Prevents uploading large and unnecessary files (like .venv and embeddings).

Thank you

