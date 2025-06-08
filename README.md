# Fashion Image Similarity Search

This project implements a fashion image similarity search system using deep learning. It processes fashion images (dresses and jeans) and creates vector embeddings that can be used to find similar items based on visual features.

## Project Structure

```
.
├── Images/                  # Sample images for testing
├── app.py                  # Main Streamlit application
├── precompute_vectors.py   # Script to generate image embeddings
├── vectors.npy            # Pre-computed image feature vectors (generated locally)
├── urls.npy               # Corresponding image URLs (generated locally)
├── dresses_bd_processed_data.csv  # Dresses dataset
├── jeans_bd_processed_data.csv    # Jeans dataset
└── requirements.txt        # Project dependencies
```

## Features

- Image feature extraction using ResNet50
- Vector-based similarity search
- Interactive web interface using Streamlit
- Support for both dresses and jeans datasets
- Efficient vector storage and retrieval

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Generating Feature Vectors

Before running the main application, you need to generate the image feature vectors. The large data files (`vectors.npy` and `urls.npy`) are not included in the repository due to size limitations. You'll need to generate them locally:

```bash
python precompute_vectors.py
```

This script will:

- Load the ResNet50 model
- Process all images from the datasets
- Generate and save feature vectors to `vectors.npy` (approximately 273MB)
- Save corresponding URLs to `urls.npy` (approximately 8MB)

Note: This process might take some time depending on your hardware. A GPU is recommended for faster processing.

### Running the Application

To start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser, where you can:

- Upload images to find similar items
- View similar fashion items from the dataset
- Compare different items

## Technical Details

- The system uses ResNet50 (pretrained on ImageNet) for feature extraction
- Images are processed to 224x224 pixels before feature extraction
- Feature vectors are 2048-dimensional
- Similarity is computed using cosine similarity

## Dependencies

- streamlit==1.32.0
- torch==2.2.0
- torchvision==0.17.0
- pillow
- pandas
- numpy
- scikit-learn
- requests
