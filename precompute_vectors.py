# Import necessary libraries
import pandas as pd  # For reading CSV files
import numpy as np  # For saving/loading image feature arrays
import requests  # For fetching images from URLs
from PIL import Image  # To open and process image files
from io import BytesIO  # To read binary image data
import torch  # PyTorch
import torchvision.models as models  # Pretrained model
import torchvision.transforms as transforms  # For image resizing and conversion

# Load pretrained ResNet50 model and remove classification layer
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Load pretrained ResNet50
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Keep all layers except the last classification layer
    model.eval()  # Set model to eval mode for inference
    return model

# Convert image from URL into a vector using pretrained model
def image_url_to_vector(url, model, transform):
    try:
        img = Image.open(BytesIO(requests.get(url).content)).convert("RGB")  # Fetch and convert image to RGB
        tensor = transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():  # Disable gradients
            vec = model(tensor).squeeze().numpy().reshape(1, -1)  # Get vector and reshape
        return vec
    except:
        return np.zeros((1, 2048))  # Return zero vector if image fails to load

# Load image URLs from both CSV datasets
def load_image_urls():
    df1 = pd.read_csv("dresses_bd_processed_data.csv")  # Load dresses dataset
    df2 = pd.read_csv("jeans_bd_processed_data.csv")  # Load jeans dataset
    urls = df1["feature_image_s3"].tolist() + df2["feature_image_s3"].tolist()  # Concatenate URL lists
    return urls

# Main script to compute and save vectors
if __name__ == "__main__":
    model = load_model()  # Load model once
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize input image
        transforms.ToTensor()  # Convert to tensor
    ])

    urls = load_image_urls()  # Get all image URLs
    vectors = []  # List to hold all feature vectors

    # Process each image and convert to feature vector
    for i, url in enumerate(urls):
        print(f"Processing image {i+1}/{len(urls)}")  # Progress indicator
        vec = image_url_to_vector(url, model, transform)  # Convert image to vector
        vectors.append(vec)

    vectors = [v.reshape(1, -1) if v.shape != (1, 2048) else v for v in vectors]
    vectors = np.vstack(vectors)  # Stack all vectors into a 2D array of shape (N, 2048)

    # Save the vectors and URLs for faster loading in app.py
    np.save("vectors.npy", np.vstack(vectors))  # Save all image feature vectors
    np.save("urls.npy", np.array(urls))  # Save corresponding image URLs
