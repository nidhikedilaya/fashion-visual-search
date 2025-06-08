# Import necessary libraries
import os
os.environ["STREAMLIT_WATCHER_IGNORE_MODULES"] = "torch"  # Prevent Streamlit from watching torch internals

import streamlit as st  # For building the interactive web UI
import numpy as np  # For array and vector ops
from PIL import Image  # To process image files from binary content
import torch  # PyTorch for deep learning model
import torchvision.models as models  # Pretrained models from torchvision
import torchvision.transforms as transforms  # For preprocessing images (resizing, converting to tensor)
from sklearn.metrics.pairwise import cosine_similarity  # For computing similarity between vectors
import pandas as pd  # For handling price data

# Initialize session state for clicked image
if 'clicked_image_idx' not in st.session_state:
    st.session_state.clicked_image_idx = None

# Load pretrained ResNet50 model, remove classification layer, and cache it
@st.cache_resource  # Ensures this runs only once per session unless code/data changes
def load_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Load pretrained ResNet50 - CNN model
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove final classification layer - keep only feature extractor layers
    model.eval()  # Set model to evaluation mode
    return model

# Load cached precomputed image vectors and URLs
@st.cache_data  # Cache loaded vectors and URLs to prevent reloading
def load_vectors_and_urls():
    vectors = np.load("vectors.npy")  # Load image feature vectors
    urls = np.load("urls.npy", allow_pickle=True).tolist()  # Load image URLs
    
    # Load price data
    df1 = pd.read_csv("dresses_bd_processed_data.csv")
    df2 = pd.read_csv("jeans_bd_processed_data.csv")
    
    # Convert price dictionaries to integers
    def extract_price(price_dict):
        if isinstance(price_dict, str):
            try:
                # Convert string representation of dict to actual dict
                price_dict = eval(price_dict)
                return int(price_dict['INR'])
            except:
                return 0
        return 0
    
    prices1 = df1['selling_price'].apply(extract_price)
    prices2 = df2['selling_price'].apply(extract_price)
    prices = pd.concat([prices1, prices2]).values
    
    # Get unique brands
    brands1 = df1['brand'].fillna('Unknown')
    brands2 = df2['brand'].fillna('Unknown')
    brands = pd.concat([brands1, brands2]).values
    
    return vectors, urls, prices, brands

# Convert uploaded image to vector
def image_to_vector(img, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224 (ResNet50 input size)
        transforms.ToTensor()  # Convert image to PyTorch tensor
    ])
    tensor = transform(img).unsqueeze(0)  # Add batch dimension -> shape (1, 3, 224, 224)
    with torch.no_grad():  # No gradient computation needed for inference
        vec = model(tensor).squeeze().numpy().reshape(1, -1)  # Extract features and reshape to (1, 2048)
    return vec

# Streamlit UI layout
st.title("Fashion Visual Search")

# Load data first
model = load_model()  # Load feature extractor model
vectors, urls, prices, brands = load_vectors_and_urls()  # Load cached image features, URLs, prices, and brands

# Create a sidebar for image upload
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a fashion image", type=["jpg", "jpeg", "png"])
    
    # Add price range filter
    st.header("Price Range")
    min_price = 0
    max_price = 100000  # Adjust this based on your actual price range
    price_range = st.slider(
        "Select price range",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=1000
    )
    
    # Add brand filter
    st.header("Brand Filter")
    selected_brand = st.selectbox(
        "Select a brand",
        options=["All Brands"] + sorted(list(set(brands))),
        index=0
    )
    
    # Display uploaded image in the sidebar
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# Main content area
if uploaded_file:
    # Convert uploaded image to feature vector
    uploaded_img = Image.open(uploaded_file).convert("RGB")  # Convert to RGB (ensure 3 channels)
    query_vec = image_to_vector(uploaded_img, model)  # Extract features from uploaded image

    # Compute cosine similarity between uploaded image and all dataset vectors
    sims = cosine_similarity(query_vec, vectors)[0]  # Get similarity scores
    
    # Filter by price range and brand
    price_mask = (prices >= price_range[0]) & (prices <= price_range[1])
    brand_mask = (brands == selected_brand) if selected_brand != "All Brands" else np.ones_like(brands, dtype=bool)
    
    filtered_sims = sims.copy()
    filtered_sims[~(price_mask & brand_mask)] = -1  # Set similarity to -1 for items outside filters
    
    top10_idx = np.argsort(filtered_sims)[-10:][::-1]  # Get indices of top 10 highest scores
    top10_idx = top10_idx[filtered_sims[top10_idx] != -1]  # Remove items outside filters

    st.subheader("Top Similar Products")  # Display results
    
    # Create two rows of 5 columns each
    for row in range(2):  # 2 rows
        cols = st.columns(5)  # 5 columns per row
        for col_idx, col in enumerate(cols):
            img_idx = row * 5 + col_idx  # Calculate the index in top10_idx
            if img_idx < len(top10_idx):  # Make sure we don't exceed the number of images
                with col:
                    # Display image
                    st.image(urls[top10_idx[img_idx]], use_container_width=True)
                    # Add a button below each image
                    if st.button("Find Similar", key=f"btn_{top10_idx[img_idx]}"):
                        st.session_state.clicked_image_idx = top10_idx[img_idx]
                    st.write(f"Price: ₹{prices[top10_idx[img_idx]]:,}")
                    st.write(f"Brand: {brands[top10_idx[img_idx]]}")

    # Show similar images when one is clicked
    if st.session_state.clicked_image_idx is not None:
        st.subheader("People also look for:")
        
        # Get the vector of the clicked image
        clicked_vec = vectors[st.session_state.clicked_image_idx].reshape(1, -1)
        
        # Compute similarity with all other images
        clicked_sims = cosine_similarity(clicked_vec, vectors)[0]
        
        # Filter by price range and brand
        clicked_filtered_sims = clicked_sims.copy()
        clicked_filtered_sims[~(price_mask & brand_mask)] = -1
        
        # Get top 10 similar images (excluding the clicked image itself)
        clicked_filtered_sims[st.session_state.clicked_image_idx] = -1
        similar_idx = np.argsort(clicked_filtered_sims)[-10:][::-1]
        similar_idx = similar_idx[clicked_filtered_sims[similar_idx] != -1]
        
        # Display similar images in two rows
        for row in range(2):
            cols = st.columns(5)
            for col_idx, col in enumerate(cols):
                img_idx = row * 5 + col_idx
                if img_idx < len(similar_idx):
                    with col:
                        st.image(urls[similar_idx[img_idx]], use_container_width=True)
                        if st.button("Find Similar", key=f"btn_similar_{similar_idx[img_idx]}"):
                            st.session_state.clicked_image_idx = similar_idx[img_idx]
                        st.write(f"Price: ₹{prices[similar_idx[img_idx]]:,}")
                        st.write(f"Brand: {brands[similar_idx[img_idx]]}")
