import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MaskedAutoencoder # Imports your model architecture

# --- 1. Page Configuration ---
st.set_page_config(page_title=ViT Masked Autoencoder, layout=wide)
st.title(Self-Supervised Masked Autoencoder (MAE))
st.markdown(Upload an image and adjust the masking ratio to see the ViT-Base encoder and ViT-Small decoder reconstruct missing patches.)

# --- 2. Model Loading & Caching ---
@st.cache_resource
def load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedAutoencoder()
    # Ensure your Kaggle .pt file is in the same directory
    model.load_state_dict(torch.load('mae_final_model.pt', map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# --- 3. Image Preprocessing & Postprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def denormalize(x)
    Reverts ImageNet normalization using your exact logic.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(x.device)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(x.device)
    return torch.clamp(x  std + mean, 0, 1)

# --- 4. User Interface ---
with st.sidebar
    st.header(Controls)
    uploaded_file = st.file_uploader(Choose an image..., type=[jpg, jpeg, png])
    mask_ratio = st.slider(Masking Ratio, min_value=0.10, max_value=0.90, value=0.75, step=0.05)

if uploaded_file is not None
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # --- Dynamic Mask Ratio Update ---
    # We update these attributes directly so the model uses the slider value
    # without needing to re-initialize the entire architecture.
    model.mask_ratio = mask_ratio
    model.num_visible = int(model.num_patches  (1 - mask_ratio))
    model.num_masked = model.num_patches - model.num_visible
    
    # --- Run Inference (Using your evaluation logic) ---
    with torch.no_grad()
        pred, mask = model(input_tensor)
        target = model.patchify(input_tensor)
        
        # Apply mask to target patches
        masked_patches = target.clone()
        masked_patches[mask == 0] = 0
        
        # Unpatchify back to images
        masked_images = model.unpatchify(masked_patches)
        reconstructed = model.unpatchify(pred)

    # Process tensors for Streamlit display
    orig_img = denormalize(input_tensor[0]).cpu().permute(1, 2, 0).numpy()
    mask_img = denormalize(masked_images[0]).cpu().permute(1, 2, 0).numpy()
    recon_img = denormalize(reconstructed[0]).cpu().permute(1, 2, 0).numpy()
    
    # --- Display Results ---
    col1, col2, col3 = st.columns(3)
    
    with col1
        st.subheader(Original Image)
        st.image(orig_img, use_container_width=True)
        
    with col2
        st.subheader(fMasked Input ({int(mask_ratio100)}%))
        st.image(mask_img, use_container_width=True)
        
    with col3
        st.subheader(Model Reconstruction)
        st.image(recon_img, use_container_width=True)
else
    st.info(Please upload an image from the sidebar to begin.)