from src.model.unet import UNet
import streamlit as st
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
import config.configure as config
from src.pipelines.predict import predict_mask

model = UNet(3, 1, [64, 128, 256, 512])
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model.load_state_dict(torch.load(config.SAVE_MODEL_PATH, map_location=torch.device(device)))
# Set up transformations for the input image


transform =  A.Compose([
        A.Resize(224, 224, p=1.0),
        ToTensorV2(),
    ])
# Streamlit app
def main():
    st.title("MRI segmenation App")

    # Upload image through Streamlit
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded and processed images side by side
        col1, col2 = st.columns(2)  # Using beta_columns for side-by-side layout

        # Display the uploaded image in the first column
        col1.header("Original Image")
        col1.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Process the image (replace this with your processing logic)
        processed_image = generate_image(uploaded_image)

        # Display the processed image in the second column
        col2.header("Processed Image")
        col2.image(processed_image, caption="Processed Image", use_column_width=True)

# Function to generate an image using the PyTorch model
def generate_image(uploaded_image):
    # Load the uploaded image
    input_image = Image.open(uploaded_image)

    image = np.array(input_image).astype(np.float32) / 255.
    # Apply transformations
    input_tensor = transform(image=image)["image"].unsqueeze(0)

    # Generate an image using the PyTorch model
    mask = predict_mask(data=input_tensor, device=device, model=model, inference=True)
    mask = mask[0].permute(1, 2, 0)
    image = input_tensor[0].permute(1, 2, 0)
    
    mask = image + mask*0.3
    mask = mask.permute(2, 0, 1)
    mask = transforms.ToPILImage()(mask)
    return mask


if __name__ == "__main__":
    main()