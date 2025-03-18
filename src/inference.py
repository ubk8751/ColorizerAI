import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from src.model import UNet
import os

def preprocess_image(img_path, device, img_size=256):
    """
    Preprocess the input grayscale image before feeding it into the model.

    Args:
        img_path (str): Path to the grayscale image.
        device (torch.device): The device to load the image onto.
        img_size (int): Target image size for resizing.

    Returns:
        torch.Tensor: Preprocessed grayscale image tensor.
        np.ndarray: Original L channel in LAB space for post-processing.
    """
    # Open image and convert to grayscale
    img = Image.open(img_path).convert("L")

    # Resize image to match model training size
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),  # Ensure size matches training
        transforms.ToTensor(),  # Convert to tensor
    ])
    
    L = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Normalize L channel to [0, 100] range (needed for LAB conversion)
    L_original = np.array(img.resize((img_size, img_size))) * (100.0 / 255.0)

    return L, L_original

def run_inference(img_name: str, img_size=256):
    """
    Run the colorization model on a grayscale image.

    Args:
        img_name (str): Filename of the grayscale image.
        img_size (int): Target size of the image for model inference.

    Outputs:
        Saves the colorized image in the "outputs/" folder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = UNet().to(device)
    model.load_state_dict(torch.load("models/colorization_net.pth", map_location=device))
    model.eval()

    # Create outputs directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Preprocess the image
    img_path = f"./input/{img_name}"
    L, L_original = preprocess_image(img_path, device, img_size)

    # Run inference
    with torch.no_grad():
        output_ab = model(L).cpu().squeeze(0).numpy() * 128.0  # Convert back to LAB range

    # 🔹 Fix: Ensure output AB has the correct shape for LAB
    output_ab = np.transpose(output_ab, (1, 2, 0))  # Convert (2, H, W) → (H, W, 2)

    # 🔹 Fix: Scale L to LAB range [0, 100] before conversion
    L_original = np.expand_dims(L_original, axis=-1)  # Convert (H, W) → (H, W, 1)

    # Convert LAB to RGB
    lab_output = np.concatenate([L_original, output_ab], axis=-1)
    rgb_output = lab2rgb(lab_output)  # Convert LAB to RGB
    rgb_output = np.clip(rgb_output, 0, 1)  # Ensure valid RGB values

    # Convert to PIL image and save
    colorized_img = Image.fromarray((rgb_output * 255).astype(np.uint8))

    # Save output
    output_path = os.path.join(output_dir, f"colorized_{os.path.basename(img_name)}")
    colorized_img.save(output_path)
    print(f"Colorized image saved at: {output_path}")

if __name__ == "__main__":
    run_inference("lady_with_hat.jpg")
