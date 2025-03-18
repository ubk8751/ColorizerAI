import torch
import torch.optim as optim
import torch.nn as nn
import os
import sys
from src.model import UNet
from src.dataset import get_dataloader
from src.loss import PerceptualLoss

def train():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    train_loader = get_dataloader(batch_size=8)

    # Initialize model and move it to the correct device
    model = UNet().to(device)

    # Loss function (Perceptual Loss) - Ensure it uses the same device
    criterion = PerceptualLoss(device=device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    EPOCHS = 10
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0

        for i, (grayscale, ab) in enumerate(train_loader):
            grayscale = grayscale.to(device)  # L channel (grayscale)
            ab = ab.to(device)  # AB channels (color information)

            # Forward pass
            output_ab = model(grayscale)

            # Compute Perceptual Loss with L channel
            loss = criterion(output_ab, ab, grayscale)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Print training progress on the same line
            sys.stdout.write(f"\rEpoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
            sys.stdout.flush()

        # Print epoch loss on a new line
        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Completed - Avg Loss: {avg_loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            model_path = f"models/colorization_net_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model checkpoint saved to {model_path}")

    # Save final trained model
    torch.save(model.state_dict(), "models/colorization_net.pth")
    print("Training complete! Final model saved.")

# Run training if script is executed directly
if __name__ == "__main__":
    train()
