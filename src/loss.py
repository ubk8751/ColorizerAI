import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super(PerceptualLoss, self).__init__()

        self.device = device  # Store device
        vgg = models.vgg16(pretrained=True).features[:16]  # Use first 16 layers of VGG16
        self.vgg = vgg.to(self.device).eval()  # Move VGG to the right device

        for param in self.vgg.parameters():
            param.requires_grad = False  # Freeze VGG weights

        self.criterion = nn.MSELoss()
        self.color_loss = nn.L1Loss()  # 🔹 Encourage stronger AB predictions

    def forward(self, pred_ab, target_ab, l_channel):
        """
        pred_ab: Predicted AB channels (batch, 2, H, W)
        target_ab: Ground truth AB channels (batch, 2, H, W)
        l_channel: Grayscale L channel (batch, 1, H, W)
        """
        # Ensure inputs are on the same device as the model
        pred_ab = pred_ab.to(self.device)
        target_ab = target_ab.to(self.device)
        l_channel = l_channel.to(self.device)

        # Convert AB channels into fake RGB images by stacking with L channel
        pred_lab = torch.cat([l_channel, pred_ab], dim=1)  # Shape: (batch, 3, H, W)
        target_lab = torch.cat([l_channel, target_ab], dim=1)  # Shape: (batch, 3, H, W)

        # Compute VGG perceptual loss
        pred_features = self.vgg(pred_lab)
        target_features = self.vgg(target_lab)

        vgg_loss = self.criterion(pred_features, target_features)
        color_loss = self.color_loss(pred_ab, target_ab)  # Encourage stronger colors

        return vgg_loss + 0.1 * color_loss  # Balance between perceptual and color loss
