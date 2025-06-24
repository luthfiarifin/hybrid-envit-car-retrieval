import torch
from torch import nn
from torchvision import models


class ChannelAttention(nn.Module):
    """
    Channel Attention Module: Decides 'what' features are important.
    It learns to assign different weights to different feature channels.
    """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # A small MLP to learn the channel weights
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module: Decides 'where' to focus in the feature map.
    It highlights the most informative spatial regions.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    The complete CBAM block, which applies Channel and then Spatial attention.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # Apply channel attention first
        x = x * self.ca(x)
        # Then apply spatial attention
        x = x * self.sa(x)
        return x


class EfficientNetB4_CBAM(nn.Module):
    """
    A powerful and efficient architecture for fine-grained classification.
    It uses a pre-trained EfficientNet-B4 backbone and enhances its feature
    extraction capabilities by integrating CBAM attention modules.
    UPDATED: Uses a custom forward pass for robust CBAM injection.
    """

    def __init__(self, num_classes=8, pretrained=True):
        """
        Args:
            num_classes (int): The number of output classes for your car dataset.
            pretrained (bool): If True, loads weights pre-trained on ImageNet.
        """
        super(EfficientNetB4_CBAM, self).__init__()

        # Load the standard EfficientNet-B4 model
        if pretrained:
            weights = models.EfficientNet_B4_Weights.DEFAULT
            print("Loading EfficientNet-B4 with pre-trained ImageNet weights.")
        else:
            weights = None
            print("Initializing EfficientNet-B4 from scratch.")

        self.backbone = models.efficientnet_b4(weights=weights)

        # --- Define CBAM modules for injection ---
        # These will be applied manually in the forward pass.
        # Channel numbers are based on the output of EfficientNet-B4 stages.
        self.cbam1 = CBAM(48)  # After features[1]
        self.cbam2 = CBAM(64)  # After features[2]
        self.cbam3 = CBAM(128)  # After features[3]

        # --- Extract feature stages ---
        self.features_stage0 = self.backbone.features[0]
        self.features_stage1 = self.backbone.features[1]
        self.features_stage2 = self.backbone.features[2]
        self.features_stage3 = self.backbone.features[3]
        self.features_stage4_to_end = self.backbone.features[4:]

        # The final parts of the original model remain the same
        self.avgpool = self.backbone.avgpool

        # Replace the final classifier
        in_features = self.backbone.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, num_classes),
        )

        self.backbone.classifier = nn.Identity()

    def forward(self, x):
        # --- Custom Forward Pass with CBAM Injection ---
        x = self.features_stage0(x)

        x = self.features_stage1(x)
        x = self.cbam1(x)  # Inject CBAM 1

        x = self.features_stage2(x)
        x = self.cbam2(x)  # Inject CBAM 2

        x = self.features_stage3(x)
        x = self.cbam3(x)  # Inject CBAM 3

        x = self.features_stage4_to_end(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
