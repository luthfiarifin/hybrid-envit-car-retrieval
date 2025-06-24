import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) as described in the paper.

    This module learns to weigh the importance of each channel in a feature map.
    It uses both average-pooled and max-pooled features to compute channel attention scores.
    """

    def __init__(self, in_planes, ratio=16):
        """
        Initializes the Channel Attention Module.

        Args:
            in_planes (int): Number of input channels.
            ratio (int): Reduction ratio for the intermediate MLP layer.
        """
        super(ChannelAttention, self).__init__()

        # Global Average Pooling layer
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Global Max Pooling layer
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared Multi-Layer Perceptron (MLP)
        # The MLP consists of two fully connected (Linear) layers.
        # The first layer reduces the dimensionality by the specified ratio.
        # The second layer restores the dimensionality to the original number of channels.
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )

        # Sigmoid activation function to generate attention scores between 0 and 1.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Channel Attention Module.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Channel attention scores of shape (B, C, 1, 1).
        """
        # Apply average pooling and pass through the shared MLP
        avg_out = self.fc(self.avg_pool(x))

        # Apply max pooling and pass through the shared MLP
        max_out = self.fc(self.max_pool(x))

        # Add the outputs from both branches
        out = avg_out + max_out

        # Apply sigmoid to get the final channel attention scores
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) as described in the paper.

    This module learns to focus on important spatial regions within a feature map.
    It uses channel-wise average and max pooling to generate a spatial attention map.
    """

    def __init__(self, kernel_size=7):
        """
        Initializes the Spatial Attention Module.

        Args:
            kernel_size (int): The kernel size for the convolutional layer.
                               The paper specifies a 7x7 kernel.
        """
        super(SpatialAttention, self).__init__()

        # A 2D convolutional layer to process the concatenated feature maps.
        # The kernel size is 7x7, as specified in the paper (f_7x7).
        # Padding is set to (kernel_size - 1) / 2 to keep the spatial dimensions the same.
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

        # Sigmoid activation function to generate the spatial attention map.
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Spatial Attention Module.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Spatial attention map of shape (B, 1, H, W).
        """
        # Apply average pooling across the channel dimension.
        avg_out = torch.mean(x, dim=1, keepdim=True)

        # Apply max pooling across the channel dimension.
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate the average-pooled and max-pooled features along the channel dimension.
        x = torch.cat([avg_out, max_out], dim=1)

        # Pass the concatenated features through the convolutional layer.
        x = self.conv1(x)

        # Apply sigmoid to get the final spatial attention map.
        return self.sigmoid(x)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    This module sequentially applies Channel Attention and Spatial Attention
    to refine the input feature map.
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        """
        Initializes the CBAM.

        Args:
            in_planes (int): Number of input channels for the feature map.
            ratio (int): Reduction ratio for the Channel Attention module.
            kernel_size (int): Kernel size for the Spatial Attention module.
        """
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Forward pass for the CBAM.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Refined feature map after applying CBAM.
        """
        # Step 1: Apply Channel Attention.
        # The channel attention scores are multiplied element-wise with the input feature map.
        x = self.ca(x) * x

        # Step 2: Apply Spatial Attention.
        # The spatial attention map is multiplied element-wise with the channel-refined feature map.
        x = self.sa(x) * x

        return x


class EfficientNetB4_CBAM(nn.Module):
    """
    Hybrid EfficientNet with CBAM integrated for classification.

    This model uses a pre-trained EfficientNet-B4 as its backbone and integrates
    CBAM blocks after each stage of the feature extractor to enhance feature representation.
    """

    def __init__(self, num_classes=8, pretrained=True):
        """
        Initializes the Hybrid EfficientNet-CBAM model.

        Args:
            num_classes (int): The number of output classes (e.g., 2 for benign/malignant).
            pretrained (bool): Whether to use a pre-trained EfficientNet backbone.
        """
        super(EfficientNetB4_CBAM, self).__init__()

        # --- Backbone: EfficientNet-B4 ---
        # Load the EfficientNet-B4 model from the 'timm' library.
        # `features_only=True` returns the output of each stage.
        self.backbone = timm.create_model(
            "efficientnet_b4", pretrained=pretrained, features_only=True
        )

        # Get the number of output channels from each stage of the backbone.
        # To create CBAM blocks with the correct number of input channels.
        feature_info = self.backbone.feature_info.channels()

        # --- Attention Layers: CBAM ---
        # Create a CBAM block for each stage of the EfficientNet backbone.
        self.cbam1 = CBAM(feature_info[0])
        self.cbam2 = CBAM(feature_info[1])
        self.cbam3 = CBAM(feature_info[2])
        self.cbam4 = CBAM(feature_info[3])
        self.cbam5 = CBAM(feature_info[4])

        # --- Classifier Head ---
        # The paper describes a classifier head following the feature extraction.

        # Global Average Pooling layer to reduce spatial dimensions to 1x1.
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer with 256 hidden neurons and dropout.
        self.classifier = nn.Sequential(
            # Flatten the output from the pooling layer.
            nn.Flatten(),
            # Dropout for regularization, as mentioned in the paper (p=0.4).
            nn.Dropout(p=0.4),
            # Fully connected layer from the final feature map size to 256 neurons.
            nn.Linear(in_features=feature_info[-1], out_features=256),
            # ReLU activation function.
            nn.ReLU(),
            # Final output layer for classification.
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        # Pass input through the backbone to get features from each stage.
        features = self.backbone(x)

        # Apply CBAM to the output of each stage.
        s1 = self.cbam1(features[0])
        s2 = self.cbam2(features[1])
        s3 = self.cbam3(features[2])
        s4 = self.cbam4(features[3])
        s5 = self.cbam5(features[4])  # This is the final feature map

        # Pass the final refined feature map through the classifier head.
        pooled_output = self.global_pool(s5)
        output = self.classifier(pooled_output)

        return output
