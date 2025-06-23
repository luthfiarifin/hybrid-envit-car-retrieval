import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models


class Conv2d(nn.Module):
    """A helper class for a standard Conv->BatchNorm->SiLU block."""

    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size=3,
        padding=1,
        stride=1,
        groups=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SE(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, in_chan, r=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_chan, in_chan // r, kernel_size=1, padding=0),
            nn.SiLU(),
            nn.Conv2d(in_chan // r, in_chan, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excitation(self.squeeze(x))


class Fused_MBConv(nn.Module):
    """The Fused MBConv block used in the early stages of EfficientNetV2."""

    def __init__(self, in_chan, out_chan, expansion=2):
        super().__init__()
        self.skip_conn = in_chan == out_chan
        expanded_chan = in_chan * expansion
        self.conv = nn.Sequential(
            Conv2d(in_chan, expanded_chan, kernel_size=3, stride=1, padding=1),
            SE(expanded_chan),
            nn.Conv2d(expanded_chan, out_chan, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )

    def forward(self, x):
        if self.skip_conn:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MBConv(nn.Module):
    """The standard MBConv block with depthwise separable convolutions."""

    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, expansion=2):
        super().__init__()
        self.skip_conn = (in_chan == out_chan) and (stride == 1)
        expanded_chan = in_chan * expansion
        self.conv = nn.Sequential(
            Conv2d(in_chan, expanded_chan, kernel_size=1, padding=0),
            Conv2d(
                expanded_chan,
                expanded_chan,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded_chan,
            ),
            SE(expanded_chan),
            nn.Conv2d(expanded_chan, out_chan, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_chan),
        )

    def forward(self, x):
        if self.skip_conn:
            return x + self.conv(x)
        else:
            return self.conv(x)


class RMSNorm(nn.Module):
    """RMSNorm for lighter computation than LayerNorm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class MultiHeadAttention(nn.Module):
    """Grouped-Query Attention from the paper."""

    def __init__(self, in_dim, num_query_heads=8, num_kv_heads=2):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = in_dim // num_query_heads
        kv_dim = self.head_dim * self.num_kv_heads
        self.q_norm = RMSNorm(in_dim)
        self.k_norm = RMSNorm(in_dim)
        self.to_q = nn.Linear(in_dim, in_dim, bias=False)
        self.to_k = nn.Linear(in_dim, kv_dim, bias=False)
        self.to_v = nn.Linear(in_dim, kv_dim, bias=False)
        self.to_out = nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x):
        b, n, _ = x.shape
        q_normed = self.q_norm(x)
        k_normed = self.k_norm(x)
        q = self.to_q(q_normed)
        k = self.to_k(k_normed)
        v = self.to_v(k_normed)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_query_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_kv_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_kv_heads)
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
        sim = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(sim, "b h n d -> b n (h d)")
        return self.to_out(out)


class FFN(nn.Module):
    """Feed Forward Network with SiLU activation."""

    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(dim * 2)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)))


class Encoder(nn.Module):
    """The Transformer Encoder block."""

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.norm1 = RMSNorm(embed_dim)
        self.mhsa = MultiHeadAttention(embed_dim, num_query_heads=num_heads)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = FFN(embed_dim)

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class HybridEfficientNetViT(nn.Module):
    """
    The full hybrid model architecture inspired by the paper.
    It interleaves CNN blocks (MBConv) and Transformer blocks (Encoder).
    This model is lightweight (~15.6M parameters).
    """

    def __init__(
        self,
        num_classes=8,
        embed_dim=192,
        num_heads=8,
        dropout=0.2,
        pretrained=False,
        freeze_stem=False,
    ):
        """
        Args:
            ...
            pretrained (bool): If True, loads weights for the CNN stem from a pre-trained EfficientNetV2-S.
            freeze_stem (bool): If True, freezes the weights of the pre-trained stem during training.
        """
        super().__init__()

        # CNN Stem Block
        self.stage1 = nn.Sequential(
            Conv2d(3, 24, stride=2), Fused_MBConv(24, 32, expansion=2)
        )

        # Transfer Learning Block
        if pretrained:
            print("Loading pre-trained weights for the Hybrid Model's CNN stem.")
            # Load a standard pre-trained model to borrow weights from
            pretrained_model = models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.DEFAULT
            )

            # Manually copy the state_dict from the first Conv layer
            # This works because our custom Conv2d and torchvision's Conv2dNormActivation[0] are both nn.Conv2d
            self.stage1[0].conv.load_state_dict(
                pretrained_model.features[0][0].state_dict()
            )

            if freeze_stem:
                print("Freezing the CNN stem.")
                for param in self.stage1.parameters():
                    param.requires_grad = False

        # Patch Embedding Block
        self.patch_embed1 = MBConv(32, embed_dim, kernel_size=16, stride=8)

        # Transformer Block 1
        self.block1_transformer = nn.Sequential(
            *[Encoder(embed_dim, num_heads) for _ in range(3)]
        )

        # CNN Block 1
        self.block1_cnn = nn.Sequential(
            *[MBConv(embed_dim, embed_dim, expansion=2) for _ in range(3)]
        )

        # Repatch Block 1
        self.repatch1 = MBConv(embed_dim, embed_dim * 2, kernel_size=5, stride=2)

        # Transformer Block 2
        self.block2_transformer = nn.Sequential(
            *[Encoder(embed_dim * 2, num_heads) for _ in range(6)]
        )

        # CNN Block 2
        self.block2_cnn = nn.Sequential(
            *[MBConv(embed_dim * 2, embed_dim * 2, expansion=2) for _ in range(6)]
        )

        # Repatch Block 2
        self.repatch2 = MBConv(embed_dim * 2, embed_dim * 4, kernel_size=3, stride=2)

        # Transformer Block 3
        self.block3_transformer = nn.Sequential(
            *[Encoder(embed_dim * 4, num_heads) for _ in range(2)]
        )

        # Pooling Block
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification Head Block
        self.head = nn.Sequential(
            nn.Dropout(dropout, inplace=True), nn.Linear(embed_dim * 4, num_classes)
        )

    def forward(self, x):
        # Forward through CNN Stem
        x = self.stage1(x)

        # Forward through Patch Embedding
        x = self.patch_embed1(x)

        # Forward through Transformer Block 1
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.block1_transformer(x)

        # Forward through CNN Block 1
        x = rearrange(x, "b (h w) c -> b c h w", h=int(x.shape[1] ** 0.5))
        x = self.block1_cnn(x)

        # Forward through Repatch Block 1
        x = self.repatch1(x)

        # Forward through Transformer Block 2
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.block2_transformer(x)

        # Forward through CNN Block 2
        x = rearrange(x, "b (h w) c -> b c h w", h=int(x.shape[1] ** 0.5))
        x = self.block2_cnn(x)

        # Forward through Repatch Block 2
        x = self.repatch2(x)

        # Forward through Transformer Block 3
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.block3_transformer(x)

        # Pooling and Head
        x = rearrange(x, "b n c -> b c n")
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.head(x)

        # Final output
        return x


# The Standard EfficientNet-S for Comparison
class StandardEfficientNetS(nn.Module):
    """
    A wrapper for the standard torchvision EfficientNet-S model.
    This is used as a baseline for comparison, as done in the paper's notebook.
    This model has ~20.1M parameters.
    """

    def __init__(
        self,
        num_classes=8,
        pretrained=False,
    ):
        """
        Args:
            num_classes (int): Number of output classes for your project.
            pretrained (bool): If True, loads weights pre-trained on ImageNet.
        """
        super().__init__()

        if pretrained:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            print("Loading EfficientNetV2-S with pre-trained ImageNet weights.")
        else:
            weights = None
            print("Initializing EfficientNetV2-S from scratch.")

        self.model = models.efficientnet_v2_s(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
