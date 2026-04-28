import torch
import torch.nn as nn
import numpy as np
from timm.models.swin_transformer import SwinTransformerBlock

class PatchEmbedding(nn.Module):
    """
    The Patch Embedding layer.
    """

    def __init__(self,
                 in_channels:int=3,
                 dim:int=96,
                 patch_size:int=4):
        
        """
        Initialises a Patch Embedding module.

        Args:
            `in_channels (int)`: The colour depth of the input image expected. Default: 3
            `dim (int)`: The quantity of embedding dimensions to produced. Default: 96
            `patch_size (int)`: The height and width of each patch. Default: 4.
        """

        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.dim = dim

        # 2D Convolution produces spatial grid of patches
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=patch_size, padding=0)

        # Flatten layer to produce sequence of patches.
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

        # Apply normalisation to patches
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Takes each image input and creates non-overlapping patches as embeddings before flattening the patches to a sequential input.

        Args:
            `x (torch.Tensor)`: Input image of shape (B, C, H, W), where H and W 
                must be divisible by patch_size.
        Returns:
            `torch.Tensor`: Patch sequence of shape (B, N, dim), where
                N = (H/patch_size) * (W/patch_size). 

        Raises:
            AssertionError: If image resolution is not divisible by patch size.
        """

        image_resolution = x.shape[-1]

        assert image_resolution % self.patch_size == 0, \
            f"Input image size must be divisible by patch size, \
            image shape: {image_resolution}, patch size: {self.patch_size}"
        
        x_patched = self.patcher(x)

        x_flattened = self.flatten(x_patched)

        x = x_flattened.permute(0,2,1)

        x = self.norm(x)

        return x
    

class PatchMerging(nn.Module):
    """
    The Patch Merging layer proceeding each Swin-Xception layer.
    """

    def __init__(self, dim):
        """
        Initialises a Patch Merging module.

        Args:
        - `dim (int)`: The current quantity of embedding dimensions.
        """

        super(PatchMerging, self).__init__()

        self.dim = dim

        # Apply normalisation to concatenated patches
        self.norm = nn.LayerNorm(4*dim)

        # Linear projection layer to reduce dimensionality before returning double channels
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)

    def forward(self, x):
        """
        Takes sequential tokens from the SwinX block,
        rearranges the tokens to achieve the spatial information in the form of a grid,
        splits the grid into a 2*2 neighbourhood of patches,
        and concatenates the grids, increasing dimensionality by a factor of 4.
        Linear projection halves spatial grid and doubles channels.

        Args:
            `x (torch.Tensor)`: Patch sequence of shape (B, N, C), where N = H*W 
                and H, W are assumed even.    
    
        Returns:
            `torch.Tensor`: Downsampled Sequence of shape (B, (H/2)*(W/2), 2C).

        Raises:
            AssertionError: If either the H or W derived from N are not even.
        """

        # Cast the data structure of the sequence input into variables
        # B - Batch number
        # N - Sequence length
        # C - Channel quantity
        B, N, C = x.shape

        # Derive the height (H) and width (W) from the square root of N
        H = W = int(np.sqrt(N))

        assert H % 2 == 0 and W % 2 == 0, f"H and W must be even, got H={H}, W={W}"

        # Reshape sequential tokens to spatial grid
        x = x.view(B, H, W, C)

        # Split spatial grid of tokens into 4 groups
        x0 = x[:, 0::2, 0::2, :] # top left
        x1 = x[:, 1::2, 0::2, :] # top right
        x2 = x[:, 0::2, 1::2, :] # bottom left
        x3 = x[:, 1::2, 1::2, :] # bottom right

        # Concatenate the groups of tokens
        x = torch.cat([x0,x1,x2,x3], dim=-1)

        # Rearrange the spatial grid of patches back into a sequence
        # flattens spatial grid back to sequence; each token now carries 4C channels from its 2*2 neighbourhood
        x = x.view(B, -1, 4*C)

        x = self.norm(x)

        # The linear projection squeezes the dimensionality from 4C to 2C
        x = self.reduction(x)

        return x
    

class DepthwiseSeparableConv(nn.Module):
    """
    The Depthwise Separable Convolution operator, present in each Depthwise Separable Feed-Forward Network.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size:int=3):
        """
        Initialises a Depthwise-Separable Convolution layer.

        Args:
        - `in_channels (int)`: The incoming channel quantity.
        - `out_channels (int)`: The resulting channel quantity.
        - `kernel_size (int)`: Kernel of odd size to enable correct symmetric padding. Default: 3.
        """

        super(DepthwiseSeparableConv, self).__init__()

        # Achieves a depthwise separable convolution, by setting out_channels=in_channels to keep the same number of channels
        # after the convolution (produces `in_channels` feature maps). groups=in_channels ensures the kernel is applied to each
        # channel separately.
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=kernel_size//2, groups=in_channels)
        
        # Depthwise-separable and pointwise operations are separated by GELU activation
        self.gelu = nn.GELU()

        # Achieves a pointwise convolution by utilising a kernel size of 1 on the channels.
        # Channels are not grouped separately
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Comprises one pair of
        depthwise-separable and pointwise convolutions; this process expands to/reduces from the MLP
        ratio in the FFN

        Args:
            `x (torch.Tensor)`: Spatial grid of patches with dimensions (B, (C/6C), H, W)

        Returns:
            `torch.Tensor`: Spatial grid of patches with dimensions (B, (6C/C), H, W)
        """

        x = self.depthwise(x)

        x = self.gelu(x)

        x = self.pointwise(x)

        return x
    

class DepthwiseSeparableFFN(nn.Module):
    """
    The Depthwise Separable Feed-Forward Network (The Xception component)

    Replaces the MLP head at the end of each Swin Block.
    """

    def __init__(self,
                 dim,
                 mlp_ratio:int=6,
                 dropout:float=0.25):
        """
        Initialises a DS-FFN

        Args:
        - `dim (int)`: The embedding dimension quantity
        - `mlp_ratio (int)`: The expansion factor for which the pointwise convolution widens/narrows the channel count. Default: 6.
        - `dropout (float)`: The dropout probability inbetween each depthwise separable convolution layer. Default: 0.25.
        """

        super(DepthwiseSeparableFFN, self).__init__()

        # hidden_dim represents the total number of channels anticipated after the pointwise convolution produces more channels
        # The number of embeddings in the hidden layer is calculated by multiplying the current number of dimensions by the expansion ratio
        hidden_dim = int(dim * mlp_ratio)

        # Calls Depthwise Separable Convolution layer, which contains one pair of depthwise-separable and pointwise convolutions
        # The first layer expands the dimensions by mlp_ratio
        self.depthwise1 = DepthwiseSeparableConv(dim, hidden_dim, kernel_size=3)

        # Apply dropout
        self.dropout1 = nn.Dropout(dropout)

        # The second layer reduces the dimensions by the same ratio to achieve the same number of dimensions as before.
        self.depthwise2 = DepthwiseSeparableConv(hidden_dim, dim, kernel_size=3)

        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Rearranges the sequence of patches into a spatial grid, before expanding and reducing
        the channel quantity through a pair of depthwise-separable convolution layers.

        Args:
            `x (torch.Tensor)`: Patch sequence of shape (B, N, C), where N = H*W 
                and H, W are assumed even. 

        Returns:
            `torch.Tensor`: Patch sequence of shape (B, N, C).
        """

        # Cast the data structure of the sequence input into variables
        # B - Batch number
        # N - Sequence length
        # C - Channel quantity
        B, N, C = x.shape

        # Derive the height (H) and width (W) from the square root of N
        H = W = int(np.sqrt(N))

        # Swap the positions of C and N in the input
        # Replace N with H and W to achieve a spatial grid
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.depthwise1(x)

        x = self.dropout1(x)

        x = self.depthwise2(x)

        x = self.dropout2(x)

        # Revert changes to the structure of the input
        x = x.reshape(B, C, N).transpose(1, 2)

        return x

class SwinXceptionBlock(nn.Module):
    """
    Swin-Xception Block.

    Combines the Shifted Window Multi-head Attention Mechanism of Swin
    with a depthwise-separable feed-forward network (The Xception) in 
    place of the standard FFN.
    """
    def __init__(self,
                 embedding_dim,
                 num_heads,
                 input_resolution,
                 window_size:int=7,
                 shift_size:int=0,
                 mlp_ratio:int=6):
        """
        Initialises a Swin-Xception Block.

        Args:
            `embedding_dim (int)`: Number of input embedding dimensions (C).
            `num_heads (int)`: Number of attention heads in the Window Multi-Head Self-Attention module.
            `input_resolution (tuple[int, int])`: Spatial resolution (H, W) of the input feature map, used to compute attention masks for cyclic shifting.
            `window_size (int)`: Height and widht of each local attention window. Default: 7.
            `shift_size (int)`: Number of tokens to cyclically shift the feature map before windowed attention. 0 for W-MSA, window_size//2 for SW-MSA. Default: 0
            `mlp_ratio (int)`: The expansion factor for the hidden dimension in the Depthwise Separable FFN. Default: 6.
        """

        super(SwinXceptionBlock, self).__init__()

        self.block = SwinTransformerBlock( # Use timm's Swin Transformer Block
            dim=embedding_dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            proj_drop=0.1,
            drop_path=0.1
        )

        self.input_resolution = input_resolution

        self.block.mlp = DepthwiseSeparableFFN(embedding_dim, mlp_ratio) # But replace the mlp head with my DS-FFN

    def forward(self, x): # X input is sequential transformer data
        """
        Computes Multi-head Self Attention and Depthwise Separable Convolutions via
        fixed-size non-overlapping windows containing patches. The feature map is then
        cyclically shifted to compute attention across window boundaries.

        Args:
            `x (torch.Tensor)`: Patch sequence of shape (B, N, C), where N = H*W 
                and H, W are assumed even. 

        Returns:
            `torch.Tensor`: Patch sequence of shape (B, N, C), unchanged in shape 
                from the input after residual addition.
        """

        # Cast the data structure of the sequence input into variables
        # B - Batch number
        # N - Sequence length
        # C - Channel quantity
        B, N, C = x.shape

        H, W = self.input_resolution # height and width

        # Reshape sequential tokens to spatial grid
        x = x.view(B, H, W, C)

        # Swin blocks accept inputs as spatial grid
        x = self.block(x)

        # back to sequential data
        x = x.view(B, N, C)
        
        return x
    
class SwinXception(nn.Module):
    """
    The entire backbone for the Swin-Xception architecture. My implementation utilises 6 blocks
    on the third layer, 3 heads for MSA computation, and an initial embedding count of 96
    """

    def __init__(self, num_classes:int=7, dropout:float=0.5):
        """
        Initialises the Swin-Xception backbone

        Args:
            `num_classes (int)`: The total classes the model is predicting. Default: 7 for my datasets.
            `dropout (float)`: The dropout probability before linear projection in the model's head.
        """

        super(SwinXception, self).__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(in_channels=3, dim=96, patch_size=4)

        # Stage 1
        self.layer1 = nn.ModuleList([SwinXceptionBlock(96, num_heads=3, input_resolution=(56, 56), shift_size=0 if i % 2 == 0 else 3) for i in range(2)]) # Parity to enable window shift
        self.merge1 = PatchMerging(96) # Reduce dimensions, increase channels

        # Stage 2
        self.layer2 = nn.ModuleList([SwinXceptionBlock(192, num_heads=6, input_resolution=(28, 28), shift_size=0 if i % 2 == 0 else 3) for i in range(2)])
        self.merge2 = PatchMerging(192)

        # Stage 3 - Utilises 6 blocks
        self.layer3 = nn.ModuleList([SwinXceptionBlock(384, num_heads=12, input_resolution=(14, 14), shift_size=0 if i % 2 == 0 else 3) for i in range(6)])
        self.merge3 = PatchMerging(384)

        # Stage 4
        self.layer4 = nn.ModuleList([SwinXceptionBlock(768, num_heads=24, input_resolution=(7, 7), shift_size=0 if i % 2 == 0 else 3) for i in range(2)])

        # Normalisation
        self.norm = nn.LayerNorm(768)
        
        # Average Pooling
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1)

        # MLP Head
        self.head = nn.Sequential(
            nn.Dropout(dropout), # Apply dropout before Linear Projection to class values
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        """
        Args:
            `x (torch.Tensor)`: Input image of shape (B, C, H, W).

        Returns:
            `torch.Tensor`: Class logits of shape (B, num_classes), where each value 
                is the unnormalised score for the corresponding class.
        """

        # x = [B, 3, 224, 224]
        x = self.patch_embed(x) # x = [B, 3136, 96]

        # Stage 1: 2 Swin-X blocks
        for block in self.layer1:
            x = block(x)
        x = self.merge1(x) # x = [B, 784, 192]

        # Stage 2: 2 Swin-X blocks
        for block in self.layer2:
            x = block(x)
        x = self.merge2(x) # x = [B, 196, 384]

        # Stage 3: 6 Swin-X blocks
        for block in self.layer3:
            x = block(x)
        x = self.merge3(x) # x = [B, 49, 768]

        # Stage 4: 2 Swin-X blocks
        for block in self.layer4:
            x = block(x)

        # Normalisation and Average Pooling
        x = self.norm(x)
        x = x.transpose(1, 2) # x = [B, 768, 49]
        x = self.avgpool1d(x) # x = [B, 768, 1]
        x = torch.flatten(x, 1) # x = [B, 768]

        # Linear Projection
        x = self.head(x) # x = [B, 7]

        return x