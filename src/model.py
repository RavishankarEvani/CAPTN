import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch_geometric
from typing import List, Tuple, Optional
from src.backbone import backbone_selection
from config import CAPTNConfig


def scaling_clamp(
    x: Tensor, 
    min_val: Optional[float]=0.0, 
    max_val: Optional[float]=1.0
) -> Tensor:
    """
    Clamps the output values to the range [min_val, max_val].

    Args:
        x (Tensor): Input tensor to clamp.
        min_val (float): Lower bound of the clamp. Defaults to 0.0.
        max_val (float): Upper bound of the clamp. Defaults to 1.0.

    Returns:
        Tensor: Output clamped between range [min_val, max_val].
    """
    return torch.clamp(
        input=x, 
        min=min_val, 
        max=max_val
    )


class SpatialAttention(nn.Module):
    """
    Computes a spatial attention map by combining per-location channel-wise max and average statistics.
    
    Forward Args:
        x (Tensor): Feature maps after convolution, shape [B, C, H, W].
    
    Returns:
        Tensor: Spatial attention map, shape [B, 1, H, W].
    """
    
    conv1: nn.Conv2d
    sigmoid: nn.Sigmoid
    
    def __init__(self) -> None:
        super(SpatialAttention, self).__init__()

        # Convolution to compress concatenated max and avg maps to a single-channel attention map
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=2, 
            out_channels=1, 
            kernel_size=1, 
            padding=0, 
            bias=False
        )

        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        
        # Channel-wise average: [B, 1, H, W]
        avg_out: Tensor = torch.mean(
            input=x, 
            dim=1, 
            keepdim=True
        )

        # Channel-wise maximum: [B, 1, H, W]
        max_out: Tensor
        _indices: Tensor
        max_out, _indices = torch.max(
            input=x, 
            dim=1, 
            keepdim=True
        )

        # Concatenate along channel dimension: [B, 2, H, W]
        concat: Tensor = torch.cat(
            tensors=[avg_out, max_out], 
            dim=1
        )

        # Generate spatial attention map: [B, 1, H, W]
        conv_out: Tensor = self.conv1(concat)
        attn_map: Tensor = self.sigmoid(conv_out)

        return attn_map

    
class ConvWithSpatialAttention(nn.Module):
    """
    Applies a convolution followed by spatial attention and a residual connection.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolutional kernel size. Defaults to 1.
        stride (int): Convolutional stride. Defaults to 1.
        padding (int): Padding for convolution. Defaults to 0.
        bias (bool): Whether to use bias in convolution. Defaults to True.

    Forward Args:
        x (Tensor): Backbone feature maps, shape [B, C, H, W].

    Returns:
        Tensor: Feature maps after attention and residual addition, shape [B, out_channels, H, W].
    
    """
    
    conv: nn.Conv2d
    spatial_attention: SpatialAttention
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int=1,
        stride: int=1,
        padding: int=0,
        bias: bool = True
    ) -> None:
        super(ConvWithSpatialAttention, self).__init__()

        # Transform backbone feature maps using convolution operation
        self.conv: nn.Conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        # Compute spatial attention map over the convolved features
        self.spatial_attention: SpatialAttention = SpatialAttention()

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        
        # Convolve input: [B, C_out, H, W]
        conv_out: Tensor = self.conv(x)

        # Residual for skip connection
        residual: Tensor = conv_out

        # Compute spatial attention map: [B, 1, H, W]
        attention_map: Tensor = self.spatial_attention(conv_out)

        # Apply attention via element-wise multiplication: [B, C_out, H, W]
        attended: Tensor = conv_out * attention_map

        # Add residual connection
        out: Tensor = attended + residual

        return out



class TFA(nn.Module):
    """
    Texture Frequency Attention (TFA) module.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dropout_prob (float): Dropout probability. Defaults to 0.9.
        
    Forward Args:
        x (Tensor): Backbone feature maps, shape [B, C, H, W].

    Returns:
        Tensor: Latent Texture Attributes (LTAs), [B, out_channels, H, W].
    """
    
    conv_attn: ConvWithSpatialAttention
    attn_norm: nn.Sequential
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout_prob: float = 0.9
    ) -> None:
        super(TFA, self).__init__()

        # Apply convolution and spatial attention with residual connection
        self.conv_attn: ConvWithSpatialAttention = ConvWithSpatialAttention(
            in_channels=in_channels, 
            out_channels=out_channels
        )

        # BatchNorm + GELU + Dropout for LTA transformation
        self.attn_norm: nn.Sequential = nn.Sequential(
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        
        # Features with attention
        x_conv: Tensor = self.conv_attn(x)  
        
        # Generate Latent Texture Attributes (LTAs)
        out: Tensor = self.attn_norm(x_conv)
        
        return out


class DDP(nn.Module):
    """
    Dual Depth Permutation (D²P) module.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        dropout_prob (float): Dropout probability.
        
    Forward Args:
        x (Tensor): Backbone feature maps of shape [B, C, H, W].

    Returns:
        Tensor: Output features after depth permutation and fusion, shape [B, out_channels, H, W].
    """
    
    path1: nn.Sequential
    path2: nn.Sequential
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        dropout_prob: float
    ) -> None:
        super(DDP, self).__init__()

        self.path1: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1,
                bias=True
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_prob)
        )

        self.path2: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1,
                bias=True
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_prob)
        )

    def depth_permutation(
        self, 
        x: Tensor, 
        groups: int
    ) -> Tensor:
        """
        Perform Depth Permutation for each set of feature maps.

        Args:
            x (Tensor): Backbone feature maps, shape [B, C, H, W].
            groups (int): Number of channel groups.

        Returns:
            Tensor: Shuffled tensor, shape [B, C, H, W].
        """
        
        B, C, H, W = x.size()
        assert C % groups == 0, "Channels must be divisible by number of groups"

        x = x.view(B, groups, C // groups, H, W)
        x = x.transpose(1, 2).contiguous()
        
        return x.view(B, C, H, W)

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        """
        Forward pass for D²P module.

        Args:
            x (Tensor): Backbone feature maps, shape [B, C, H, W].

        Returns:
            Tensor: Output features after depth permutation and fusion, shape [B, out_channels, H, W].
        """
        
        x1: Tensor = self.path1(self.depth_permutation(x, 4))
        x2: Tensor = self.path2(self.depth_permutation(x, 8))
        
        out: Tensor = x1 + x2
        
        return out



class ELTA(nn.Module):
    """
    Enhanced Latent Texture Attributes (ELTA) module that combines outputs from the TFA and D²P modules.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output enhanced LTAs.
        dropout_prob (float): Dropout probability. Defaults to 0.9.
        
    Forward Args:
        x (Tensor): Backbone feature maps, shape [B, C, H, W].

    Returns:
        Tensor: Enhanced Latent Texture Attributes (ELTAs), shape [B, out_channels, H, W].
    """
    
    ddp: DDP
    tfa: TFA
    scaling_tfa: nn.Parameter
    scaling_ddp: nn.Parameter
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_prob: float = 0.9
    ) -> None:
        super(ELTA, self).__init__()


        # Dual Depth Permutation (D²P) module.
        self.ddp: DDP = DDP(
            in_channels, 
            out_channels, 
            dropout_prob
        )
        
        # Texture Frequency Attention (TFA) module.
        self.tfa: TFA = TFA(
            in_channels, 
            out_channels, 
            dropout_prob
        )

        # Weights to scale outputs from TFA and D²P.
        self.scaling_tfa: nn.Parameter = nn.Parameter(torch.tensor(1.0))
        self.scaling_ddp: nn.Parameter = nn.Parameter(torch.tensor(0.4))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for ELTA module.

        Args:
            x (Tensor): Backbone feature maps, shape [B, C, H, W].

        Returns:
            Tensor: Output enhanced LTAs.
        """
        tfa_out: Tensor = scaling_clamp(self.scaling_tfa, 0.0, 1.0) * self.tfa(x)
        ddp_out: Tensor = scaling_clamp(self.scaling_ddp, 0.0, 1.0) * self.ddp(x)
        
        out: Tensor = tfa_out + ddp_out
        
        return out



class LCP(nn.Module):
    """
    Learnable Chebyshev Polynomial (LCP) transformation module.

    Args:
        degree (int): Degree of the polynomial expansion.
        
    Forward Args:
        x (Tensor): Orderless Enhanced LTAs, shape [B, C].

    Returns:
        Tensor: Transformed representation, shape [B, C].
    """
    degree: int
    coefficients: nn.Parameter

    def __init__(
        self, 
        degree: int
    ) -> None:
        super(LCP, self).__init__()
        
        self.degree: int = degree

        # Learnable coefficients to provide more flexibility.
        self.coefficients: nn.Parameter = nn.Parameter(torch.ones(degree + 1))

    def _chebyshev_polynomial(
        self, x: Tensor, 
        n: int
    ) -> Tensor:
        """
        Recursively computes the n-th order Chebyshev polynomial

        Args:
            x (Tensor): Orderless enhanced LTAs, shape [B, C].
            n (int): Degree of the polynomial.

        Returns:
            Tensor: Chebyshev polynomial of degree n applied element-wise to x, shape [B, C].
        """
        if n == 0:
            return torch.ones_like(x)
        elif n == 1:
            return x
        else:
            T0: Tensor = torch.ones_like(x)  
            T1: Tensor = x                   
            for _ in range(2, n + 1):
                T2: Tensor = 2 * x * T1 - T0 
                T0, T1 = T1, T2
            return T1

    def forward(
        self, 
        x: Tensor
    ) -> Tensor:
        """
        Applies the Learnable Chebyshev Polynomial (LCP) transformation.

        Args:
            x (Tensor): Orderless enhanced LTAs, shape [B, C].

        Returns:
            Tensor: Transformed representation, shape [B, C].
        """
        # Apply sigmoid to learnable coefficients
        coeffs: Tensor = torch.sigmoid(self.coefficients)

        # Compute weighted sum
        result: Tensor = coeffs[0] * self._chebyshev_polynomial(x, 0)
        for i in range(1, self.degree + 1):
            result += coeffs[i] * self._chebyshev_polynomial(x, i)

        return result



class CAPTN(nn.Module):
    """
    Chebyshev Attention Depth Permutation Texture Network (CAPTN).
    
    Args:
        cfg (CAPTNConfig): Configuration object for the model.
        n_classes (int): Number of output classes.
        depth_dims (List[int]): Channel dimensions at various backbone depths.
        patch_size (int): Patch size needed for Spatial Latent Attribute Representation (SLAR).
        regional_attribute_start_index (int): Start index for regional crop.
        regional_attribute_end_index (int): End index for regional crop.
        embedding_dim (int): Dimensionality of concatenated orderless LTAs for classification.
        backbone_name (str): Name of the backbone architecture to use.
        fine_tune_backbone (bool): Whether to fine-tune or freeze the backbone.
        
    Forward Args:
        x (Tensor): Input image tensor, shape [B, C, H, W].
        return_patch (bool): Whether to return Spatial Latent Attribute Representation (SLAR) for LTA loss.

    Returns:
        Tuple[Tensor, Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor, Tensor]:
            - classification_output (Tensor): Classification logits, shape [B, num_classes].
            - obfm_combined (Tensor): Orderless backbone feature map (OBFM) vector, shape [B, D].
            - oelta_combined (Tensor): Orderless enhanced LTAs, shape [B, D].
            - slar_output (Tensor): Spatial Latent Attribute Representation (SLAR), shape [B, 3, patch_size, patch_size], only returned if `return_patch=True`.
    """
    
    cfg: CAPTNConfig
    n_classes: int
    patch_size: int
    regional_attribute_start_index: int
    regional_attribute_end_index: int

    backbone: nn.Module
    elta: nn.ModuleList # Enhanced LTA
    bn: nn.ModuleList # BatchNorm
    lcp: nn.ModuleList # Learnable Chebyshev Polynomial

    classifier: torch_geometric.nn.dense.linear.Linear
    slar: nn.Sequential # SLAR: Spatial Latent Attribute Representation

    def __init__(
        self,
        cfg: CAPTNConfig,
        n_classes: int,
        depth_dims: List[int],
        patch_size: int,
        regional_attribute_start_index: int,
        regional_attribute_end_index: int,
        embedding_dim: int,
        backbone_name: str,
        fine_tune_backbone: bool
    ) -> None:
        super().__init__()

        self.cfg: CAPTNConfig = cfg
        self.n_classes: int = n_classes
        self.patch_size: int = patch_size
        self.regional_attribute_start_index: int = regional_attribute_start_index
        self.regional_attribute_end_index: int = regional_attribute_end_index

        # Backbone selection
        self.backbone: nn.Module = backbone_selection(backbone_name)

        # Select layers
        selected_dims: List[int] = depth_dims[-self.cfg.common.layers:]

        # Enhanced LTA
        self.elta: nn.ModuleList = nn.ModuleList([
            ELTA(in_channels=dim, out_channels=dim)
            for dim in selected_dims
        ])

        self.bn: nn.ModuleList = nn.ModuleList([
            nn.BatchNorm2d(dim) for dim in selected_dims
        ])

        # Learnable Chebyshev Polynomial (LCP)
        self.lcp: nn.ModuleList = nn.ModuleList([
            LCP(cfg.chebyshev_polynomial_degree) for _ in selected_dims
        ])

        self.classifier: torch_geometric.nn.dense.linear.Linear = torch_geometric.nn.dense.linear.Linear(
            embedding_dim, 
            n_classes, 
            bias=False, 
            weight_initializer='glorot'
        )

        # SLAR: Spatial Latent Attribute Representation
        self.slar: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                selected_dims[-1] * 2, 
                3, 
                kernel_size=3, 
                padding=0
            )
        )

        if not fine_tune_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self) -> None:
        """
        Freezes all parameters in the backbone.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(
        self,
        x: Tensor,
        return_patch: bool = False
    ) -> Tuple[Tensor, ...]:
        """
        Forward pass for CAPTN.

        Args:
            x (Tensor): Input image tensor of shape [B, C, H, W].
            return_patch (bool, optional): Whether to return Spatial Latent Attribute Representation (SLAR) for LTA loss.

        Returns:
            Tuple[Tensor, Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor, Tensor]:
                - classification_output (Tensor): Classification logits of shape [B, num_classes].
                - obfm_combined (Tensor): Orderless backbone feature map (OBFM) vector of shape [B, D].
                - oelta_combined (Tensor): Orderless enhanced LTAs of shape [B, D].
                - slar_output (Tensor, optional): Spatial Latent Attribute Representation (SLAR) of shape [B, 3, patch_size, patch_size], only returned if `return_patch=True`.
        """
        
        # Extract features maps from backbone
        x_list: List[Tensor] = self.backbone(x)[-self.cfg.common.layers:]

        # Generate enhanced LTAs
        elta_outputs: List[Tensor] = [elta(xi) for elta, xi in zip(self.elta, x_list)]

        obfm_outs: List[Tensor] = []
        oelta_outs: List[Tensor] = []
        lcp_outs: List[Tensor] = []

        for lcp, elta_out, xi, bn in zip(self.lcp, elta_outputs, x_list, self.bn):
            
            obfm: Tensor = bn(xi).mean(dim=(-2, -1))           
            oelta: Tensor = elta_out.mean(dim=(-2, -1))       
            
            # LCP expansion
            lcp_transformed: Tensor = lcp(oelta)               

            obfm_outs.append(obfm)
            oelta_outs.append(oelta)
            lcp_outs.append(lcp_transformed)

        obfm_combined: Tensor = torch.cat(obfm_outs, dim=1)
        oelta_combined: Tensor = torch.cat(oelta_outs, dim=1)
        lcp_combined: Tensor = torch.cat(lcp_outs, dim=1)

        classification_output: Tensor = self.classifier(lcp_combined)

        if return_patch:
            # Concatenate the enhanced LTAs with final layer's original feature maps from the frozen backbone
            patch_input: Tensor = torch.cat([elta_outputs[-1], x_list[-1]], dim=1)

            # The crop region is adjustable using a delta offset
            delta: int = self.cfg.delta
            start: int = self.regional_attribute_start_index + delta
            end: int = self.regional_attribute_end_index + delta
            cropped: Tensor = patch_input[:, :, start:end, start:end]

            # Generate Spatial Latent Attribute Representation (SLAR) for LTA loss
            slar_output: Tensor = self.slar(cropped)
            slar_output: Tensor = F.interpolate(
                slar_output,
                size=(self.patch_size, self.patch_size),
                mode="bilinear",
                align_corners=False
            )
            return classification_output, slar_output, obfm_combined, oelta_combined

        return classification_output, obfm_combined, oelta_combined