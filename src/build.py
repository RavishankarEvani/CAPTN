from src.model import CAPTN
from config import CAPTNConfig
import torch.nn as nn

def build_model(
        cfg: CAPTNConfig, 
        num_classes: int
) -> nn.Module:
        
    """
    Returns a Chebyshev Attention Depth Permutation Texture Network (CAPTN) model based on the given configuration.

    Args:
        cfg (CAPTNConfig): Configuration object containing model and training settings.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: An instance of the CAPTN model configured as per settings in the configuration object.
    """
    
    cfg_model = cfg.common[cfg.backbone]

    model = CAPTN(
            cfg = cfg,
            n_classes = num_classes,
            depth_dims = cfg_model.depth_dims,
            patch_size = cfg_model.patch_size,
            regional_attribute_start_index = cfg_model.regional_attribute_start_index,
            regional_attribute_end_index = cfg_model.regional_attribute_end_index,
            embedding_dim = cfg_model.embedding_dim,
            backbone_name = cfg.backbone,
            fine_tune_backbone = cfg_model.fine_tune_backbone,
            )
    
    return model
