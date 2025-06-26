from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Optimizer:
    lr: float
    weight_decay: float

@dataclass
class Loss:
    alpha: float
    gamma: float
    label_smoothing: float

@dataclass
class Scheduler:
    eta_min_scaling_factor: float

@dataclass
class Training:
    seed: int
    batch_size: int
    start_epoch: int
    num_epochs: int
    split: str

@dataclass
class Testing:
    batch_size: int

@dataclass
class Accelerator:
    cuda: bool
    device: str
    resume: bool

@dataclass
class Paths:
    data: str
    log: str
    snapshot: str

@dataclass
class Tracking:
    uri: str

@dataclass
class ConvnextConfig:
    depth_dims: List[int]
    spatial_dims: List[int]
    depth_compression_ratio: int
    embedding_dim: int
    backbone_name: str
    fine_tune_backbone: bool
    patch_size: int
    regional_attribute_start_index: int
    regional_attribute_end_index: int

@dataclass
class CAPTNConfig:
    optimizer: Optimizer
    loss: Loss
    scheduler: Scheduler
    training: Training
    testing: Testing
    accelerator: Accelerator
    paths: Paths
    tracking: Tracking
    convnext_nano: ConvnextConfig
    convnext_tiny: ConvnextConfig
    convnext_base: ConvnextConfig
    convnext_large: ConvnextConfig
    experiment: str
    backbone: str
    architecture_name: str
    additional_augmentation: bool
    layers: int
    delta: int
    chebyshev_polynomial_degree: int
    common: Dict[str, ConvnextConfig]
