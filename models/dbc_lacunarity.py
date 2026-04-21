"""
DBC Lacunarity model for texture analysis.

This module implements the Differential Box Counting (DBC) method
for computing lacunarity features.
"""

import torch
import torch.nn as nn
from typing import Optional


class DBC_Lacunarity(nn.Module):
    """
    Differential Box Counting Lacunarity model.
    
    This model computes lacunarity features using the DBC method,
    which is useful for texture analysis in plant images.
    """
    
    def __init__(self, model_name: str = 'Net', window_size: int = 3, eps: float = 1e-6):
        """
        Initialize DBC Lacunarity model.
        
        Args:
            model_name: Name of the model
            window_size: Size of the sliding window
            eps: Small value to avoid division by zero
        """
        super(DBC_Lacunarity, self).__init__()
        self.window_size = window_size
        self.normalize = nn.Tanh()
        self.num_output_channels = 3
        self.eps = eps
        self.r = 1
        self.model_name = model_name
        self.max_pool = nn.MaxPool2d(kernel_size=self.window_size, stride=1)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DBC Lacunarity model.
        
        Args:
            image: Input image tensor [B, C, H, W]
            
        Returns:
            Lacunarity features tensor
        """
        # Normalize image to 0-255 range
        image = ((self.normalize(image) + 1) / 2) * 255
        
        # Perform operations independently for each window in the current channel
        max_pool_output = self.max_pool(image)
        min_pool_output = -self.max_pool(-image)
        
        # Compute DBC lacunarity
        nr = torch.ceil(max_pool_output / (self.r + self.eps)) - torch.ceil(min_pool_output / (self.r + self.eps)) - 1
        Mr = torch.sum(nr)
        Q_mr = nr / (self.window_size - self.r + 1)
        L_r = (Mr**2) * Q_mr / (Mr * Q_mr + self.eps)**2
        
        return L_r
    
    def compute_lacunarity(self, image: torch.Tensor) -> torch.Tensor:
        """
        Compute lacunarity for a single image.
        
        Args:
            image: Input image tensor [1, 1, H, W]
            
        Returns:
            Lacunarity tensor
        """
        with torch.no_grad():
            return self.forward(image)
    
    def get_model_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary containing model parameters
        """
        return {
            'model_name': self.model_name,
            'window_size': self.window_size,
            'eps': self.eps,
            'r': self.r,
            'num_output_channels': self.num_output_channels
        }
