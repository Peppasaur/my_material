import torch
import torch.nn as nn
import torch.nn.functional as NF
import numpy as np
import math
import imageio
import cv2
from openexr_numpy import imread, imwrite

class EnvMapEmitter(nn.Module):
    """ Environment map emitter using HDRI """
    def __init__(self, envmap_path):
        """
        Args:
            envmap_path: Path to the .exr or .hdr environment map
        """
        super(EnvMapEmitter, self).__init__()
        
        # Load environment map (assumed to be in lat-long format)
        envmap = imageio.imread(envmap_path).astype('float32')[:,:,:3]  # Shape: (H, W, 3)
        envmap = torch.from_numpy(envmap).permute(2, 0, 1)  # Convert to (3, H, W)
        
        self.register_buffer('envmap', envmap)
        self.H, self.W = envmap.shape[1:]  # Get resolution

    def sample_emitter(self, sample, position):
        """
        Sample a direction from the environment map
        Args:
            sample: Bx2 uniform samples for spherical sampling
            position: Bx3 surface positions (unused)
        Returns:
            wi: Bx3 sampled directions
            pdf: Bx1 sampling pdf
            idx: B dummy indices (-1)
        """
        # Convert uniform samples to spherical coordinates
        phi = 2 * math.pi * sample[..., 0]  # Azimuth
        theta = torch.acos(1 - 2 * sample[..., 1])  # Elevation
        
        sin_theta = torch.sin(theta)
        x = sin_theta * torch.cos(phi)
        y = sin_theta * torch.sin(phi)
        z = torch.cos(theta)
        
        wi = torch.stack([x, y, z], dim=-1)  # Direction vectors
        
        # Compute PDF (uniform for now)
        pdf = torch.full((position.shape[0], 1), 1.0 / (4 * math.pi), device=position.device)
        
        idx = torch.full((position.shape[0],), -1, dtype=torch.long, device=position.device)
        
        return wi, pdf, idx

    def eval_emitter(self, position, light_dir):
        """
        Evaluate environment map radiance along given directions
        Args:
            position: Bx3 intersection points (unused)
            light_dir: Bx3 light directions
        Returns:
            Le: Bx3 radiance
            pdf: Bx1 pdf
            valid: B valid samples (always True for envmap)
        """
        # TODO: return the radiance of the environment map from the light_dir
        Le = torch.zeros_like(light_dir)
        pdf = torch.ones_like(light_dir)[:,0]
        return Le, pdf, torch.ones_like(pdf, dtype=torch.bool)  # Always valid
    
class DynamicPointEmitter(nn.Module):
    def __init__(self, dist=4.0, num_lights=8):
        """
        Args:
            dist: Radius of the sphere
            num_lights: Number of lights to sample
        """
        super(DynamicPointEmitter, self).__init__()

        self.dist = dist
        self.num_lights = num_lights

        # Sample spherical coordinates on GPU
        theta = torch.arccos(1 - 2 * torch.rand(num_lights, device='cuda'))  # theta ∈ [0, pi]
        phi = 2 * torch.pi * torch.rand(num_lights, device='cuda')          # phi ∈ [0, 2pi]

        x = dist * torch.sin(theta) * torch.cos(phi)
        y = dist * torch.sin(theta) * torch.sin(phi)
        z = dist * torch.cos(theta)

        positions = torch.stack([x, y, z], dim=1)  # [N, 3]
        intensities = torch.rand(num_lights, 1, device='cuda') * 49.0 + 1.0  # Uniform [1.0, 50.0]

        self.register_buffer('light_positions', positions)  # [N, 3]
        self.register_buffer('light_intensities', intensities)  # [N, 1]

    def sample_emitter(self, position):
        """
        Deterministic sampling: For each position, sample toward every light.

        Args:
            sample1: (unused)
            sample2: (unused)
            position: (B, 3) surface positions

        Returns:
            wi: (B, N, 3) directions toward lights
            pdf: (B, N, 1) uniform pdf
            light_pos: (B, N, 3) selected light positions
            idx: (B, N) selected light indices
        """
        B = position.shape[0]
        N = self.light_positions.shape[0]

        position_expand = position.unsqueeze(1).expand(B, N, 3)  # [B, N, 3]
        light_pos_expand = self.light_positions.unsqueeze(0).expand(B, N, 3)  # [B, N, 3]

        vec = light_pos_expand - position_expand  # [B, N, 3]
        wi = NF.normalize(vec, dim=-1).reshape(-1, 3)  # [B*N, 3]

        pdf = torch.full((B*N, 1), 1.0 / N, device=position.device)

        idx = torch.arange(N, device=position.device).unsqueeze(0).expand(B, N).reshape(-1)  # [B*N]

        light_pos = light_pos_expand.reshape(-1, 3)

        return wi, pdf, light_pos, idx

    def eval_emitter(self, position, idx):
        """
        Evaluate radiance from selected lights.

        Args:
            position: (B, 3) surface points
            idx: (B,) selected light indices

        Returns:
            Le: (B, 3) radiance
            pdf: (B, 1) pdf
            valid: (B,) valid mask
        """
        B = position.shape[0]

        intensities = self.light_intensities[idx].expand(-1, 3)  # [B, 3]
        pdf = torch.full((B, 1), 1.0 / self.light_positions.shape[0], device=position.device)
        valid = torch.ones(B, dtype=torch.bool, device=position.device)

        return intensities, pdf, valid