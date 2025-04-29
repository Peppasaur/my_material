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
    
    def direction_to_uv(self, light_dir):
        """
        将光线方向向量转换为环境光贴图上的UV坐标
        Args:
            light_dir: Bx3 光线方向
        Returns:
            u: B UV坐标的u分量，范围[0,1]
            v: B UV坐标的v分量，范围[0,1]
        """
        
        # 对光线方向进行归一化，确保是单位向量
        light_dir = NF.normalize(light_dir, dim=-1)
        
        # 计算球面坐标 (theta, phi)
        # theta是y轴的极角，phi是xz平面的方位角
        theta = torch.acos(light_dir[..., 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6))  # 从y轴测量的角度
        phi = torch.atan2(light_dir[..., 0], light_dir[..., 2])  # xz平面上的角度
        
        # 映射到UV坐标 [0,1]x[0,1]
        # 对于标准经纬度环境贴图：
        u = (phi / (2 * math.pi) + 0.5) % 1.0  # 将[-pi, pi]映射到[0, 1]
        v = theta / math.pi  # 将[0, pi]映射到[0, 1]
        #print("u v", u, v)
        return u, v

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
        # 将光线方向转换为UV坐标
        u, v = self.direction_to_uv(light_dir)
        
        # 计算环境贴图上的像素坐标
        x = (u * self.W).clamp(0, self.W - 1)
        y = (v * self.H).clamp(0, self.H - 1)
        #print("x y", x, y)
        #
        # 将坐标转换为整数以查询环境贴图
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp(0, self.W - 1), (y0 + 1).clamp(0, self.H - 1)
        
        # 计算双线性插值权重
        wx = x - x0.float()
        wy = y - y0.float()
        
        # 从环境贴图中采样并执行双线性插值
        C00 = self.envmap[:, y0, x0].permute(1, 0)  # Bx3
        C01 = self.envmap[:, y0, x1].permute(1, 0)  # Bx3
        C10 = self.envmap[:, y1, x0].permute(1, 0)  # Bx3
        C11 = self.envmap[:, y1, x1].permute(1, 0)  # Bx3
        
        # 双线性插值
        wx = wx.unsqueeze(-1)
        wy = wy.unsqueeze(-1)
        C0 = C00 * (1 - wx) + C01 * wx
        C1 = C10 * (1 - wx) + C11 * wx
        Le = C0 * (1 - wy) + C1 * wy
        Le/=256
        #print("env_map", self.envmap.shape)
        #print("Le", Le)
        # 计算pdf (这里使用简单的均匀pdf)
        pdf = torch.full((light_dir.shape[0], 1), 1.0 / (4 * math.pi), device=light_dir.device)
        
        # 所有样本都有效
        valid = torch.ones_like(pdf, dtype=torch.bool)
        
        return Le, pdf, valid
    
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