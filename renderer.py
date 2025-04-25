import torch
from utils.path_tracing import path_tracing_envmap_emitter, path_tracing_dynamic_emitter
from mitsuba import load_dict
from model.emitter import EnvMapEmitter, DynamicPointEmitter
class ForwardRenderer:
    def __init__(self, cfg, material):
        self.cfg = cfg
        self.device = 'cuda'
        self.scene = load_dict({
            "type": "scene",
            "shape_id": {
                "type": "sphere",
                "center": [0, 0, 0], 
                "radius": 0.2,
                "flip_normals": False
            }
        })
        self.material = material.to(self.device)

        if cfg.renderer.emitter.type == 'envmap':
            self.ray_tracer = path_tracing_envmap_emitter
        else:
            self.ray_tracer = path_tracing_dynamic_emitter
        emitter_cfg = cfg.renderer.emitter
        if cfg.renderer.emitter.type == 'envmap':
            self.emitter = EnvMapEmitter(emitter_cfg.envmap_path).to(self.device)

        self.SPP_chunk = cfg.renderer.SPP_chunk
    
    def render(self, emitter, rays, spp):
        rays_x, rays_d, dxdu, dydv = rays[..., :3], rays[..., 3:6], rays[..., 6:9], rays[..., 9:12]
        L = torch.zeros_like(rays_x)
        if spp < self.SPP_chunk:
            self.SPP_chunk = spp
        if emitter is None:
            for _ in range(spp // self.SPP_chunk):
                L += self.ray_tracer(
                    self.scene, self.emitter, self.material,
                    rays_x, rays_d, dxdu, dydv, 
                    self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling
                )
        else:
            L = self.ray_tracer(
                self.scene, emitter, self.material,
                rays_x, rays_d, dxdu, dydv, 
                self.SPP_chunk, brdf_sampling=self.cfg.renderer.brdf_sampling, emitter_sampling=self.cfg.renderer.emitter_sampling
            )
        rgbs = L / (spp // self.SPP_chunk)
        return rgbs
