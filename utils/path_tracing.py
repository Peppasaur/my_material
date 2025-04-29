import torch
import torch.nn.functional as NF
from model.brdf import MLPPBRBRDF

import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

from .ops import *

def ray_intersect(scene,xs,ds):
    """ warpper of mitsuba ray-mesh intersection 
    Args:
        xs: Bx3 pytorch ray origin
        ds: Bx3 pytorch ray direction
    Return:
        positions: Bx3 intersection location
        normals: Bx3 normals
        uvs: Bx2 uv coordinates
        idx: B triangle indices, -1 indicates no intersection
        valid: B whether a valid intersection
    """
    # convert pytorch tensor to mitsuba
    xs_mi = mitsuba.Point3f(xs[...,0],xs[...,1],xs[...,2])
    ds_mi = mitsuba.Vector3f(ds[...,0],ds[...,1],ds[...,2])
    rays_mi = mitsuba.Ray3f(xs_mi,ds_mi)
    
    ret = scene.ray_intersect_preliminary(rays_mi)
    idx = mitsuba.Int(ret.prim_index).torch().long()
    ret = ret.compute_surface_interaction(rays_mi)
    
    positions = ret.p.torch()
    normals = ret.n.torch()
    normals = NF.normalize(normals,dim=-1)
    
    # check if invalid intersection
    ts  = ret.t.torch()
    valid = (~ts.isinf())
    
    idx[~valid] = -1
    normals = double_sided(-ds,normals)
    return positions,normals,ret.uv.torch(),idx,valid

def path_tracing_dynamic_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
        indir_depth: indirect illumination depth
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    N_lights = emitter_net.light_positions.shape[0]
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    # position, normal, vis = ray_sphere_intersect(scene,position,wi)
    L = torch.zeros(vis.shape[0],3,device=device)
    if not vis.any():
        return L.reshape(B,spp,3).mean(1)
    position = position[vis]
    normal = normal[vis]
    wo = -wi[vis]
    
    # deterministic sampling
    wi,emit_pdf, emit_position, idx = emitter_net.sample_emitter(position)
    normal = normal.repeat_interleave(emitter_net.light_positions.shape[0],0)
    position = position.repeat_interleave(emitter_net.light_positions.shape[0],0)
    wo = wo.repeat_interleave(emitter_net.light_positions.shape[0],0)
    
    # visibility test
    emit_weight,_,_ = emitter_net.eval_emitter(emit_position, idx)
    emit_vis = (wi*normal).sum(-1,keepdim=True) > 0 # B, 1
    
    # goemetry term (assume double sided area light)
    G = 1 / (emit_position-position).pow(2).sum(-1).clamp_min(1e-6) # B, 1
    emit_weight = emit_weight*emit_vis*G[...,None]/emit_pdf.clamp_min(1e-6)
    
    # Now, reshape and average over light dimension
    emit_brdf,_ = material_net.eval_brdf(wi,wo,normal)
    L[vis] += (emit_brdf*emit_weight).reshape(-1, N_lights,3).mean(1)
    L = L.reshape(B,spp,3).mean(1)
    return L


def path_tracing_envmap_emitter(scene,emitter_net,material_net,rays_o,rays_d,dx_du,dy_dv,spp, brdf_sampling, emitter_sampling):
    """ Path trace current scene
    Args:
        scene: mitsuba scene
        emitter_net: emitter object
        material_net: material object
        rays_o: Bx3 ray origin
        rays_d: Bx3 ray direction
        dx_du,dy_dv: Bx3 ray differential
        spp: sampler per pixel
        indir_depth: indirect illumination depth
    Return:
        L: Bx3 traced results
    """
    B = len(rays_o)
    device = rays_o.device
    
    # sample camera ray
    du,dv = torch.rand(2,len(rays_o),spp,1,device=device)-0.5
    wi = NF.normalize(rays_d[:,None]+dx_du[:,None]*du+dy_dv[:,None]*dv,dim=-1).reshape(-1,3)
    
    # Add mask for wi z component
    position = rays_o.repeat_interleave(spp,0)
    
    # compute first intersection
    position,normal,_, _,vis = ray_intersect(scene,position,wi)
    L,_,valid_next = emitter_net.eval_emitter(position,wi)
    L[vis] = 0 # set the radiance to 0 for the valid intersection
    valid_next = vis
    # drop invalid intersection
    if not valid_next.any():
        return L.reshape(B,spp,3).mean(1)
    position = position[valid_next]
    normal = normal[valid_next]
    wo = -wi[valid_next]
    active_next = valid_next.clone()

    # Sample the environment map instead of a point emitter
    if emitter_sampling:
        wi, emit_pdf, _ = emitter_net.sample_emitter(torch.rand_like(position[..., :2]), position)

        # Evaluate the environment map along sampled directions
        emit_weight, emit_pdf, _ = emitter_net.eval_emitter(position, wi)
        emit_weight = emit_weight / emit_pdf.clamp_min(1e-6)
        # emit brdf
        emit_brdf,brdf_pdf = material_net.eval_brdf(wi,wo,normal)
        w_mis = torch.where((emit_pdf>0)&(~brdf_pdf.isinf()),emit_pdf*emit_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[emit_pdf.isinf()|(brdf_pdf==0)] = 1
        L[active_next] += emit_brdf*emit_weight * w_mis
    

    # sample brdf
    if brdf_sampling:
        wi,brdf_pdf,brdf_weight = material_net.sample_brdf(
            torch.rand(len(normal),device=device),
            torch.rand(len(normal),2,device=device),
            wo,normal)
    
        # Evaluate Le from environment map
        Le, emit_pdf, valid_next = emitter_net.eval_emitter(position, wi)

        # Update BRDF PDF
        brdf_pdf = brdf_pdf 
        
        w_mis = torch.where((brdf_pdf>0)&(~emit_pdf.isinf()),brdf_pdf*brdf_pdf/(emit_pdf*emit_pdf+brdf_pdf*brdf_pdf),0)
        w_mis[brdf_pdf.isinf()|(emit_pdf==0)] = 1
        w_mis[w_mis.isnan()] = 0
        L[active_next] += brdf_weight*Le * w_mis
    
    #print("L",torch.mean(L))
    has_invalid = torch.isnan(L).any() or torch.isinf(L).any()
    if has_invalid:
        print("Contains NaN or inf:", has_invalid)
    L = L.reshape(B,spp,3).mean(1)
    return L

