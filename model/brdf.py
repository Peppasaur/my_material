import torch
import torch.nn as nn
import torch.nn.functional as NF
import math

import sys
sys.path.append('..')

from utils.ops import *

from nerfstudio.field_components import encodings as encoding


class PBRBRDF(nn.Module):
    """ Base BRDF class """
    def __init__(self, albedo=torch.ones(1, 3), roughness=0.2, metallic=0.5):
        super(PBRBRDF,self).__init__()
        """ In current setting we don't need to learn the albedo, roughness, metallic """
        self.albedo = nn.Parameter(albedo)
        self.roughness = nn.Parameter(torch.full((1, 1), roughness).cuda())  # Scalar roughness 
        self.metallic = nn.Parameter(torch.full((1, 1), metallic).cuda())  # Scalar metallic
        # Initialize parameters as dictionary
        self.mat = {
            'albedo': self.albedo,
            'roughness': self.roughness, 
            'metallic': self.metallic
        }
        return
    
    def forward(self, wi, wo, normal):
        brdf, pdf = self.eval_brdf(wi, wo, normal)
        return brdf, pdf

    
    def diffuse_sampler(self, sample2, normal):
        """ sampling diffuse lobe: wi ~ NoV/math.pi 
        Args:
            sample2: Bx2 unIform samples
            normal: Bx3 normal
        Return:
            wi: Bx3 sampled direction in world space
        """
        theta = torch.asin(sample2[...,0].sqrt())
        phi = math.pi*2*sample2[...,1]
        wi = angle2xyz(theta,phi)
        
        Nmat = get_normal_space(normal)
        wi = (wi[:,None]@Nmat.permute(0,2,1)).squeeze(1)    
        if wi.isnan().any():
            print("wi is nan")
        return wi


    def specular_sampler(self, sample2,roughness, wo, normal):
        """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
        Args:
            sample2: Bx3 uniform samples
            roughness: Bx1 roughness
            wo: Bx3 viewing direction
            normal: Bx3 normal
        Return:
            wi: Bx3 sampled direction in world space
        """
        alpha = (roughness * roughness).squeeze(-1)
        
        # sample half vector
        theta = (1-sample2[...,0])/((sample2[...,0]*(alpha*alpha-1)+1))
        theta = torch.acos(theta.sqrt())

        phi = 2*math.pi*sample2[...,1]
        wh = angle2xyz(theta,phi)

        # half vector to wi
        Nmat = get_normal_space(normal)
        wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
        wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
        wi = NF.normalize(wi,dim=-1)
        return wi
    

    def eval_brdf(self, wi, wo, normal):
        """
        Args:
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        # 检查几何有效性（光线和视线都在法线正半球）
        NoL = (wi*normal).sum(-1, keepdim=True)
        NoV = (wo*normal).sum(-1, keepdim=True)
        no_valid_geometry = (NoL < 0) | (NoV < 0)  # 按元素判断是否有效

        # 如果没有有效的几何配置，提前返回零值
        
        if no_valid_geometry.any():
            print("no_valid_geometry")
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)
        
        # 计算半矢量
        h = NF.normalize(wi+wo, dim=-1)
        NoL = NoL.relu()  # 在检查后可以安全地使用relu
        NoV = NoV.relu()
        VoH = (wo*h).sum(-1, keepdim=True).relu()
        NoH = (normal*h).sum(-1, keepdim=True).relu()

        '''
        NoL = NoL.clamp(1e-6, 1.0)
        NoV = NoV.clamp(1e-6, 1.0)
        VoH = VoH.clamp(1e-6, 1.0)
        NoH = NoH.clamp(1e-6, 1.0)
        '''
        '''
        print("NoL",torch.mean(NoL))
        print("NoV",torch.mean(NoV))
        print("VoH",torch.mean(VoH))
        print("NoH",torch.mean(NoH))
        '''
        # 计算漫反射系数和镜面反射系数
        # k_d = a * (1 - m)
        k_d = self.albedo * (1 - self.metallic)
        
        # k_s = 0.04 * (1 - m) + a * m
        k_s = 0.04 * (1 - self.metallic) + self.albedo * self.metallic
        
        # 计算漫反射项 (k_d/π) * (n·ω_i)
        diffuse = k_d * NoL / math.pi
        
        # 计算PBR BRDF的组成部分
        D = D_GGX(NoH, self.roughness)
        G = G_Smith(NoV, NoL, self.roughness)
        F = fresnelSchlick(VoH, k_s)
        
        
        # 计算镜面反射项 F(ω_i, h, k_s) * D(h, η, σ) * G(ω_i, ω_o, η, σ) / (4 * (n·ω_o))
        specular = F * D * G / (4 * NoV + 1e-4)
        #print("F D G", torch.mean(F), torch.mean(D), torch.mean(G))
        # 完整的BRDF：f(x, ω_i, ω_o) = (k_d/π) * (n·ω_i) + F(...) * D(...) * G(...) / (4 * (n·ω_o))
        brdf = diffuse + specular
        '''
        if (brdf > 1).any():
            print("D",torch.mean(D))
            print("G",torch.mean(G))
            print("F",torch.mean(F))
            print("diffuse",torch.mean(diffuse))
            print("specular",torch.mean(specular))
        '''
        '''
        if torch.any(diffuse < 0):
            print("警告：张量diffuse中存在小于0的元素！")
        if torch.any(specular < 0):
            print("警告：张量specular中存在小于0的元素！")
        '''
        # 计算PDF（重要性采样的PDF）
        pdf_spec = D/((4*VoH.clamp_min(1e-4))*NoH.clamp_min(1e-4))
        pdf_diff = NoL / math.pi
        pdf = 0.5 * pdf_spec + 0.5 * pdf_diff
        #print("albedo roughness metallic", self.albedo, self.roughness, self.metallic)
        #print("pbrbrdf",torch.mean(brdf))
        #brdf=brdf.clamp(0,1)
        return brdf, pdf
    
    def sample_brdf(self,sample1,sample2,wo,normal):
        """ importance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)
        brdf = torch.zeros(B,3,device=device)


        mask = (sample1 > 0.5)
        wi_diffuse = self.diffuse_sampler(sample2[mask], normal[mask])
        wi_specular = self.specular_sampler(sample2[~mask], self.mat['roughness'].expand(normal.shape[0], 1)[~mask], wo[~mask], normal[~mask])

        # Construct wi without gradient-breaking assignment
        wi = torch.zeros(B, 3, device=device)
        wi = wi.clone()  # explicitly ensures gradients
        wi[mask] = wi_diffuse
        wi[~mask] = wi_specular
        # get brdf,pdf
        brdf,pdf = self.forward(wi,wo,normal)
        brdf_weight = torch.where(pdf>0,brdf/pdf,0)
        brdf_weight[brdf_weight.isnan()] = 0
        return wi,pdf,brdf_weight

class ProxyPBRBRDF(nn.Module):
    def __init__(self, roughness=0.1):
        super(ProxyPBRBRDF, self).__init__()
        self.roughness = nn.Parameter(torch.full((1, 1), roughness).cuda())  # Scalar roughness 
        self.metallic = 0.2
        self.albedo = torch.ones(1, 3).cuda()
    
    def eval_pdf(self,wi,wo,normal):
        """ evaluate BRDF and pdf
            wi: Bx3 light direction
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: surface BRDF dict
        Return:
            brdf: Bx3
            pdf: Bx1
        """
        # Check if both directions are on the same side
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)
        valid_geometry = (NoL > 0) & (NoV > 0)
        
        # Early return for invalid geometry
        if not valid_geometry.any():
            return torch.zeros_like(wi), torch.zeros(wi.shape[0], 1, device=wi.device)

        h = NF.normalize(wi+wo,dim=-1)
        NoL = NoL.relu()  # Now safe to relu after check
        NoV = NoV.relu()
        VoH = (wo*h).sum(-1,keepdim=True).relu()
        NoH = (normal*h).sum(-1,keepdim=True).relu()

        # get pdf
        D = D_GGX(NoH, self.roughness.expand(normal.shape[0], 1))
        pdf_spec = D/((4*VoH.clamp_min(1e-4))*NoH.clamp_min(1e-4))
        pdf_diff = NoL/math.pi
        pdf = 0.5*pdf_spec + 0.5*pdf_diff

        return pdf

    def diffuse_sampler(self, sample2, normal):
        """ sampling diffuse lobe: wi ~ NoV/math.pi 
        Args:
            sample2: Bx2 unIform samples
            normal: Bx3 normal
        Return:
            wi: Bx3 sampled direction in world space
        """
        theta = torch.asin(sample2[...,0].sqrt())
        phi = math.pi*2*sample2[...,1]
        wi = angle2xyz(theta,phi)
        
        Nmat = get_normal_space(normal)
        wi = (wi[:,None]@Nmat.permute(0,2,1)).squeeze(1)    
        if wi.isnan().any():
            print("wi is nan")
        return wi

    def specular_sampler(self, sample2,roughness, wo, normal):
        """ sampling ggx lobe: h ~ D/(VoH*4)*NoH
        Args:
            sample2: Bx3 uniform samples
            roughness: Bx1 roughness
            wo: Bx3 viewing direction
            normal: Bx3 normal
        Return:
            wi: Bx3 sampled direction in world space
        """
        alpha = (roughness * roughness).squeeze(-1)
        theta = (1-sample2[...,0])/((sample2[...,0]*(alpha*alpha-1)+1))
        theta = torch.acos(theta.sqrt())

        phi = 2*math.pi*sample2[...,1]
        wh = angle2xyz(theta,phi)

        # half vector to wi
        Nmat = get_normal_space(normal)
        wh = (wh[:,None]@Nmat.permute(0,2,1)).squeeze(1)
        wi = 2*(wo*wh).sum(-1,keepdim=True)*wh-wo
        wi = NF.normalize(wi,dim=-1)
        return wi
    
    def sample_brdf(self,sample1,sample2,wo,normal):
        """ importance sampling brdf and get brdf/pdf
        Args:
            sample1: B unifrom samples
            sample2: Bx2 uniform samples
            wo: Bx3 viewing direction
            normal: Bx3 normal
            mat: material dict
        Return:
            wi: Bx3 sampled direction
            pdf: Bx1
            brdf_weight: Bx3 brdf/pdf
        """
        B = sample1.shape[0]
        device = sample1.device

        pdf = torch.zeros(B,device=device)

        mask = (sample1 > 0.5)
        wi_diffuse = self.diffuse_sampler(sample2[mask], normal[mask])
        wi_specular = self.specular_sampler(sample2[~mask], self.roughness.expand(normal.shape[0], 1)[~mask], wo[~mask], normal[~mask])

        # Construct wi without gradient-breaking assignment
        wi = torch.zeros(B, 3, device=device)
        wi = wi.clone()  # explicitly ensures gradients
        wi[mask] = wi_diffuse
        wi[~mask] = wi_specular
        wi = wi.detach()

        pdf = self.eval_pdf(wi,wo,normal)
        return wi, pdf

class MLPPBRBRDF(nn.Module):
    """ MLP-based BRDF class """
    def __init__(self, cfg, gt_roughness):
        super(MLPPBRBRDF, self).__init__()

        self.proxy_brdf = ProxyPBRBRDF(roughness=gt_roughness)
        self.Linear = nn.Linear(9, 1)
        
        self.encoding_wi = encoding.SHEncoding(levels=4)  # 使用4阶球谐函数
        self.encoding_wo = encoding.SHEncoding(levels=4)
        
        # 计算编码后的输入维度
        wi_enc_dim = self.encoding_wi.get_out_dim()
        wo_enc_dim = self.encoding_wo.get_out_dim()
        
        class ExpActivation(nn.Module):
            def forward(self, x):
                return torch.exp(x)
        input_dim = wi_enc_dim + wo_enc_dim 
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            ExpActivation()
            #nn.ReLU()
            #nn.Sigmoid()  # 输出层仍使用Sigmoid保证值在[0,1]范围
        )


    def forward(self, wi, wo, normal):
        """
        Evaluate BRDF using MLP
        Args:
            wi: Bx3 incoming light direction 
            wo: Bx3 outgoing view direction
        Returns:
            brdf: Bx1 BRDF values
        """
        # 将世界坐标转换为局部坐标
        wi_local, wo_local, normal_local = self.world_to_local(wi, wo, normal)
        
        #test
        #wi_local=wi
        #wo_local=wo
        #normal_local=normal
        #test_end
        # 对局部坐标中的方向向量进行编码
        wi_encoded = self.encoding_wi(wi_local)
        wo_encoded = self.encoding_wo(wo_local)
        
        # 拼接编码后的向量和法线
        x = torch.cat([wi_encoded, wo_encoded], dim=-1)
        
        # 通过MLP预测BRDF值
        return self.mlp(x)

    def world_to_local(self, wi, wo, normal):
        """
        将世界坐标系中的向量转换到以normal为z轴的局部坐标系
        Args:
            wi: Bx3 light direction in world space
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
        Returns:
            wi_local, wo_local, normal_local: 局部坐标系中的向量
        """
        # 获取从世界坐标到法线坐标空间的变换矩阵
        Nmat = get_normal_space(normal)  # Bx3x3
        
        # 将向量从世界坐标转换到局部坐标
        wi_local = torch.bmm(wi.unsqueeze(1), Nmat).squeeze(1)    # Bx1x3 @ Bx3x3 -> Bx1x3 -> Bx3
        wo_local = torch.bmm(wo.unsqueeze(1), Nmat).squeeze(1)
        
        # 在局部坐标系中，法线应该是(0,0,1)
        # 但为了保持一致性，我们使用实际计算得到的值
        normal_local = torch.bmm(normal.unsqueeze(1), Nmat).squeeze(1)
        
        return wi_local, wo_local, normal_local
    
    def eval_brdf(self, wi, wo, normal):
        """
        Evaluate BRDF and pdf after transforming world-space vectors to local space.
        Args:
            wi: Bx3 light direction in world space
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
        Returns:
            brdf: Bx3 BRDF values
            pdf: Bx1 probability
        """
        # Ensure normal is normalized
        NoL = (wi*normal).sum(-1,keepdim=True)
        NoV = (wo*normal).sum(-1,keepdim=True)

        brdf_value = self.forward(wi, wo, normal)
        brdf = brdf_value.expand(-1, 3)

        pdf = NoL / math.pi

        return brdf, pdf
        
    def sample_brdf(self, sample1, sample2, wo, normal):
        """
        Importance sampling BRDF using proxy (PBRBRDF) and evaluating MLP BRDF.
        
        Args:
            sample1: B uniform samples [0,1] to select diffuse or specular sampling
            sample2: Bx2 uniform samples for hemisphere sampling
            wo: Bx3 viewing direction in world space
            normal: Bx3 normal in world space
            proxy_brdf: instance of PBRBRDF to guide sampling (importance sampling proxy)

        Returns:
            wi: Bx3 sampled incoming directions (world space)
            pdf: Bx1 sampling pdf values from proxy_brdf
            brdf_weight: Bx3 ratio (MLP evaluated BRDF / pdf)
        """

        wi_proxy, pdf_proxy = self.proxy_brdf.sample_brdf(sample1, sample2, wo, normal)
        stop_gradient_pdf_proxy = pdf_proxy.detach()
        mlp_brdf, _ = self.eval_brdf(wi_proxy, wo, normal)
        mlp_brdf = mlp_brdf * pdf_proxy / (stop_gradient_pdf_proxy + 1e-8)
        brdf_weight = torch.where(pdf_proxy > 0, mlp_brdf / (stop_gradient_pdf_proxy + 1e-8), torch.zeros_like(mlp_brdf))

        return wi_proxy, stop_gradient_pdf_proxy, brdf_weight
    
