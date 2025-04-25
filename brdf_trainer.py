import torch
import torch.nn.functional as NF
import pytorch_lightning as pl
import pl_bolts
from renderer import ForwardRenderer
import torchvision
from torchviz import make_dot
from viztracer import VizTracer
from model.emitter import DynamicPointEmitter
import os

class BRDFTrainer(pl.LightningModule):
    def __init__(self, cfg, material, gt_material, roughness, metallic):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(cfg)
        self.roughness = roughness
        self.metallic = metallic

        self.material = material
        self.gt_material = gt_material
        self.material_latents = {}
        self.renderer = ForwardRenderer(cfg, self.material)
        self.gt_renderer = ForwardRenderer(cfg, self.gt_material)
        self.img_hw = cfg.renderer.resolution

    def gamma(self, x):
        mask = x <= 0.0031308
        ret = torch.empty_like(x)
        ret[mask] = 12.92 * x[mask]
        ret[~mask] = 1.055 * x[~mask].pow(1/2.4) - 0.055
        return ret

    def configure_optimizers(self):  
        params_to_optimize = self.parameters()
        
        if self.hparams.model.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                momentum=0.9,
                weight_decay=1e-4,
            )
            scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=int(self.hparams.model.optimizer.warmup_steps_ratio * self.hparams.model.trainer.max_steps),
                max_epochs=self.hparams.model.trainer.max_steps,
                eta_min=0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step"
                }
            }

        elif self.hparams.model.optimizer.name == 'Adam':
            optimizer = torch.optim.Adam(
                params_to_optimize,
                lr=self.hparams.model.optimizer.lr,
                betas=(0.9, 0.999),
                weight_decay=self.hparams.model.optimizer.weight_decay,
            )
            return optimizer

        else:
            logging.error('Optimizer type not supported')

    def training_step(self, batch, batch_idx):
        # randomly initialize emitter
        emitter = DynamicPointEmitter(
            dist=self.cfg.renderer.emitter.dist,
            num_lights=self.cfg.renderer.emitter.num_lights
        )
        
        # Render step logic
        rays, rgbs_gt = batch['rays'], batch['rgbs']
        rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.train)

        if rgbs_gt is None:
            with torch.no_grad():
                rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.train)
        
        loss = NF.l1_loss(rgbs, rgbs_gt)
        psnr_loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
        self.log('train/loss', loss)
        self.log('train/psnr', psnr)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # randomly initialize emitter
        emitter = DynamicPointEmitter(
            dist=self.cfg.renderer.emitter.dist,
            num_lights=self.cfg.renderer.emitter.num_lights
        )
        
        # Render step logic
        rays, rgbs_gt = batch['rays'], batch['rgbs']
        rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.val)

        if rgbs_gt is None:
            with torch.no_grad():
                rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.val)
        
        loss = NF.l1_loss(rgbs, rgbs_gt)
        psnr_loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        return

    def test_step(self, batch, batch_idx):
        rays, rgbs_gt = batch['rays'], batch['rgbs']
        
        if self.cfg.renderer.emitter.type == 'envmap':
            rgbs = self.renderer.render(None, rays, self.cfg.renderer.spp.test)
            if rgbs_gt is None:
                with torch.no_grad():
                    rgbs_gt = self.gt_renderer.render(None, rays, self.cfg.renderer.spp.test, None)
        else:
            emitter = DynamicPointEmitter(
                dist=self.cfg.renderer.emitter.dist,
                num_lights=self.cfg.renderer.emitter.num_lights
            )
            rgbs = self.renderer.render(emitter, rays, self.cfg.renderer.spp.test)
            if rgbs_gt is None:
                with torch.no_grad():
                    rgbs_gt = self.gt_renderer.render(emitter, rays, self.cfg.renderer.spp.test)
        
        psnr_loss = NF.mse_loss(self.gamma(rgbs), self.gamma(rgbs_gt))
        psnr = -10.0 * torch.log10(psnr_loss.clamp_min(1e-5))
        
        rgbs = rgbs.reshape(*self.img_hw, -1)
        rgbs_gt = rgbs_gt.reshape(*self.img_hw, -1)
        os.makedirs(os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}'), exist_ok=True)

        torchvision.utils.save_image(
            self.gamma(rgbs_gt.permute(2, 0, 1)),
            os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}', f'gt_view_{batch_idx}.png')
            )
        torchvision.utils.save_image(
            self.gamma(rgbs.permute(2, 0, 1)),
            os.path.join(self.cfg.exp_output_root_path,f'roughness_{self.roughness:.2f}_metallic_{self.metallic:.2f}', f'result_view_{batch_idx}.png')
        )
        self.log('test/psnr', psnr)
        return 
