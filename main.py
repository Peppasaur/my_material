import torch
import torchvision
import json
import os
from tqdm import tqdm
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from renderer import ForwardRenderer
from brdf_trainer import BRDFTrainer
from model.brdf import MLPPBRBRDF, PBRBRDF, ProxyPBRBRDF
from torch.utils.data import DataLoader
from utils.dataset import SphereDataset
import hydra
from omegaconf import DictConfig
from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
from viztracer import VizTracer
import cv2
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)



def get_dataset(cfg, split, gt_path=None):
    return SphereDataset(
        cfg,
        gt_path,
        split
    )

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.model.checkpoint_monitor)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    return [checkpoint_monitor, lr_monitor]

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # fix the seed
    pl.seed_everything(cfg.global_train_seed, workers=True)
    os.makedirs(cfg.exp_output_root_path, exist_ok=True)
    checkpoint_output_path = os.path.join(cfg.exp_output_root_path, "training")
    os.makedirs(checkpoint_output_path, exist_ok=True)

    # Load ground truth material parameters from pbr config
    gt_material_cfg = hydra.compose(config_name="config", overrides=["material=pbr"]).material
    
    # Use ground truth parameters from pbr.yaml
    albedo = gt_material_cfg.albedo
    roughness = gt_material_cfg.roughness
    metallic = gt_material_cfg.metallic
    
    output_folder = os.path.join(cfg.exp_output_root_path, f'roughness_{roughness:.2f}_metallic_{metallic:.2f}')
    os.makedirs(output_folder, exist_ok=True)

    # Initialize materials using different configs
    material = MLPPBRBRDF(cfg.material, roughness)  # MLP model uses mlp_pbr config
    gt_material = PBRBRDF(
        albedo=torch.tensor(albedo), 
        roughness=roughness, 
        metallic=metallic
    )  # Ground truth uses pbr config

    model = BRDFTrainer(cfg, material, gt_material, roughness, metallic)

    print("==> initializing data ...")          
    train_loader = DataLoader(get_dataset(cfg, 'train', None), batch_size=None, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(get_dataset(cfg, 'val', None), batch_size=None, num_workers=cfg.data.num_workers)
    test_loader = DataLoader(get_dataset(cfg, 'test', None), batch_size=None, num_workers=cfg.data.num_workers)

    print("==> initializing logger ...")
    logger = hydra.utils.instantiate(cfg.model.logger, save_dir=cfg.exp_output_root_path)

    print("==> initializing monitor ...")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.model.checkpoint_monitor.dirpath, f'model_{roughness:.2f}_{metallic:.2f}'),
        filename=cfg.model.checkpoint_monitor.filename,
        save_top_k=cfg.model.checkpoint_monitor.save_top_k, 
        every_n_epochs=cfg.model.checkpoint_monitor.every_n_epochs,
        monitor='val/loss',
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    print("==> initializing trainer ...")
    
    ckpt_path = None
    if hasattr(cfg.model, 'resume_training') and cfg.model.resume_training:
        if hasattr(cfg.model, 'ckpt_path') and os.path.exists(cfg.model.ckpt_path):
            ckpt_path = cfg.model.ckpt_path
            print(f"==> 从检查点恢复训练: {ckpt_path}")
        else:
            # 尝试查找最新的last.ckpt
            model_dir = os.path.join(cfg.model.checkpoint_monitor.dirpath, f'model_{roughness:.2f}_{metallic:.2f}')
            last_ckpt = os.path.join(model_dir, 'last.ckpt')
            if os.path.exists(last_ckpt):
                ckpt_path = last_ckpt
                print(f"==> 从最新检查点恢复训练: {ckpt_path}")
            else:
                print("==> 未找到有效检查点，将从头开始训练")

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor], logger=logger, **cfg.model.trainer, strategy=DDPStrategy(find_unused_parameters=True)
    )

    trainer.fit(model, train_loader, val_loader)
    test_results = trainer.test(model, dataloaders=test_loader)

    test_psnr = sum(result['test/psnr'] for result in test_results) / len(test_results)
    print(f"PSNR for roughness {roughness:.2f}, metallic {metallic:.2f}: {test_psnr:.2f}")

    torch.cuda.empty_cache()

    print('Training and Testing Complete!')


if __name__ == "__main__":
    main()