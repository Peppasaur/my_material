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

from pytorch_lightning.strategies import DDPStrategy
import importlib
import warnings
import logging
import cv2
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

def generate_video_from_results(output_folder, video_name="results_video.mp4", path_string="result_view_*.png", fps=10):
    import cv2
    import os
    from glob import glob
    img_paths = sorted(glob(os.path.join(output_folder, path_string)))
    if len(img_paths) == 0:
        print(f"No PNG images found in {output_folder}")
        return
    frame = cv2.imread(img_paths[0])
    height, width, _ = frame.shape
    out_path = os.path.join(output_folder, video_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for img_path in img_paths:
        frame = cv2.imread(img_path)
        writer.write(frame)

    writer.release()
    print(f"Saved video to {out_path}")

def gamma(x):
    mask = x <= 0.0031308
    ret = torch.empty_like(x)
    ret[mask] = 12.92 * x[mask]
    ret[~mask] = 1.055 * x[~mask].pow(1/2.4) - 0.055
    return ret


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
    
    # Create material with configured parameters
    material = PBRBRDF(albedo=torch.tensor(albedo), roughness=roughness, metallic=metallic)
    renderer = ForwardRenderer(cfg, material)
    dataset = get_dataset(cfg, 'test')
    
    """ Render ground truth """
    output_folder = os.path.join(cfg.exp_output_root_path, f'roughness_{roughness:.2f}_metallic_{metallic:.2f}')
    os.makedirs(output_folder, exist_ok=True)
    for idx, batch in tqdm(enumerate(dataset)):
        rays = batch['rays'].to(renderer.device)
        with torch.no_grad():
            img = renderer.render(None, rays, cfg.renderer.spp.test)
            img = img.reshape(*cfg.renderer.resolution, -1)

        filename = f'output_view_{idx}.exr'
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img[...,[2,1,0]].cpu().numpy())

        vis_filename = f'output_gamma_view_{idx}.png'
        vis_path = os.path.join(output_folder, vis_filename)
        torchvision.utils.save_image(gamma(img.permute(2, 0, 1)), vis_path)
                          
    
    material = MLPPBRBRDF(cfg.material, roughness)
    gt_material = PBRBRDF(albedo=torch.tensor([1.0, 1.0, 1.0]), roughness=roughness, metallic=metallic)
    model = BRDFTrainer(cfg, material, gt_material, roughness, metallic)
    
    # Load checkpoint
    if os.path.isfile(cfg.model.ckpt_path):
        print(f"=> loading model checkpoint '{cfg.model.ckpt_path}'")
        checkpoint = torch.load(cfg.model.ckpt_path, map_location=model.device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{cfg.model.ckpt_path}'. Please ensure the path is correct.")
    
    print("==> initializing data ...")          
    test_loader = DataLoader(get_dataset(cfg, 'test', output_folder), batch_size=None, num_workers=cfg.data.num_workers)

    print("==> initializing trainer ...")
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        inference_mode=True
    )

    test_results = trainer.test(model, dataloaders=test_loader)
    test_psnr = sum(result['test/psnr'] for result in test_results) / len(test_results)
    print(f"PSNR for roughness {roughness:.2f}, metallic {metallic:.2f}: {test_psnr:.2f}")

    torch.cuda.empty_cache()

    print('Training and Testing Complete!')

    # Generate video for each output folder
    generate_video_from_results(output_folder, video_name="results_video.mp4", path_string="result_view_*.png", fps=10)
    generate_video_from_results(output_folder, video_name="gt_video.mp4", path_string="output_gamma_view_*.png", fps=10)


if __name__ == "__main__":
    main()