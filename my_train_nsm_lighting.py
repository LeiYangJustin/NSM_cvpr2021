import pymeshlab
import os
from preprocessing.convert_sample import generate_sample, generate_sample_customized
import yaml
import argparse

from omegaconf import DictConfig
from utils import compose_config_folders
from utils import copy_config_to_experiment_folder
from utils import save_meta_sample
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ProgressBar
from mains import SurfaceMap
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import trange
import sys
from datasets.mixin import DatasetMixin
import torch
import trimesh
from PIL import Image
class LitProgressBar(ProgressBar):

    def __init__(self, max_epochs: int):
        super().__init__()  # don't forget this :)
        self.enable = True
        self.bar = trange(max_epochs, desc='Training Progress', leave=True)

    def disable(self):
        self.enable = False

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        self.bar.update(1)
        self.bar.refresh()


def main(cfg: DictConfig) -> None:
    print("start training")

    compose_config_folders(cfg)
    copy_config_to_experiment_folder(cfg)

    model = SurfaceMap(cfg)
    print(cfg)
    print(cfg.checkpointing.checkpoint_path)
    
    logger = TensorBoardLogger(cfg.checkpointing.checkpoint_path, name="my_model")
    lr_logger = LearningRateMonitor(logging_interval='step')
    # progress_bar = LitProgressBar(cfg.dataset.num_epochs)

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=cfg.checkpointing.checkpoint_path,
        logger=logger, 
        callbacks=[lr_logger],
        )
    trainer.fit(model) # save surface map as sample for inter surface map in this loop

    # # save surface map as sample for inter surface map
    # save_meta_sample(cfg.checkpointing.checkpoint_path, model.dataset.sample, model.net)


def prepare_data_nsm(input_file, output_file):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_file)
    ms.apply_filter('meshing_surface_subdivision_midpoint', iterations=3)
    ms.save_current_mesh(output_file)



argparser = argparse.ArgumentParser()
argparser.add_argument('--expname', type=str, default='default')
argparser.add_argument('--model_name', type=str, required=True)



if __name__ == '__main__':
    
    args = argparser.parse_args()
    model_name = args.model_name
    
    texture_image = Image.open('asset/checkerboard.png')
    
    # model_name = "bimba_fix4_1_20240201_2005_copy_2"
    # model_name = "bimba"
    # model_name = "bimba_hair"
    # model_name = "disk"
    folder = f'data/{model_name}'
    data_pth = os.path.join(folder, 'sample.pth')
    sample = torch.load(data_pth)
    print(sample.keys())

    mesh = trimesh.Trimesh(sample['points'].numpy(), sample['faces'].numpy(), process=False, maintain_order=True)
    mesh.visual = trimesh.visual.TextureVisuals(
                    uv=sample['grid'].numpy(),
                    image=texture_image)

    # if not os.path.exists(data_pth):
    #     # input_file = 'slim.obj'
    #     # input_file = os.path.join(folder, input_file)

    #     ## this is the sampling code; if you already have the data, you can skip this part
    #     ## sampling a large model can be very slow for SLIM
    #     output_file = 'model_slim.obj'
    #     output_file = os.path.join(folder, output_file)
    #     assert os.path.exists(output_file), 'Please run SLIM to get the paramemterization!'
        
    #     # prepare_data_nsm(input_file, output_file)
    #     generate_sample_customized(output_file, data_pth)
    #     print('Data prepared successfully!')

    ## manual set config insteal of hydra
    yml_path = 'experiments/my_surface_map.yaml'
    with open(yml_path) as f:
        cfg = yaml.safe_load(f)
    cfg = DictConfig(cfg)
    cfg.dataset.sample_path = data_pth
    cfg.checkpointing.prefix += f'_{args.expname}'

    ## run training
    main(cfg)    


    """
    Documemtation:
    1) Dataset: datasets/surface_map.py

    data contains:
    2d grid points (GT cat Random)
    3d points (GT cat Random)
    normals (GT cat Random)

    2) Training: models/surface_map.py

    model: SurfaceMapModel contains 1xLinear + 9xResBlock + 1xLinear with all layers having Softplus activation function

    3) Loss: loss/ssd_loss.py (seems not good enough; try MAE later)
    """