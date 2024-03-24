import pymeshlab
import os
from preprocessing.convert_sample import generate_sample, generate_sample_customized
import yaml

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
    lr_logger = LearningRateMonitor(logging_interval='epoch')
    progress_bar = LitProgressBar(cfg.dataset.num_epochs)

    trainer = Trainer(
        max_epochs=cfg.dataset.num_epochs,
        default_root_dir=cfg.checkpointing.checkpoint_path,
        logger=logger, 
        callbacks=[lr_logger, progress_bar],
        )
    trainer.fit(model) # save surface map as sample for inter surface map in this loop

    # # save surface map as sample for inter surface map
    # save_meta_sample(cfg.checkpointing.checkpoint_path, model.dataset.sample, model.net)


def prepare_data_nsm(input_file, output_file):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_file)
    ms.apply_filter('meshing_surface_subdivision_midpoint', iterations=3)
    ms.save_current_mesh(output_file)


if __name__ == '__main__':

    folder = 'data/bimba'
    

    input_file = 'slim.obj'
    output_file = 'slim_oversampled.obj'

    input_file = os.path.join(folder, input_file)
    output_file = os.path.join(folder, output_file)
    prepare_data_nsm(input_file, output_file)

    data_pth = os.path.join(folder, 'sample.pth')
    generate_sample_customized(output_file, data_pth)
    print('Data prepared successfully!')

    ## manual set config insteal of hydra
    yml_path = 'experiments/surface_map.yaml'
    with open(yml_path) as f:
        cfg = yaml.safe_load(f)
    cfg = DictConfig(cfg)
    cfg.dataset.sample_path = data_pth

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