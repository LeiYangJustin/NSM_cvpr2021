
import torch

from pytorch_lightning import LightningModule
from torch.optim import RMSprop, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from datasets import SurfaceMapDataset, SurfaceMapDatasetCustomized
from loss import SSDLoss, MAELoss
from models import SurfaceMapModel
from utils import DifferentialMixin
from utils import save_meta_sample


class SurfaceMap(DifferentialMixin, LightningModule):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.net = SurfaceMapModel() # map

        if self.config.loss.type == 'ssd':
            self.loss_function = SSDLoss()
        elif self.config.loss.type == 'mae':
            self.loss_function = MAELoss()
        else:
            raise NotImplementedError


    def train_dataloader(self):
        # self.dataset = SurfaceMapDataset(self.config.dataset)
        self.dataset = SurfaceMapDatasetCustomized(self.config.dataset)
        dataloader   = DataLoader(self.dataset, batch_size=None, shuffle=True,
                                num_workers=self.config.dataset.num_workers)

        return dataloader


    # def configure_optimizers(self):
    #     LR        = 1.0e-4
    #     optimizer = RMSprop(self.net.parameters(), lr=LR, momentum=0.9)
    #     restart   = int(self.config.dataset.num_epochs)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=restart)
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        LR        = 1.0e-4
        optimizer = RMSprop(self.net.parameters(), lr=LR, momentum=0.9)
        # optimizer = Adam(self.net.parameters(), lr=LR, amsgrad=False)
        restart   = int(self.config.dataset.num_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingLR(optimizer, T_max=restart),
                'interval': 'step',
                'frequency': 1
            },
        }



    def training_step(self, batch, batch_idx):

        source  = batch['source']  # Nx2
        gt      = batch['gt']      # Nx3
        normals = batch['normals'] # Nx3

        # activate gradient so can compute the normals through differentiation
        source.requires_grad_(True)

        # forward network
        out          = self.net(source)
        # estimate normals through autodiff
        pred_normals = self.compute_normals(out=out, wrt=source)

        # loss
        loss_dist    = self.loss_function(out, gt)
        loss_normals = self.loss_function(pred_normals, normals)

        loss = 0.0
        loss += loss_dist
        loss += 0.01 * loss_normals

        # add here logging if needed
        self.log('train_loss', loss)
        self.log('surf', loss_dist)
        self.log('normal', loss_normals)
        

        if batch_idx % self.config.eval_step == 0:
            print(f"Batch: {batch_idx} - Loss: {loss}")
            save_meta_sample(self.config.checkpointing.checkpoint_path, self.dataset.sample, self.net, epoch=batch_idx)

        return loss
