
import torch
import os
import shutil

from models import SurfaceMapModel
from utils import show_mesh, show_textured_mesh
from torch.nn import functional as F

import argparse

# CHECKPOINT_PATH = '/SET/HERE/YOUR/PATH/TO/PTH'
# CHECKPOINT_PATH = "outputs/neural_map/overfit_1711263627.271986/sample/flat_model.pth" ## 2000 epochs 10000 samples
# CHECKPOINT_PATH = "outputs/neural_map/overfit_1711262297.812526/sample/flat_model.pth" ## 10000 epochs 4000 samples
# CHECKPOINT_PATH = "outputs/neural_map/overfit_1711264724.864906/sample/flat_model.pth" ## 10000 epochs 10000 samples

argparser = argparse.ArgumentParser()

argparser.add_argument('--checkpoint', type=str, required=True)

def main() -> None:

    torch.set_grad_enabled(False)

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    net = SurfaceMapModel()
    net = net.to(device)

    args = argparser.parse_args()
    CHECKPOINT_PATH = args.checkpoint

    save_dir = CHECKPOINT_PATH.split('/')
    epoch = save_dir[-1].split('_')[-1].split('.')[0]

    save_dir = '/'.join(save_dir[:-2])
    save_dir = os.path.join(save_dir,'results/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    data    = torch.load(CHECKPOINT_PATH)
    source  = data['grid'].to(device).float()
    gt      = data['points'].to(device).float()
    faces   = data['faces'].long()
    weights = data['weights']

    for k in weights.keys():
        weights[k] = weights[k].to(device).detach()

    # generate mesh at GT vertices
    out     = net(source, weights)
    # pp_loss = (out - gt).pow(2).sum(-1)

    pp_loss = F.l1_loss(out, gt)
    print("pp_loss", pp_loss.item())

    # show_mesh('neural_surface_small.obj', source, out, faces, pp_loss)
    show_textured_mesh(f'{save_dir}/neural_surface_small_{epoch}.obj', source, out, faces, pp_loss)
    show_textured_mesh(f'{save_dir}/neural_surface_small_gt.obj', source, gt, faces)


    # # generate mesh at sample vertices
    # source = data['visual_grid'].to(device).float()
    # faces  = data['visual_faces'].long()

    # out = net(source, weights)

    # # show_mesh('neural_surface_big.obj', source, out, faces)
    # show_textured_mesh(f'{save_dir}/neural_surface_big_{epoch}.obj', source, out, faces)


if __name__ == '__main__':
    main()
