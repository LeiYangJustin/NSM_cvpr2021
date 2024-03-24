import torch
import os

from models import SurfaceMapModel
from utils import show_mesh, show_textured_mesh


# DATA_PATH = "data/disk/sample.pth"
DATA_PATH = "data/bimba/bimba_surface_map.pth"
OUT_PATH = "debug"

if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

def main() -> None:

    torch.set_grad_enabled(False)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data    = torch.load(DATA_PATH)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape)

    source  = data['grid'].to(device).float()
    gt      = data['points'].to(device).float()
    faces   = data['faces'].long()
    
    # show_mesh('neural_surface_small.obj', source, out, faces, pp_loss)
    show_textured_mesh(f'{OUT_PATH}/gt_neural_surface.obj', source, gt, faces)

    # # generate mesh at sample vertices
    # source = data['visual_grid'].to(device).float()
    # gt    = torch.cat(
    #     [data['points'].to(device).float(),
    #     data['samples_3d'].to(device).float()], dim=0)
    # faces  = data['visual_faces'].long()

    # # show_mesh('neural_surface_big.obj', source, out, faces)
    # show_textured_mesh(f'{OUT_PATH}/gt_neural_surface_big.obj', source, gt, faces)


if __name__ == '__main__':
    main()
