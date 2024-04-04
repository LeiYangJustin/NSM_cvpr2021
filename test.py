
import sys
import torch
import numpy as np
import trimesh
from tqdm import trange, tqdm
from PIL import Image
import os

# model_name = "bimba_fix4_1_20240201_2005_copy_2"
model_name = "bimba"
# model_name = "bimba_hair"
# model_name = "disk"
folder = f'data/{model_name}'
data_pth = os.path.join(folder, 'original_sample.pth')
sample = torch.load(data_pth)
print(sample.keys())
for k, v in sample.items():
    if type(v) == torch.Tensor:
        print(k, v.shape)

new_sample = {}
# new_sample['faces']        = sample['faces']                          # faces
# new_sample['grid']         = sample['grid']                        # GT grid
# new_sample['points']       = sample['points']                         # GT points
# new_sample['normals']      = sample['grid_normals']

new_sample['faces']        = sample['visual_faces']                       # faces
new_sample['grid']         = sample['visual_grid']                        # GT grid
new_sample['points']       = torch.cat([sample['points'], sample['samples_3d']], dim=0)                      # GT points
new_sample['normals']      = torch.cat([sample['grid_normals'], sample['normals']], dim=0)
new_sample['boundary_idx'] = sample['boundary_idx']
new_sample['boundary']     = sample['boundary']
new_sample['C']            = sample['C']

for k, v in new_sample.items():
    if type(v) == torch.Tensor:
        print(k, v.shape)

torch.save(new_sample, os.path.join(folder, 'sample.pth'))
new_sample_loaded= torch.load(os.path.join(folder, 'sample.pth'))
print(new_sample_loaded.keys())