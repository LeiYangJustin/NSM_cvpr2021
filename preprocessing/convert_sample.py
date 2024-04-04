
import sys
import torch
import numpy as np
import trimesh
from tqdm import trange, tqdm
from PIL import Image

from .read_obj import *
texture_image = Image.open('asset/checkerboard.png')

def generate_sample(obj_file_small, obj_file_large, output_file):

    ## reading data
    points_small, uv_grid_small, faces_small, normals_small = read_mesh_from_obj(obj_file_small)
    print('Read sample small')
    points_large, uv_grid_large, faces_large, normals_large = read_mesh_from_obj(obj_file_large)
    print('Read sample large')

    ## checking normals
    has_normals = normals_small.shape[0] > 0 and normals_large.shape[0] > 0
    if not has_normals:
        print('No normals found! Computing them with trimesh')
        mesh          = trimesh.Trimesh(points_small, faces_small)
        normals_small = mesh.vertex_normals
        mesh          = trimesh.Trimesh(points_large, faces_large)
        normals_large = mesh.vertex_normals

    ## remove points from points_large by points_small
    # idxs = []
    # for pt in tqdm(uv_grid_small):
    #     idxs.append(np.absolute(uv_grid_large - pt).sum(axis=1).argmin().item())

    # filtered_grid = []
    # filtered_points = []
    # filtered_normals = []
    # for i in trange(uv_grid_large.shape[0]):
    #     if i in idxs:
    #         continue
    #     filtered_grid.append(uv_grid_large[i].tolist())
    #     filtered_points.append(points_large[i].tolist())
    #     filtered_normals.append(normals_large[i].tolist())

    # filtered_grid    = torch.tensor(filtered_grid)
    # filtered_points  = torch.tensor(filtered_points)
    # filtered_normals = torch.tensor(filtered_normals)

    ## if data is generated with meshlab than order is preserved (can comment block above)
    filtered_grid    = torch.from_numpy(uv_grid_large[uv_grid_small.shape[0]:])
    filtered_points  = torch.from_numpy(points_large[points_small.shape[0]:])
    filtered_normals = torch.from_numpy(normals_large[normals_small.shape[0]:])

    ## convert everything to torch
    grid         = torch.from_numpy(uv_grid_small)
    points       = torch.from_numpy(points_small)
    faces        = torch.tensor(faces_small).long()
    visual_grid  = torch.from_numpy(uv_grid_large)
    visual_faces = torch.tensor(faces_large)

    ## extract boundary
    mesh = trimesh.Trimesh(points, faces_small, process=False)

    boundary              = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1) # edges which appears only once
    vertices_index        = mesh.edges[boundary]
    unique_vertices_index = np.unique(vertices_index.reshape(-1))
    boundary_grid         = grid[unique_vertices_index]

    ## compute normalization constant
    # normalization to make surface area = unit circle area
    C = np.sqrt( 1.0 / mesh.area )


    sample = {}
    sample['faces']        = torch.tensor(faces_small)      # faces
    sample['grid']         = grid                           # GT grid
    sample['grid_normals'] = torch.tensor(normals_small).float()
    sample['points']       = points                         # GT points
    sample['normals']      = filtered_normals.float()
    sample['samples_2d']   = filtered_grid                  # random on grid
    sample['samples_3d']   = filtered_points                # random on surface
    sample['visual_faces'] = torch.tensor(faces_large).long()
    sample['visual_grid']  = visual_grid                    # grid for visualization
    sample['boundary_idx'] = torch.from_numpy(unique_vertices_index).long()
    sample['boundary']     = boundary_grid
    sample['C']            = C * np.sqrt(np.pi)

    torch.save(sample, output_file)
    print('Done')


def generate_sample_customized(obj_file_large, output_file):
    ## reading data
    points, uv_grid, faces, normals = read_mesh_from_obj(obj_file_large)
    print('Read sample large')

    mesh    = trimesh.Trimesh(points, faces, process=False, maintain_order=True)
    mesh.visual = trimesh.visual.TextureVisuals(
                    uv=uv_grid,
                    image=texture_image)
    normals = mesh.vertex_normals
    mesh.export('gt.obj')

    uv_grid  = torch.from_numpy(uv_grid)
    faces = torch.tensor(mesh.faces).long()
    normals = torch.tensor(normals).float()
    points = torch.from_numpy(mesh.vertices).float()

    boundary              = trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1) # edges which appears only once
    vertices_index        = mesh.edges[boundary]
    unique_vertices_index = np.unique(vertices_index.reshape(-1))
    boundary_grid         = uv_grid[unique_vertices_index]

    ## compute normalization constant
    # normalization to make surface area = unit circle area
    C = np.sqrt( 1.0 / mesh.area )


    sample = {}
    sample['faces']        = faces                          # faces
    sample['grid']         = uv_grid                        # GT grid
    sample['points']       = points                         # GT points
    sample['normals']      = normals
    sample['boundary_idx'] = torch.from_numpy(unique_vertices_index).long()
    sample['boundary']     = boundary_grid
    sample['C']            = C * np.sqrt(np.pi)

    torch.save(sample, output_file)
    print('Done')

if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print('Usage: python convert_sample.py small.obj large.obj out.pth')
    #     sys.exit(1)
        
    # small_file = sys.argv[1]
    # large_file = sys.argv[2]
    # out_file   = sys.argv[3]

    # generate_sample(small_file, large_file, out_file)

    generate_sample_customized('data/bimba_hair/mesh_slim.obj', 'data/bimba_hair/sample.pth')

    sample = torch.load('data/bimba_hair/sample.pth')
    print(sample.keys())

    mesh = trimesh.Trimesh(sample['points'].numpy(), sample['faces'].numpy(), process=False, maintain_order=True)
    mesh.export('gt1.obj')

