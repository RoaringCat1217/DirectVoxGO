import os.path

import numpy as np
from tqdm import tqdm
from meshio import load_mesh, save_mesh
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from lib.load_data import load_data
import mmcv
from pytorch3d.renderer.cameras import look_at_view_transform
from matplotlib import pyplot as plt
from torch import nn
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
import argparse
import mmcv


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True, help='config file path')
    return parser


def filter_verts(verts_keep, verts, colors, normals, faces):
    remapping = {}
    for i, idx in enumerate(verts_keep):
        remapping[idx] = i
    verts_new = []
    colors_new = []
    normals_new = []
    faces_new = []
    print("Filtering vertices...")
    for idx in tqdm(verts_keep):
        verts_new.append(verts[idx])
        colors_new.append(colors[idx])
        normals_new.append(normals[idx])
    print("Filtering triangles...")
    for i, j, k in tqdm(faces):
        if i in remapping and j in remapping and k in remapping:
            faces_new.append([remapping[i], remapping[j], remapping[k]])
    verts_new = np.array(verts_new, dtype=np.float32)
    colors_new = np.array(colors_new, dtype=np.float32)
    normals_new = np.array(normals_new, dtype=np.float32)
    faces_new = np.array(faces_new, dtype=np.int)
    return verts_new, colors_new, normals_new, faces_new


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    meshdir = os.path.join(cfg.basedir, cfg.expname)

    print("Loading mesh...")
    verts, colors, normals, faces, edges = load_mesh(os.path.join(meshdir, 'mesh.obj'))
    n = len(verts)
    parent = np.arange(n)
    size = np.ones(n)

    def find_root(i):
        if parent[i] == i:
            return i
        root = find_root(parent[i])
        parent[i] = root
        return root

    def union(i, j):
        i = find_root(i)
        j = find_root(j)
        if i == j:
            return
        if size[i] <= size[j]:
            parent[i] = j
            size[j] += size[i]
            size[i] = 0
        else:
            parent[j] = i
            size[i] += size[j]
            size[j] = 0

    print("Processing edges...")
    for i, j in tqdm(edges):
        union(i, j)

    root = np.argmax(size)
    verts_keep = []
    for i in range(n):
        if find_root(i) == root:
            verts_keep.append(i)
    verts_new, colors_new, normals_new, faces_new = filter_verts(verts_keep, verts, colors, normals, faces)

    verts = verts_new
    colors = colors_new
    normals = normals_new
    faces = faces_new

    V = len(verts)
    verts_ = torch.tensor(verts)
    faces_ = torch.tensor(faces)
    verts_ = verts_[None, ...]
    faces_ = faces_[None, ...]
    mesh = Meshes(verts_, faces_)
    device = 'cuda'
    mesh = mesh.to(device)

    cfg = mmcv.Config.fromfile('./configs/custom/controller.py')
    data_dict = load_data(cfg.data)
    poses = data_dict['poses']
    N = len(poses)
    pad = np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 1, 4).repeat(120, 0)
    poses = np.concatenate([poses, pad], axis=1)
    poses = np.linalg.inv(poses)
    poses[:, 0] = -poses[:, 0]
    poses[:, 2] = -poses[:, 2]
    R = poses[:, :3, :3].transpose(0, 2, 1)
    T = poses[:, :3, 3]
    R = torch.tensor(R)
    T = torch.tensor(T)
    h, w, f = data_dict['hwf']
    fov = np.arctan(h / 2 / f) * 2 * 180 / np.pi
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=10.0, fov=fov)

    raster_settings = RasterizationSettings(
        image_size=data_dict['hwf'][:2],
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=-1
    )
    packed_faces = mesh.faces_packed()
    packed_verts = mesh.verts_packed()
    visibility_map = torch.zeros(V, dtype=bool, device=device)
    for i in tqdm(range(N)):
        rasterizer = MeshRasterizer(
            cameras=cameras[i],
            raster_settings=raster_settings
        )
        fragments = rasterizer(mesh)
        pix_to_face = fragments.pix_to_face[fragments.pix_to_face != -1]
        visible_faces = pix_to_face.unique()
        visible_verts_idx = packed_faces[visible_faces].unique()  # (num_visible_faces,  3)
        visibility_map[visible_verts_idx] = True
    verts_keep = []
    for i in range(len(visibility_map)):
        if visibility_map[i].item():
            verts_keep.append(i)
    verts_new, colors_new, normals_new, faces_new = filter_verts(verts_keep, verts, colors, normals, faces)
    print("Saving mesh...")
    save_mesh(os.path.join(meshdir, 'cleaned.obj'), verts_new, colors_new, normals_new, faces_new)

