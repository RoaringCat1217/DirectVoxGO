import torch
from meshio import load_mesh, save_mesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex, RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, SoftPhongShader, SoftSilhouetteShader
from pytorch3d.renderer.cameras import PerspectiveCameras, FoVPerspectiveCameras
from lib.load_data import load_data
import mmcv
import numpy as np
from pytorch3d.renderer.cameras import look_at_view_transform
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend
import argparse
import os
import cv2
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from tensorboardX import SummaryWriter
from tqdm import tqdm


class MyDataset(Dataset):
    def __init__(self, images, Rs, Ts):
        self.images = images
        self.Rs = Rs
        self.Ts = Ts

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.Rs[idx], self.Ts[idx]


class SimpleShader(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.blend_params = BlendParams(background_color=(0., 0., 0.))
        self.device = device

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(texels, fragments, blend_params)
        return images


def update_mesh_shape_prior_losses(mesh, loss):
    # and (b) the edge length of the predicted mesh
    loss["edge"] = mesh_edge_loss(mesh)

    # mesh normal consistency
    loss["normal"] = mesh_normal_consistency(mesh)

    # mesh laplacian smoothing
    loss["laplacian"] = mesh_laplacian_smoothing(mesh, method="uniform")


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True, help='config file path')
    return parser


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    meshdir = os.path.join(cfg.basedir, cfg.expname)

    device = 'cuda'
    verts, colors, normals, faces, _ = load_mesh(os.path.join(meshdir, './mid.obj'))
    verts = torch.tensor(verts)[None, ...].to(device)
    faces = torch.tensor(faces)[None, ...].to(device)
    colors = torch.tensor(colors)[None, ...].to(device)

    data_dict = load_data(cfg.data)
    poses = data_dict['poses']
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

    images = torch.tensor(data_dict['images'])
    batch_size = 4
    dataset = MyDataset(images, R, T)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    offset_v = torch.zeros_like(verts, requires_grad=True)
    offset_c = torch.zeros_like(colors, requires_grad=True)
    optimizer = Adam([{'params': [offset_v], 'lr': 0},
                      {'params': [offset_c], 'lr': 1e-3}],
                     lr=1e-4)

    writer = SummaryWriter(os.path.join(cfg.basedir, cfg.expname, 'refine'))
    raster_settings = RasterizationSettings(
        image_size=data_dict['hwf'][:2],
        blur_radius=1e-3,
        faces_per_pixel=50,
        perspective_correct=False,
        bin_size=-1
    )
    n_epoch = 10
    pbar = tqdm(total=n_epoch * len(poses))
    weight = {
        'edge': 0.,
        'normal': 0.,
        'laplacian': 0.,
        'rgb': 1.,
    }
    step = 0
    for epoch in range(n_epoch):
        for image, R, T in dataloader:
            image = image.to(device)
            camera = FoVPerspectiveCameras(R=R, T=T, znear=0.01, zfar=10.0, fov=fov, device=device)
            textures = TexturesVertex((colors + offset_c).clamp(min=0, max=1))
            mesh = Meshes(verts + offset_v, faces, textures).extend(batch_size)
            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=camera,
                    raster_settings=raster_settings
                ),
                shader=SimpleShader(device=device)
            )
            result = renderer(mesh)
            optimizer.zero_grad()
            loss_dict = {}
            update_mesh_shape_prior_losses(mesh[:1], loss_dict)
            loss_dict['rgb'] = torch.mean(torch.abs(result[..., :3] - image))
            loss = torch.tensor(0., device=device)
            for k in loss_dict:
                loss += loss_dict[k] * weight[k]
            loss.backward()
            print(loss, loss_dict)
            optimizer.step()
            for k in loss_dict:
                writer.add_scalar(k, loss_dict[k].item(), step)
            writer.add_scalar('loss', loss.item(), step)
            step += 1
            pbar.update(batch_size)

        if (epoch + 1) % 1 == 0:
            verts_cpu = verts + offset_v.detach()
            colors_cpu = (colors + offset_c.detach()).clamp(min=0, max=1)
            verts_cpu = verts_cpu.squeeze(0).cpu().numpy()
            colors_cpu = colors_cpu.squeeze(0).cpu().numpy()
            faces_cpu = faces.squeeze(0).cpu().numpy()
            save_mesh(os.path.join(cfg.basedir, cfg.expname, f'refined_{epoch + 1}.obj'), verts_cpu, colors_cpu, None, faces_cpu)
