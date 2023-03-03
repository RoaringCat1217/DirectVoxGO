from run import *
from matplotlib import pyplot as plt


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', required=True, help='config file path')
    return parser

def normalize(x):
    return x / np.linalg.norm(x)


if __name__ == '__main__':
    os.chdir("/home2/linyf/GradDesign/DirectVoxGO")
    # load setup
    parser = config_parser()
    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)

    # init enviroment
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load images / poses / camera settings / data split
    data_dict = load_everything(args=args, cfg=cfg)

    ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
    ckpt_name = ckpt_path.split('/')[-1][:-4]
    if cfg.data.ndc:
        model_class = dmpigo.DirectMPIGO
    elif cfg.data.unbounded_inward:
        model_class = dcvgo.DirectContractedVoxGO
    else:
        model_class = dvgo.DirectVoxGO
    model = utils.load_model(model_class, ckpt_path).to(device)
    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'model': model,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
        },
    }

    HW = data_dict['HW'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0)
    Ks = data_dict['Ks'][data_dict['i_test']][[0]].repeat(len(data_dict['render_poses']), 0)
    render_kwargs = render_viewpoints_kwargs['render_kwargs']


    @torch.no_grad()
    def render(origin, lookat):
        origin = np.array(origin)
        lookat = np.array(lookat)
        up = np.array([0, -1., 0])
        vec2 = normalize(origin - lookat)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        c2w = np.stack([vec0, vec1, vec2, origin], 1).astype(np.float32)
        c2w = torch.tensor(c2w)

        H, W = HW[0]
        K = Ks[0]
        c2w = torch.Tensor(c2w)
        ndc = render_viewpoints_kwargs['ndc']
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
            H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
            flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last', 'density']
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        render_result_chunks = [
            {k: v for k, v in model(ro, rd, vd, **render_kwargs).items() if k in keys}
            for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        render_result = {
            k: torch.cat([ret[k] for ret in render_result_chunks]).reshape(H, W, -1)
            for k in render_result_chunks[0].keys()
        }
        rgb = render_result['rgb_marched'].cpu().numpy()
        depth = render_result['depth'].cpu().numpy()
        bgmap = render_result['alphainv_last'].cpu().numpy()
        plt.imshow(rgb)
        plt.show()

    @torch.no_grad()
    def density(origin):
        density = model.sample_density(torch.tensor(origin[None].astype(np.float32)), **render_kwargs)
        return density.item()
