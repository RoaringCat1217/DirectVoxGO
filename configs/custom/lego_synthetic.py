_base_ = './default_ubd_inward_facing.py'

expname = 'lego_synthetic'

data = dict(
    datadir='./data/custom/lego_synthetic/dense',
    factor=2,
    white_bkgd=False,
    rand_bkgd=False
)
