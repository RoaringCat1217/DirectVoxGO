_base_ = './default_ubd_inward_facing.py'

expname = 'cup'

data = dict(
    datadir='./data/custom/cup/dense',
    factor=2,
    white_bkgd=False,
    rand_bkgd=False
)
