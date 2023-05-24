_base_ = './default_ubd_inward_facing.py'

expname = 'bottle'

data = dict(
    datadir='./data/custom/bottle/dense',
    factor=2,
    white_bkgd=False,
    rand_bkgd=False
)
