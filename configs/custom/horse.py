_base_ = './default_ubd_inward_facing.py'

expname = 'horse'

data = dict(
    datadir='./data/custom/horse/dense',
    factor=2,
    white_bkgd=False,
    rand_bkgd=False
)

