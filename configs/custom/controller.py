_base_ = './default_ubd_inward_facing.py'

expname = 'controller'

data = dict(
    datadir='./data/custom/controller/dense',
    factor=2,
    white_bkgd=False,
    rand_bkgd=False
)

