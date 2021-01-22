from argparse import Namespace


def get_hparams(options):
    hparams = Namespace()

    hparams.n_step = 3
    hparams.sa_radii = [0.2, ]
    hparams.sa_channels = [[3, 64, 64, 128],
                           [128 + 3, 128, 128, 256],
                           [256 + 3, 256, 512, 1024]]

    return hparams


def get_data_hparams(options):
    data_hparams = Namespace()

    data_hparams.window = [2, 2, 2]
    data_hparams.batch_size = [16, 16, 1]

    return data_hparams
