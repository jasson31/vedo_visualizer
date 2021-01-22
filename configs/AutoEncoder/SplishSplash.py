from argparse import Namespace


def get_hparams(options):
    hparams = Namespace()

    hparams.n_grid = [60, 105, 60]
    hparams.grid_min = [-30 * 0.05, -1 * 0.05, -30 * 0.05]
    hparams.dx = 0.05

    return hparams


def get_data_hparams(options):
    data_hparams = Namespace()

    data_hparams.window = [2, 2, 2]
    data_hparams.batch_size = [16, 16, 1]

    return data_hparams
