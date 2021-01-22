import glob
import numpy as np

from datasets.SplishSplash import store_h5, load_h5


def init_stat(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def combine_stat(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt((std_0**2 * n_0 + std_1**2 * n_1 + (mean_0 - mean)**2 * n_0 + (mean_1 - mean)**2 * n_1)
                  / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


if __name__ == '__main__':
    # pos, vel, acc
    data_names = ['pos', 'vel', 'acc']
    stats = [init_stat(3), init_stat(3), init_stat(3)]

    for path in glob.glob('../data/train/*/*/*/*.h5'):
        # hf = h5py.File(path, 'r')
        # acc = np.array(hf.get('acc'))
        print(f'Now processing {path}')
        datas = load_h5(data_names, path)

        for j in range(len(stats)):
            stat = init_stat(stats[j].shape[0])
            stat[:, 0] = np.mean(datas[j], axis=0)
            stat[:, 1] = np.std(datas[j], axis=0)
            stat[:, 2] = datas[j].shape[0]
            stats[j] = combine_stat(stats[j], stat)

    store_h5(data_names, stats, '../data/stat.h5')
