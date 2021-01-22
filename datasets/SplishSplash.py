import os
import numpy as np
import h5py
import torch
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader


def store_h5(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_h5(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


class SplishSplashDataset(Dataset):
    def __init__(self, train=True, window=2, shuffle=False):
        # dataset parameters
        if train:
            self.phase = 'train'
            self.n_fluid_scenes = 3
            self.n_fluid_and_obs_scene = 0
        else:
            self.phase = 'valid'
            self.n_fluid_scenes = 0
            self.n_fluid_and_obs_scene = 0
        self.n_frames = 401
        self.window = window
        self.shuffle = shuffle
        self.data_dir = 'data'
        # self.data_names = ['pos', 'vel', 'acc', 'm', 'viscosity']
        self.data_names = ['pos', 'vel', 'acc']  # TODO: get data_names from args

        if train:
            stat_path = os.path.join(self.data_dir, 'stat.h5')
            stat_np = load_h5(self.data_names, stat_path)
            self.stat = []
            for i in range(len(stat_np)):
                self.stat.append(torch.from_numpy(stat_np[i]).float())

        self.len_fluid_obj = (self.n_fluid_scenes + self.n_fluid_and_obs_scene) * (self.n_frames - self.window + 1)
        self.len_fluid_scene = (self.n_frames - self.window + 1)

        np.random.seed()
        # data access lists
        self.fluid_obj_dir_list = np.arange(1)
        if shuffle:
            np.random.shuffle(self.fluid_obj_dir_list)
        self.fluid_scene_dir_list = np.arange(self.n_fluid_scenes + self.n_fluid_and_obs_scene)
        self.fluid_scene_frame_dir_list = np.arange(self.n_frames - self.window + 1)

    def __len__(self):
        return self.len_fluid_obj

    def __getitem__(self, idx):
        fluid_scene_idx = idx % self.len_fluid_obj
        fluid_obj_idx = idx // self.len_fluid_obj
        fluid_scene_frame_idx = fluid_scene_idx % self.len_fluid_scene
        fluid_scene_idx = fluid_scene_idx // self.len_fluid_scene

        fluid_obj_dir = 'fluid_obj_' + str(self.fluid_obj_dir_list[fluid_obj_idx]+1)
        if self.shuffle and fluid_scene_idx == 0 and fluid_scene_frame_idx == 0:
            np.random.shuffle(self.fluid_scene_dir_list)
        fluid_scene_num = self.fluid_scene_dir_list[fluid_scene_idx]
        if fluid_scene_num < self.n_fluid_scenes:
            fluid_scene_dir = os.path.join('fluid', 'scene_'+str(fluid_scene_num))
        else:
            fluid_scene_dir = os.path.join('fluid_obstacle', 'scene_'+str(fluid_scene_num-self.n_fluid_scenes))
        if self.shuffle and fluid_scene_frame_idx == 0:
            np.random.shuffle(self.fluid_scene_frame_dir_list)
        fluid_scene_frame_num = self.fluid_scene_frame_dir_list[fluid_scene_frame_idx]

        item = dict()
        if self.phase == 'train':
            item['stat'] = self.stat
        for w_idx in range(self.window):
            fluid_scene_frame_path = str(fluid_scene_frame_num+w_idx)+'.h5'
            data_path = os.path.join(self.data_dir, self.phase, fluid_obj_dir, fluid_scene_dir, fluid_scene_frame_path)
            if fluid_scene_num < self.n_fluid_scenes:
                data = load_h5(self.data_names, data_path)
            else:
                data = load_h5(self.data_names+['obstacle', 'obstacle_normal'], data_path)

            box_path = os.path.join(self.data_dir, 'box.h5')
            box_data = load_h5(['box', 'box_normals'], box_path)
            data = data + box_data

            for i in range(len(data)):
                data[i] = torch.from_numpy(data[i]).float()

            item['data'+str(w_idx)] = data
            item['data_path'+str(w_idx)] = data_path

        return item


class SplishSplashDataModule(pl.LightningDataModule):
    def __init__(self, data_hparams):
        super(SplishSplashDataModule, self).__init__()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.window = data_hparams.window
        self.batch_size = data_hparams.batch_size

    def prepare_data(self):
        # it's possible to put create_data.sh in this function...
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SplishSplashDataset(train=True, window=self.window[0], shuffle=True)
            self.valid_dataset = SplishSplashDataset(train=False, window=self.window[1], shuffle=False)
        if stage == 'test' or stage is None:
            self.test_dataset = SplishSplashDataset(train=False, window=self.window[2], shuffle=False)

    def train_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=False)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size[0],
                          pin_memory=True,
                          shuffle=False,
                          sampler=sampler)

    def val_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.valid_dataset, shuffle=False)
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size[0],
                          pin_memory=True,
                          shuffle=False,
                          sampler=sampler)

    def test_dataloader(self):
        sampler = torch.utils.data.distributed.DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size[0],
                          pin_memory=True,
                          shuffle=False,
                          sampler=sampler)


if __name__ == '__main__':
    import time

    dataset_train = SplishSplashDataset(train=True, shuffle=False, window=1)
    dataset_valid = SplishSplashDataset(train=False, shuffle=False, window=1)

    loader_train = DataLoader(dataset_train,
                              batch_size=1,
                              pin_memory=True)

    loader_valid = DataLoader(dataset_valid,
                              batch_size=1,
                              pin_memory=True)

    grid_min = torch.tensor([-30 * 0.05, -1 * 0.05, -30 * 0.05])
    dx = 0.05
    n_grid = [60, 105, 60]

    max_x = -np.inf
    max_y = -np.inf
    max_z = -np.inf

    min_x = np.inf
    min_y = np.inf
    min_z = np.inf

    log_file = "/home/abslon/Documents/dataset_valid.txt"

    for i, item in enumerate(loader_valid):
        if i > 0 and i % 100 == 0:
            print(i, max_x, max_y, max_z, min_x, min_y, min_z)
        data = item['data0']
        data_path = item['data_path0']
        if len(data) == 5:
            pos, _, _, _, _ = data  # pos vel acc box box_norm
        else:
            pos, _, _, _, _, _, _ = data  # pos vel acc obs obs_norm box box_norm

        Xp = (pos - grid_min) / dx  # [B, N, 3]
        base = Xp.int() + 1

        base = base.view(-1, 3)

        max = torch.max(base, dim=0)
        min = torch.min(base, dim=0)

        max_x = int(max.values[0]) if int(max.values[0]) > max_x else max_x
        max_y = int(max.values[1]) if int(max.values[1]) > max_y else max_y
        max_z = int(max.values[2]) if int(max.values[2]) > max_z else max_z

        min_x = int(min.values[0]) if int(min.values[0]) < min_x else min_x
        min_y = int(min.values[1]) if int(min.values[1]) < min_y else min_y
        min_z = int(min.values[2]) if int(min.values[2]) < min_z else min_z

        if max_x > n_grid[0] + 2 or max_y > n_grid[1] + 2 or max_z > n_grid[2] + 2:
            print("max: invalid data found, index is: {0}".format(i))
            with open(log_file, "a") as f:
                f.write("max: invalid data found, index is: {0}\n".format(i))
                f.write("data path is: {0}\n".format(data_path))
                f.write("{0}, {1}, {2}\n\n".format(max_x, max_y, max_z))
            max_x = max_y = max_z = -np.inf
        if min_x < 1 or min_y < 1 or min_z < 1:
            print("min: invalid data found, index is: {0}".format(i))
            with open(log_file, "a") as f:
                f.write("min: invalid data found, index is: {0}\n".format(i))
                f.write("data path is: {0}\n".format(data_path))
                f.write("{0}, {1}, {2}\n\n".format(min_x, min_y, min_z))
            min_x = min_y = min_z = np.inf

    print(max_x, max_y, max_z)