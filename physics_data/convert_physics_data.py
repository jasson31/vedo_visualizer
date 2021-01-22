"""This script converts physics data into h5 file for training the network"""
import re
import os
import sys
import json
import argparse
import numpy as np
import h5py
import itertools
import multiprocessing as mp

from glob import glob
from create_physics_data import PARTICLE_RADIUS
from physics_data_helper import *


OFFSET = 3000

N_TRAIN_FLUID_SCENES = 45
N_VALID_FLUID_SCENES = 5                # total 50
N_FLUID_SCENES = N_TRAIN_FLUID_SCENES + N_VALID_FLUID_SCENES

N_TRAIN_FLUID_OBSTACLE_SCENE = 25
N_VALID_FLUID_OBSTACLE_SCENE = 5        # total 30
N_FLUID_OBSTACLE_SCENE = N_TRAIN_FLUID_OBSTACLE_SCENE + N_VALID_FLUID_OBSTACLE_SCENE


def create_scene_files(scene_dir, out_dir):
    print(scene_dir)
    scene_id = os.path.basename(scene_dir)
    with open(os.path.join(scene_dir, 'scene.json'), 'r') as f:
        scene_dict = json.load(f)

    # create box file
    box_path = os.path.join(out_dir, 'box.h5')
    if not os.path.isfile(box_path):
        box, box_normals = numpy_from_bgeo(os.path.join(scene_dir, 'box.bgeo'))
        box_dict = dict()
        box_dict['box'] = box.astype(np.float32)
        box_dict['box_normals'] = box_normals.astype(np.float32)
        create_h5py(box_dict, box_path)

    # read obstacle file if it exists
    is_obstacle = False
    if os.path.isfile(os.path.join(scene_dir, 'obstacle.bgeo')):
        obs, obs_normals = numpy_from_bgeo(os.path.join(scene_dir, 'obstacle.bgeo'))
        is_obstacle = True

    seed = int(re.findall(r'\d+', scene_id)[0])
    partio_dir = os.path.join(scene_dir, 'partio')
    fluid_ids = get_fluid_ids_from_partio_dir(partio_dir)
    fluid_id_bgeo_map = {
        k: get_fluid_bgeo_files(partio_dir, k) for k in fluid_ids
    }

    frames = None

    for k, v in fluid_id_bgeo_map.items():
        if frames is None:
            frames = list(range(len(v)))
        if len(v) != len(frames):
            raise Exception(
                'number of frames for fluid {} ({}) is different from {}'.
                format(k, len(v), len(frames)))

    # set output file name
    fluid_obj_num = seed % 3 + 1
    fluid_obj_num_folder = 'fluid_obj_' + str(fluid_obj_num)
    obstacle_folder = 'fluid'
    if is_obstacle:
        obstacle_folder = 'fluid_obstacle'
    scene_num = (seed - OFFSET) // 3
    if fluid_obj_num == 1:
        scene_num -= 1

    phase = 'train'
    if N_FLUID_SCENES <= scene_num < (N_FLUID_SCENES + N_TRAIN_FLUID_OBSTACLE_SCENE):
        scene_num -= N_FLUID_SCENES
    # valid fluid scene
    if N_TRAIN_FLUID_SCENES <= scene_num < (N_TRAIN_FLUID_SCENES + N_VALID_FLUID_SCENES):
        phase = 'valid'
        scene_num -= N_TRAIN_FLUID_SCENES
    elif N_TRAIN_FLUID_OBSTACLE_SCENE <= (scene_num-N_FLUID_SCENES) < (N_TRAIN_FLUID_OBSTACLE_SCENE +
                                                                       N_VALID_FLUID_OBSTACLE_SCENE):
        phase = 'valid'
        scene_num -= (N_FLUID_SCENES + N_TRAIN_FLUID_OBSTACLE_SCENE)

    scene_num_folder = 'scene_'+str(scene_num)

    outfile_path = os.path.join(out_dir, phase, fluid_obj_num_folder, obstacle_folder, scene_num_folder)
    os.makedirs(outfile_path, exist_ok=True)
    # collect physics data
    prev_vel = None
    for fr in frames:
        frame_path = os.path.join(outfile_path, '{0}.h5'.format(fr))
        if not os.path.isfile(frame_path):
            feat_dict = {}

            pos = []
            vel = []
            mass = []
            viscosity = []

            sizes = np.array([0, 0, 0, 0], dtype=np.int32)

            for flid in fluid_ids:
                bgeo_path = fluid_id_bgeo_map[flid][fr]
                pos_, vel_ = numpy_from_bgeo(bgeo_path)
                pos.append(pos_)
                vel.append(vel_)
                viscosity.append(
                    np.full(pos_.shape[0:1],
                            scene_dict[flid]['viscosity'],
                            dtype=np.float32))
                mass.append(
                    np.full(pos_.shape[0:1],
                            scene_dict[flid]['density0'],
                            dtype=np.float32))
                sizes[0] += pos_.shape[0]

            pos = np.concatenate(pos, axis=0)
            vel = np.concatenate(vel, axis=0)
            mass = np.concatenate(mass, axis=0)
            mass *= (2 * PARTICLE_RADIUS)**3
            viscosity = np.concatenate(viscosity, axis=0)

            if prev_vel is None:  # first frame
                acc = np.zeros_like(vel)
            else:
                acc = (vel - prev_vel) / (1.0 / 50.0)

            prev_vel = vel

            feat_dict['pos'] = pos.astype(np.float32)
            feat_dict['vel'] = vel.astype(np.float32)
            feat_dict['acc'] = acc.astype(np.float32)
            feat_dict['m'] = mass.astype(np.float32)
            feat_dict['viscosity'] = viscosity.astype(np.float32)

            if is_obstacle:
                feat_dict['obstacle'] = obs.astype(np.float32)
                feat_dict['obstacle_normal'] = obs_normals.astype(np.float32)

            create_h5py(feat_dict, frame_path)


def create_h5py(data, outfile_path):
    hf = h5py.File(outfile_path, 'w')
    for k, v in data.items():
        hf.create_dataset(k, data=v, chunks=True, compression="gzip", compression_opts=9)
    hf.close()


def main():
    parser = argparse.ArgumentParser(
        description=
        "Creates compressed msgpacks for directories with SplishSplash scenes")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The path to the output directory")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="The path to the input directory with the simulation data")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    out_dir = args.output

    scene_dirs = sorted(glob(os.path.join(args.input, '*')))
    # print(scene_dirs)
    # with mp.Pool(processes=12) as pool:
    #     pool.starmap(create_scene_files, zip(scene_dirs, itertools.repeat(out_dir)), chunksize=1) # bugs
    for scene_dir in scene_dirs:
        create_scene_files(scene_dir, out_dir)


if __name__ == '__main__':
    main()