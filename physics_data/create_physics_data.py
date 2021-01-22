import argparse
import os
import numpy as np
import subprocess
import tempfile
import itertools
import open3d as o3d
import json

from glob import glob
from copy import deepcopy
from shutil import copyfile
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
from splishsplash_config import SIMULATOR_BIN, VOLUME_SAMPLING_BIN
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from physics_data_config import *


def random_rotation_matrix(strength=None, dtype=None):
    """Generates a random rotation matrix

    strength: scalar in [0,1]. 1 generates fully random rotations. 0 generates the identity. Default is 1.
    dtype: output dtype. Default is np.float32
    """
    if strength is None:
        strength = 1.0

    if dtype is None:
        dtype = np.float32

    x = np.random.rand(3)
    theta = x[0] * 2 * np.pi * strength
    phi = x[1] * 2 * np.pi
    z = x[2] * strength

    r = np.sqrt(z)
    V = np.array([np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)])

    st = np.sin(theta)
    ct = np.cos(theta)

    Rz = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    rand_R = (np.outer(V, V) - np.eye(3)).dot(Rz)
    return rand_R.astype(dtype)


def rotation_matrix_to_axis_angle(R):
    r = Rotation.from_matrix(R)
    rotvec = r.as_rotvec()

    angle = np.linalg.norm(rotvec)
    if angle == 0:
        vec = rotvec
    else:
        vec = rotvec / angle
    return vec, angle


def obj_volume_to_particles(objpath, scale=1, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'out.bgeo')
        scale_str = '{0}'.format(scale)
        radius_str = str(radius)
        status = subprocess.run([
            VOLUME_SAMPLING_BIN, '-i', objpath, '-o', outpath, '-r', radius_str,
            '-s', scale_str
        ])
        return numpy_from_bgeo(outpath)


def obj_surface_to_particles(objpath, num_points=None, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)
    particle_area = np.pi * radius**2
    # 1.9 to roughly match the number of points of SPlisHSPlasHs surface sampling
    if num_points is None:
        num_points = int(1.9 * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True)
    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)
    return points, normals


def fluid_block_to_particles(start, end, radius):
    size = end - start
    particle_size = (size / radius).astype(int)

    points = list()
    for x in range(particle_size[0]):
        for y in range(particle_size[1]):
            for z in range(particle_size[2]):
                offset = np.array([x, y, z]) * radius
                pos = start + offset
                points.append(pos)

    points = np.stack(points, axis=0)
    return points


def rasterize_points(points, voxel_size, particle_radius):
    if not (voxel_size > 2 * particle_radius):
        raise ValueError(
            "voxel_size > 2*particle_radius is not true. {} > 2*{}".format(
                voxel_size, particle_radius))

    points_min = (points - particle_radius).min(axis=0)
    points_max = (points + particle_radius).max(axis=0)

    arr_min = np.floor_divide(points_min, voxel_size).astype(np.int32)
    arr_max = np.floor_divide(points_max, voxel_size).astype(np.int32) + 1

    arr_size = arr_max - arr_min

    arr = np.zeros(arr_size)

    offsets = []
    for z in range(-1, 2, 2):
        for y in range(-1, 2, 2):
            for x in range(-1, 2, 2):
                offsets.append(
                    np.array([
                        z * particle_radius, y * particle_radius,
                        x * particle_radius
                    ]))

    for offset in offsets:
        idx = np.floor_divide(points + offset, voxel_size).astype(
            np.int32) - arr_min
        arr[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return arr_min, voxel_size, arr


def find_valid_fluid_start_positions(box_rasterized, fluid_rasterized):
    """Tries to find a valid starting position using the rasterized free space and fluid"""
    fluid_shape = np.array(fluid_rasterized[2].shape)
    box_shape = np.array(box_rasterized[2].shape)
    last_pos = box_shape - fluid_shape

    valid_fluid_start_positions_arr = np.zeros(box_shape)
    for idx in itertools.product(range(0, last_pos[0] + 1),
                                 range(0, last_pos[1] + 1),
                                 range(0, last_pos[2] + 1)):
        pos = np.array(idx, np.int32)
        pos2 = pos + fluid_shape
        view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
        if np.alltrue(
                np.logical_and(view, fluid_rasterized[2]) ==
                fluid_rasterized[2]):
            if idx[1] == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1
            elif np.count_nonzero(valid_fluid_start_positions_arr[idx[0],
                                                                  0:idx[1],
                                                                  idx[2]]) == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1

    valid_pos = np.stack(np.nonzero(valid_fluid_start_positions_arr), axis=-1)
    selected_pos = valid_pos[np.random.randint(0, valid_pos.shape[0])]

    # update the rasterized bounding box volume by substracting the fluid volume
    pos = selected_pos
    pos2 = pos + fluid_shape
    view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
    box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1],
                      pos[2]:pos2[2]] = np.logical_and(
                          np.logical_not(fluid_rasterized[2]), view)

    selected_pos += box_rasterized[0]
    selected_pos = selected_pos.astype(np.float) * box_rasterized[1]

    return selected_pos


# create fluids and place them randomly
def create_fluid_object(fluid_shapes, bb_rasterized):
    idx = np.random.randint(len(fluid_shapes))
    fluid_obj = fluid_shapes[idx]
    fluid = obj_volume_to_particles(fluid_obj,
                                    scale=default_fluid_scale[idx])[0]
    R = random_rotation_matrix(1.0)
    fluid = fluid @ R

    fluid_rasterized = rasterize_points(fluid, 2.01 * PARTICLE_RADIUS, PARTICLE_RADIUS)

    selected_pos = find_valid_fluid_start_positions(bb_rasterized,
                                                    fluid_rasterized)
    fluid_pos = selected_pos - fluid_rasterized[0] * fluid_rasterized[1]
    fluid += fluid_pos

    fluid_vel = np.zeros_like(fluid)
    max_vel = MAX_FLUID_START_VELOCITY_XZ
    fluid_vel[:, 0] = np.random.uniform(-max_vel, max_vel)
    fluid_vel[:, 2] = np.random.uniform(-max_vel, max_vel)
    max_vel = MAX_FLUID_START_VELOCITY_Y
    fluid_vel[:, 1] = np.random.uniform(-max_vel, max_vel)

    # ignore all density/viscosity options
    # density = np.random.uniform(500, 2000)
    # viscosity = np.random.exponential(scale=1 / 20) + 0.01
    # if options.uniform_viscosity:
    #     viscosity = np.random.uniform(0.01, 0.3)
    # elif options.log10_uniform_viscosity:
    #     viscosity = 0.01 * 10 ** np.random.uniform(0.0, 1.5)
    #
    # if options.default_density:
    #     density = 1000
    # if options.default_viscosity:
    #     viscosity = 0.01

    density = 1000
    viscosity = 0.01

    return {
        'type': 'fluid',
        'positions': fluid,
        'velocities': fluid_vel,
        'density': density,
        'viscosity': viscosity,
    }


def find_valid_rigid_start_positions(box_rasterized, rigid_rasterized):
    """Tries to find a valid starting position using the rasterized free space and rigid"""
    rigid_shape = np.array(rigid_rasterized[2].shape)
    box_shape = np.array(box_rasterized[2].shape)
    last_pos = box_shape - rigid_shape

    valid_rigid_start_positions_arr = np.zeros(box_shape)
    for idx in itertools.product(range(0, last_pos[0] + 1),
                                 range(0, last_pos[1] + 1),
                                 range(0, last_pos[2] + 1)):
        pos = np.array(idx, np.int32)
        pos2 = pos + rigid_shape
        view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
        if np.alltrue(
                np.logical_and(view, rigid_rasterized[2]) ==
                rigid_rasterized[2]):
            if idx[1] == 3:
                valid_rigid_start_positions_arr[idx[0], idx[1], idx[2]] = 1
            # if idx[1] == 0:
            #     valid_rigid_start_positions_arr[idx[0], idx[1], idx[2]] = 1
            # elif np.count_nonzero(valid_rigid_start_positions_arr[idx[0],
            #                                                       0:idx[1],
            #                                                       idx[2]]) == 0:
            #     valid_rigid_start_positions_arr[idx[0], idx[1], idx[2]] = 1

    valid_pos = np.stack(np.nonzero(valid_rigid_start_positions_arr), axis=-1)
    selected_pos = valid_pos[np.random.randint(0, valid_pos.shape[0])]

    # update the rasterized bounding box volume by substracting the rigid volume
    pos = selected_pos
    pos2 = pos + rigid_shape
    view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
    box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1],
                      pos[2]:pos2[2]] = np.logical_and(
                          np.logical_not(rigid_rasterized[2]), view)

    selected_pos += box_rasterized[0]
    selected_pos = selected_pos.astype(np.float) * box_rasterized[1]

    return selected_pos


# create rigid objects and place them randomly
def create_rigid_object(rigid_obj, R, bb_rasterized):
    rigid = obj_volume_to_particles(rigid_obj)[0]
    rigid = rigid @ R

    rigid_rasterized = rasterize_points(rigid, 2.01 * PARTICLE_RADIUS, PARTICLE_RADIUS)

    selected_pos = find_valid_rigid_start_positions(bb_rasterized,
                                                    rigid_rasterized)
    #rigid_pos = selected_pos - rigid_rasterized[0] * rigid_rasterized[1]
    rigid_pos = selected_pos + (np.array(rigid_rasterized[2].shape) / 2 * rigid_rasterized[1])
    return {
        'type': 'rigid',
        'positions': rigid_pos
    }


# create data with randomly placed fluid objects and rigid objects
def create_data(output_dir, seed, options):
    """Creates a random scene for a specific seed and runs the simulator"""
    np.random.seed(seed)
    script_dir = os.path.dirname(__file__)

    # convert bounding box to particles
    boundary_box = os.path.join(script_dir, 'models', 'Box2.obj')
    bb, bb_normals = obj_surface_to_particles(boundary_box)
    bb_vol = obj_volume_to_particles(boundary_box)[0]

    fluid_shapes = sorted(glob(os.path.join(script_dir, 'models', 'Fluid*.obj')))
    # obstacle_shapes = sorted(glob(os.path.join(script_dir, 'models', 'Rigid*.obj')))
    obstacle_shapes = [os.path.join(script_dir, 'models', 'Rigid_001.obj')]

    num_objects = seed % 3 + 1
    print('num_objects', num_objects)

    # randomly placed fluid object's position can be invalid
    scene_is_valid = False

    for create_scene_i in range(100):
        if scene_is_valid:
            break

        # rasterize free volume
        bb_rasterized = rasterize_points(np.concatenate([bb_vol, bb], axis=0),
                                         2.01 * PARTICLE_RADIUS,
                                         PARTICLE_RADIUS)
        bb_rasterized = bb_rasterized[0], bb_rasterized[1], binary_erosion(
            bb_rasterized[2], structure=np.ones((3, 3, 3)), iterations=3)

        # create obstacle
        if options.obstacle:
            create_success = False
            for i in range(10):
                if create_success:
                    break
                try:
                    obstacle_obj = np.random.choice(obstacle_shapes)
                    R = random_rotation_matrix(1.0)
                    obstacle_info = create_rigid_object(obstacle_obj, R, bb_rasterized)
                    create_success = True
                    print('create obstacle success')
                except:
                    print('create obstacle failed')
                    pass

        objects = []

        create_fn_list = [create_fluid_object]

        for object_i in range(num_objects):

            create_fn = np.random.choice(create_fn_list)

            create_success = False
            for i in range(10):
                if create_success:
                    break
                try:
                    obj = create_fn(fluid_shapes, bb_rasterized)
                    objects.append(obj)
                    create_success = True
                    print('create object success')
                except:
                    print('create object failed')
                    pass

        scene_is_valid = True

        # ignore all scene options
        # total_number_of_fluid_particles = get_total_number_of_fluid_particles()
        #
        # if options.const_fluid_particles:
        #     if options.const_fluid_particles > total_number_of_fluid_particles:
        #         scene_is_valid = False
        #     else:
        #         while get_total_number_of_fluid_particles(
        #         ) != options.const_fluid_particles:
        #             difference = get_total_number_of_fluid_particles(
        #             ) - options.const_fluid_particles
        #             obj_idx, num_particles = get_smallest_fluid_object()
        #             if num_particles < difference:
        #                 del objects[obj_idx]
        #             else:
        #                 objects[obj_idx]['positions'] = objects[obj_idx][
        #                                                     'positions'][:-difference]
        #                 objects[obj_idx]['velocities'] = objects[obj_idx][
        #                                                      'velocities'][:-difference]
        #
        # if options.max_fluid_particles:
        #     if options.max_fluid_particles < total_number_of_fluid_particles:
        #         scene_is_valid = False

    sim_directory = os.path.join(output_dir, 'sim_{0:04d}'.format(seed))
    os.makedirs(sim_directory, exist_ok=False)

    # generate scene json file
    scene = {
        'Configuration': default_configuration,
        'Simulation': default_simulation,
        # 'Fluid': default_fluid,
        'RigidBodies': [],
        'FluidModels': [],
    }
    rigid_body_next_id = 1

    # bounding box
    box_output_path = os.path.join(sim_directory, 'box.bgeo')
    write_bgeo_from_numpy(box_output_path, bb, bb_normals)

    box_obj_output_path = os.path.join(sim_directory, 'box.obj')
    copyfile(boundary_box, box_obj_output_path)

    rigid_body = deepcopy(default_rigidbody)
    rigid_body['id'] = rigid_body_next_id
    rigid_body_next_id += 1
    rigid_body['geometryFile'] = os.path.basename(
        os.path.abspath(box_obj_output_path))
    rigid_body['resolutionSDF'] = [64, 64, 64]
    rigid_body["collisionObjectType"] = 5
    scene['RigidBodies'].append(rigid_body)

    # obstacle
    if options.obstacle:
        obstacle_output_path = os.path.join(sim_directory, 'obstacle.bgeo')
        obs, obs_normals = obj_surface_to_particles(obstacle_obj, num_points=default_obstacle_size)
        obs = obs @ R
        obs_normals = obs_normals @ R
        write_bgeo_from_numpy(obstacle_output_path, obs, obs_normals)

        obstacle_obj_output_path = os.path.join(sim_directory, 'obstacle.obj')
        copyfile(obstacle_obj, obstacle_obj_output_path)

        obstacle = deepcopy(default_rigidbody)
        obstacle['id'] = rigid_body_next_id
        rigid_body_next_id += 1
        obstacle['translation'] = obstacle_info['positions'].tolist()
        vec, angle = rotation_matrix_to_axis_angle(R)
        obstacle['rotationAxis'] = vec.tolist()
        obstacle['rotationAngle'] = angle
        obstacle['geometryFile'] = os.path.basename(
            os.path.abspath(obstacle_obj_output_path))
        obstacle['resolutionSDF'] = [64, 64, 64]
        obstacle["collisionObjectType"] = 5
        scene['RigidBodies'].append(obstacle)

    fluid_count = 0
    for obj in objects:
        fluid_id = 'fluid{0}'.format(fluid_count)
        fluid_count += 1
        fluid = deepcopy(default_fluid)
        fluid['viscosity'] = obj['viscosity']
        fluid['density0'] = obj['density']
        scene[fluid_id] = fluid

        fluid_model = deepcopy(default_fluidmodel)
        fluid_model['id'] = fluid_id

        fluid_output_path = os.path.join(sim_directory, fluid_id + '.bgeo')
        write_bgeo_from_numpy(fluid_output_path, obj['positions'],
                              obj['velocities'])
        fluid_model['particleFile'] = os.path.basename(fluid_output_path)
        scene['FluidModels'].append(fluid_model)

    scene_output_path = os.path.join(sim_directory, 'scene.json')
    with open(scene_output_path, 'w') as f:
        json.dump(scene, f, indent=4)

    run_simulator(os.path.abspath(scene_output_path), sim_directory)


def run_simulator(scene, output_dir):
    """Runs the simulator for the specified scene file"""
    # SIMULATOR_BIN, '--no-cache', '--no-gui', '--no-initial-pause',
    with tempfile.TemporaryDirectory() as tmpdir:
        status = subprocess.run([
            SIMULATOR_BIN, '--no-cache', '--no-gui',  '--no-initial-pause',
            '--output-dir', output_dir, scene
        ])


def main():
    parser = argparse.ArgumentParser('Creates SplishSplash simulator data')
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The path to the output directory")
    parser.add_argument("--seed",
                        type=int,
                        required=True,
                        help="The random seed for initialization")
    parser.add_argument("--obstacle",
                        dest='obstacle',
                        action='store_true',
                        help="add obstacle to physics data scene")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    create_data(args.output, args.seed, args)


if __name__ == '__main__':
    import sys
    sys.exit(main())


