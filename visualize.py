import os
import argparse
import torch
import vedo
import numpy as np

from datasets.SplishSplash import SplishSplashDataset

data_idx = 0
dataset = SplishSplashDataset(train=True, shuffle=False, window=1)
data_length = len(dataset)

def getpos(i):
    data = dataset[i]['data0']
    path = dataset[i]["data_path0"]

    for j in range(len(data)):
        data[j] = data[j]

    if len(data) == 5:
        pos, vel, acc, _, _ = data
    else:
        pos, vel, acc, _, _, _, _ = data

    return pos.numpy(), path


def keyfunc(key):
    global data_idx
    if key == 'Left' and data_idx > 0:
        data_idx -= 1
    elif key == 'Right' and data_idx < data_length - 1:
        data_idx += 1

    pos, path = getpos(data_idx)
    plt.clear()

    plt.add(vedo.Points(pos, c='b'))
    plt.add(vedo.Text2D(path, pos=(.02, .02), c='k'))

    plt.add(boundary_mesh)
    plt.render()


plt = vedo.Plotter(interactive=False)

pos, path = getpos(data_idx)

pts = vedo.Points(pos, c='b')
plt.keyPressFunction = keyfunc

data_info = vedo.Text2D(path, pos=(.02, .02), c='k')


verts = [(-1.5, 0, -1.5), (-1.5, 0, 1.5), (1.5, 0, 1.5), (1.5, 0, -1.5), (-1.5, 5, -1.5), (-1.5, 5, 1.5), (1.5, 5, 1.5),
         (1.5, 5, -1.5)]
# faces = [(3,2,1,0),(0,1,5,4),(4,5,6,7),(2,3,7,6),(1,2,6,5),(4,7,3,0)]
faces = [(3, 2, 1, 0), (0, 1, 5, 4), (4, 7, 3, 0)]

boundary_mesh = vedo.Mesh([verts, faces]).lineColor('black').lineWidth(1)

plt += boundary_mesh
plt += pts
plt += data_info


plt.show()

vedo.interactive()
