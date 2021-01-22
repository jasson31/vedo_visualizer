import os
import argparse
import torch
import vedo
import numpy as np

from datasets.SplishSplash import SplishSplashDataset


idx = 0

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


def keyfunc(evt):
    vedo.printc('keyfunc called, pressed key:', evt.keyPressed)
    if evt.keyPressed == '[':
        pass
    elif evt.keyPressed == ']':
        pass

    plt.render()


def slider2(widget, event):
    value = int(widget.GetRepresentation().GetValue())
    pos, path = getpos(value)
    plt.show(pts, f'{idx}: {path}')
    print('slider called, current value:', value)

def buttonfunc():
    print("button pressed")
    global idx
    idx += 1
    print(idx)
    pos, path = getpos(idx)
    plt.show(pts, f'{idx}: {path}')


def print_points(print_idx):
    pos, path = getpos(print_idx)
    show



dataset = SplishSplashDataset(train=True, shuffle=False, window=1)


pos, path = getpos(idx)
plt = vedo.Plotter(axes=1, interactive=True)
pts = vedo.Points(pos, c='b')
plt.addCallback('KeyPress', keyfunc)

data_length = len(dataset)

plt.addSlider2D(slider2, 0, data_length,
               pos=[(0.1, 0.1), (0.9, 0.1)], c="blue", title="alpha value (opacity)")


bu = plt.addButton(
    buttonfunc,
    pos=(0.7, 0.05),  # x,y fraction from bottom left corner
    states=["Next state"],
)

plt.show(pts, f'{idx}: {path}')