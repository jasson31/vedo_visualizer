"""Manually build a mesh from points and faces"""
from vedo import Mesh, printc, show

verts = [(50,50,50), (70,40,50), (50,40,80), (80,70,50)]
faces = [(0,1,2), (2,1,3), (1,0,3), (0, 2, 3)]



verts = [(-1.5,0,-1.5), (-1.5,0,1.5),(1.5,0,1.5),(1.5,0,-1.5),(-1.5,2,-1.5), (-1.5,2,1.5),(1.5,2,1.5),(1.5,2,-1.5)]
faces = [(3,2,1,0),(0,1,5,4),(4,5,6,7),(2,3,7,6),(1,2,6,5),(4,7,3,0)]




mesh = Mesh([verts, faces])
mesh.frontFaceCulling(True).backColor('violet').lineColor('tomato').lineWidth(2)
labs = mesh.labels('id')


show(mesh, labs, __doc__, viewup='z', axes=1)