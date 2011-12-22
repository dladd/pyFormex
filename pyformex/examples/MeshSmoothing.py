#!/usr/bin/env pyformex --gui
# $Id:$

"""MeshSmoothing

level = 'normal'
topics = ['geometry', 'mesh','illustration']
techniques = ['dialog','smooth]

.. Description

MeshSmoothing
==============
This example illustrates the use of the mesh smoothing algrithm.

The smoothing is applied to a hexahedral, tetrahedral, quadrilateral, and triangular mesh.
"""

from simple import cuboid
from mesh import *

clear()
n = 12        #   Number of elements in each direction
tol = 4       #   Amount of noise added to the coordinates
iter = 10   #   Number of smoothing iterations 

res = askItems(items=[
    _I('n', n, text='Number of elements', itemtype='slider', min=2, max=24),
   _I('tol', tol, text='Noise', itemtype='slider', min=0, max=10),
  _I('iter', iter, text='Smoothing iterations', itemtype='slider', min=1, max=20),  
])

if not res:
    exit()
globals().update(res)

tol /= 10.
cube = cuboid().replic2(n, n, 1., 1.).rep(n, 1., 2).toMesh()
cubeTet = cuboid().toMesh().convert('tet4')
cubeTet += cubeTet.reflect()            #Reflecting is needed to align the edges of adjacent tetrahedrons
cubeTet += cubeTet.reflect(dir=1)
cubeTet += cubeTet.reflect(dir=2)
cubeTet = cubeTet.trl([1., 1., 1.])
cubeTet = cubeTet.toFormex().replic2(n/2, n/2, 2., 2.).rep(n/2., 2., 2).toMesh().convert('tet4').trl(1, -2.*n)
surf = Formex(xpattern('0123', nplex=4)).replic2(2*n, 2*n, 1., 1.).scale(0.5).bump(2, [n/2., n/2., 1.], lambda x: 1.-x**2/n).toMesh().trl(1, -4.*n)
surfTri = surf.convert('tri3').trl(1, -2.*n)

for part in [cube, cubeTet, surf, surfTri]:
    noise = tol * random.random(part.coords.shape) - tol/2.
    noisy = Mesh(part.coords + noise, part.elems, 1).trl(0, 2.*n)
    smoothed = noisy.smooth(iterations=iter).trl(0, 2.*n)
    smoothed.setProp(2)
    draw([part, noisy, smoothed])

zoomAll()

#End
