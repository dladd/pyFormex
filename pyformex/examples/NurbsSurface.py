#!/usr/bin/pyformex --gui
# $Id$

"""NurbsSurface

level = 'advanced'
topics = ['geometry', 'surface']
techniques = ['nurbs']

.. Description

Nurbs
=====
This example is under development.
"""

clear()
smooth()

from plugins.nurbs import *

# size of the grid
nx,ny = 6,4

# degree of the NURBS surface
px,py = 3,2

# grid position and value of peaks
peaks = [
    (1,1,3.),
    (2,2,-2.)
    ]

# number of random points
nP = 100


X = Formex(origin()).replic2(nx,ny).coords.reshape(ny,nx,3)
for x,y,v in peaks:
    X[x,y,2] = v

# draw the numbered control points
draw(X,nolight=True)
drawNumbers(X.reshape(-1,3),trl=[0.05,0.05,0.0])

# draw 1st degree curves in both parametric directions
draw([NurbsCurve(X[i],degree=1) for i in range(ny)],color=red,nolight=True,ontop=True)
draw([NurbsCurve(X[:,i],degree=1) for i in range(nx)],color=blue,nolight=True,ontop=True)

# draw surface
S = NurbsSurface(X,degree=(px,py))
draw(S,color=magenta,bkcolor=cyan)

# add random points

u = random.random(2*nP).reshape(-1,2)
P = S.pointsAt(u)
draw(P,color=black,nolight=True,bbox='last',ontop=True)

# End
