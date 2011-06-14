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

from plugins.nurbs import *

clear()
smooth()

nx,ny = 4,4  # Currently, nx and ny need to be equal !
X = Formex(origin()).replic2(nx,ny).coords.reshape(ny,nx,3)
X[1,1] += [0.,0.,2.]
draw(X,nolight=True)
drawNumbers(X.reshape(-1,3),trl=[0.05,0.05,0.0])

draw([NurbsCurve(X[i],degree=1) for i in range(ny)],color=red,nolight=True,ontop=True)
draw([NurbsCurve(X[:,i],degree=1) for i in range(nx)],color=blue,nolight=True,ontop=True)

S = NurbsSurface(X,degree=(2,1))
draw(S,color=magenta)


# End
