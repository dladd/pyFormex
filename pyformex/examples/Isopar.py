#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Isopar

level = 'normal'
topics = ['geometry']
techniques = ['dialog', 'colors']

"""

from plugins.isopar import *
import simple
import elements


ttype = ask("Select type of transformation",['Cancel','1D','2D','3D'])
if not ttype or ttype ==  'Cancel':
    exit()

tdim = int(ttype[0])

# create a unit quadratic grid in tdim dimensions
x = Coords(simple.regularGrid([0.]*tdim, [1.]*tdim, [2]*tdim)).reshape(-1,3)
x1 = Formex(x)
x2 = x1.copy()

# move a few points
if tdim == 1:
    eltype = 'line3'
    x2[1] = x2[1].rot(-22.5)
    x2[2] = x2[2].rot(22.5)
elif tdim == 2:
    eltype = 'quad9'
    x2[5] = x2[2].rot(-22.5)
    x2[8] = x2[2].rot(-45.)
    x2[7] = x2[2].rot(-67.5)
    x2[4] = x2[8] * 0.6
else:
    eltype = 'hex27'
    tol = 0.01
    d = x2.distanceFromPoint(x2[0])
    w = where((d > 0.5+tol) * (d < 1.0 - tol))[0]
    # avoid error messages during projection 
    errh = seterr(all='ignore')
    x2[w] = x2.projectOnSphere(0.5)[w]
    w = where(d > 1.+tol)[0]
    x2[w] = x2.projectOnSphere(1.)[w]
    seterr(**errh)

clear()
message('This is the set of nodes in natural coordinates')
draw(x1,color=blue)
message('This is the set of nodes in cartesian coordinates')
draw(x2,color=red)
drawNumbers(x2,color=red)
drawNumbers(x1)

n = 8
stype = ask("Select type of structure",['Cancel','1D','2D','3D'])
if stype == 'Cancel':
    exit()

sdim = int(stype[0])
if sdim == 1:
    F = simple.line([0.,0.,0.],[1.,1.,0.],10)
elif sdim == 2:
    F = simple.rectangle(1,1,1.,1.)
else:
    v = array(elements.Hex8.vertices)
    f = array(elements.Hex8.faces)
    F = Formex(v[f])

if sdim > 1:
    for i in range(sdim):
        F = F.replic(n,1.,dir=i)

if sdim < tdim:
    F = F.trl(2,0.5)
clear()
message('This is the initial Formex')
FA=draw(F)
sz = F.sizes()


if sdim < tdim:
    sz[sdim:tdim] = 2.
x1 = x1.scale(sz)
x2 = x2.scale(sz)


trf = Isopar(eltype,x2.points(),x1.points())
#G = trf.transformFormex(F)
G = F.isopar(trf)
G.setProp(1)

message('This is the transformed Formex')
draw(G)

sleep(0.8)
undraw(FA)

# End
