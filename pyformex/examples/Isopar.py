#!/usr/bin/env pyformex --hui
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

from numpy import *
from plugins import elements
import simple


def build_matrix(atoms,x,y=0,z=0):
    """Build a matrix of functions of coords.

    Atoms is a list of text strings representing some function of
    x(,y)(,z). x is a list of x-coordinats of the nodes, y and z can be set
    to lists of y,z coordinates of the nodes.
    Each line of the returned matrix contains the atoms evaluated at a
    node.
    """
    aa = zeros((len(x),len(atoms)),Float)
    for k,a in enumerate(atoms):
        aa[:,k] = eval(a)
    return aa   


def isopar(F,type,coords,oldcoords):
    """Apply isoparametric transform.

    coords and oldcoords can be either arrays, Coords or Formex instances
    """
    isodata = {
        'tri3'  : (2, ('1','x','y')),
        'tri6'  : (2, ('1','x','y','x*x','y*y','x*y')),
        'quad4' : (2, ('1','x','y','x*y')),
        'quad8' : (2, ('1','x','y','x*x','y*y','x*y','x*x*y','x*y*y')),
        'quad9' : (2, ('1','x','y','x*x','y*y','x*y','x*x*y','x*y*y','x*x*y*y')),
        'tet4'  : (3, ('1','x','y','z')),
        'tet10' : (3, ('1','x','y','z','x*x','y*y','z*z','x*y','x*z','y*z')),
        'hex8'  : (3, ('1','x','y','z','x*y','x*z','y*z','x*y*z')),
        'hex27' : (3, ('1','x','y','z','x*x','y*y','z*z','x*y','x*z','y*z',
                       'x*x*y','x*x*z','x*y*y','y*y*z','x*z*z','y*z*z','x*y*z',
                       'x*x*y*y','x*x*z*z','y*y*z*z','x*x*y*z','x*y*y*z',
                       'x*y*z*z',
                       'x*x*y*y*z','x*x*y*z*z','x*y*y*z*z',
                       'x*x*y*y*z*z')),
        }
    ndim,atoms = isodata[type]
    coords = coords.view().reshape(-1,3)
    oldcoords = oldcoords.view().reshape(-1,3)
    x = oldcoords[:,0]
    if ndim > 1:
        y = oldcoords[:,1]
    else:
        y = 0
    if ndim > 2:
        z = oldcoords[:,2]
    else:
        z = 0
    aa = build_matrix(atoms,x,y,z)
    ab = linalg.solve(aa,coords)
    x = F.x().ravel()
    y = F.y().ravel()
    z = F.z().ravel()
    aa = build_matrix(atoms,x,y,z)
    xx = dot(aa,ab)
    xx = reshape(xx,F.shape())
    if ndim < 3:
        xx[...,ndim:] += F.f[...,ndim:]
    return Formex(xx)
##     x = F.x().ravel()
##     if ndim > 1:
##         y = F.y().ravel()
##     else:
##         y = 0
##     if ndim > 2:
##         z = F.z().ravel()
##     else:
##         z = 0
##     aa = build_matrix(atoms,x,y,z)
##     xx = dot(aa,ab)
##     return Formex(reshape(xx,F.shape()))


def base(type,m,n=None):
    """A regular pattern for type.

    type = 'tri' or 'quad' or 'triline' or 'quadline'
    m = number of cells in direction 0
    n = number of cells in direction 1
    """
    n = n or m
    if type == 'triline':
        return Formex(pattern('164')).replic2(m,n,1,1,0,1,0,-1)
    elif type == 'quadline':
        return Formex(pattern('2')).replic2(m+1,n,1,1) + \
               Formex(pattern('1')).replic2(m,n+1,1,1)
    elif type == 'tri':
        return Formex(mpattern('12-34')).replic2(m,n)
    elif type == 'quad':
        return Formex(mpattern('123')).replic2(m,n)
    else:
        raise ValueError,"Unknown type '%s'" % str(type)
    

#F = base('quad',10,10)
v = array(elements.Hex8.vertices)
f = array(elements.Hex8.faces)
F = Formex(v[f]).replic(8,1.,dir=0).replic(8,1.,dir=1).replic(8,1.,dir=2)
clear()
message('This is the base pattern in natural coordinates')
draw(F)
pause()

ll,ur = F.bbox()
sc = array(ur)-array(ll)
sc[2] = 1.
x1 = Formex(simple.regularGrid([0.,0.,0.],[1.,1.,0.],[2,2,0]).reshape(-1,3))
x2 = x1.copy()
x2[5] = x2[2].rot(-22.5)
x2[8] = x2[2].rot(-45.)
x2[7] = x2[2].rot(-67.5)
x2[4] = x2[8] * 0.6

x1 = x1.scale(sc)
x2 = x2.scale(sc)

clear()
message('This is the set of nodes in natural coordinates')
draw(x1,color=black)
message('This is the set of nodes in cartesian coordinates')
draw(x2,color=red)
pause()

G = isopar(F,'quad9',x2.points(),x1.points())
G.setProp(1)

clear()
draw(F)
draw(G)
pause()


clear()
#draw(F)
draw(G)

# End
