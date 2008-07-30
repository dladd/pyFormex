#!/usr/bin/env pyformex --gui
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

#from plugins.isopar import *
import simple
import elements


def build_matrix(atoms,x,y=0,z=0):
    """Build a matrix of functions of coords.

    Atoms is a list of text strings representing some function of
    x(,y)(,z). x is a list of x-coordinats of the nodes, y and z can be set
    to lists of y,z coordinates of the nodes.
    Each line of the returned matrix contains the atoms evaluated at a
    node.
    """
    print len(atoms)
    print len(x)
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
        'hex20' : (3, ('1','x','y','z','x*x','y*y','z*z','x*y','x*z','y*z',
                       'x*x*y','x*x*z','x*y*y','y*y*z','x*z*z','y*z*z','x*y*z',
                       'x*x*y*z','x*y*y*z','x*y*z*z')),
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
    print aa.shape
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
    
# Processing starts here

ttype = ask("Select type of transformation",['Cancel','2D','3D'])
if not ttype or ttype ==  'Cancel':
    exit()

tdim = int(ttype[0])

# create a unit quadratic grid in tdim dimensions
x1 = Formex(simple.regularGrid([0.]*tdim, [1.]*tdim, [2]*tdim).reshape(-1,tdim))
x2 = x1.copy()

# move a few points
if tdim == 2:
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
drawNumbers(x1)


n = 8
stype = ask("Select type of structure",['Cancel','2D','3D'])
if stype == 'Cancel':
    exit()

sdim = int(stype[0])
if sdim == 2:
    F = simple.rectangle(1,1,1.,1.)
else:
    v = array(elements.Hex8.vertices)
    f = array(elements.Hex8.faces)
    F = Formex(v[f])

for i in range(sdim):
    F = F.replic(n,1.,dir=i)
clear()
message('This is the base pattern in natural coordinates')
draw(F)
sz = F.sizes()
#pause()


if sdim < tdim:
    sz[sdim:tdim] = 2.
x1 = x1.scale(sz)
x2 = x2.scale(sz)

G = isopar(F,eltype,x2.points(),x1.points())
G.setProp(1)

clear()
draw(F)
draw(G)
#pause()


clear()
#draw(F)
draw(G)

# End
