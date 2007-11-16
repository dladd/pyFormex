#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""inertia.py

Compute inertia related quantities of a Formex.
This comprises: center of gravity, inertia tensor, principal axes

Currently, these functions work on arrays of nodes, not on Formices!
Use func(F,f) to operate on a Formex F.
"""

from numpy import *

def centroids(X):
    """Compute the centroids of the points of a set of elements.

    X (nelems,nplex,3)
    """
    return X.sum(axis=1) / X.shape[1]


def center(X,mass=None):
    """Compute the center of gravity of an array of points.

    mass is an optional array of masses to be atributed to the
    points. The default is to attribute a mass=1 to all points.

    If you also need the inertia tensor, it is more efficient to
    use the inertia() function. 
    """
    X = X.reshape((-1,X.shape[-1]))
    if mass is not None:
        mass = array(mass)
        ctr = (X*mass).mean(axis=0) / mass.sum()
    else:
        ctr = X.mean(axis=0)
    return ctr


def inertia(X,mass=None):
    """Compute the inertia tensor of an array of points.

    mass is an optional array of masses to be atributed to the
    points. The default is to attribute a mass=1 to all points.

    The result is a tuple of two float arrays:
      - the center of gravity: shape (3,)
      - the inertia tensor: shape (6,) with the following values (in order):
        Ixx, Iyy, Izz, Ixy, Ixz, Iyz 
    """
    X = X.reshape((-1,X.shape[-1]))
    if mass is not None:
        mass = array(mass)
        ctr = (X*mass).mean(axis=0) / mass.sum()
    else:
        ctr = X.mean(axis=0)
    Xc = X - ctr
    x,y,z = Xc[:,0],Xc[:,1],Xc[:,2]
    xx,yy,zz,yz,zx,xy = x*x, y*y, z*z, y*z, z*x, x*y
    I = column_stack([ yy+zz, zz+xx, xx+yy, -yz, -zx, -xy ])
    if mass is not None:
        I *= mass
    return ctr,I.sum(axis=0)


def principal(inertia):
    """Returns the principal values and axes of the inertia tensor."""
    Ixx,Iyy,Izz,Iyz,Izx,Ixy = inertia
    Itensor = array([ [Ixx,Ixy,Izx], [Ixy,Iyy,Iyz], [Izx,Iyz,Izz] ])
    Iprin,Iaxes = linalg.eig(Itensor)
    return Iprin,Iaxes



# End

