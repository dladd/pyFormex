#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Predefined Formex samples with simple geometric shapes."""

from formex import *

# A collection of simple line element shapes, to be constructed by passing
# the string to the formex.pattern() function.
# The shape() function below returns the corresponding Formex.
Pattern = {
    'line'   : '1',
    'angle'  : '102',
    'square' : '1234',
    'plus'   : '1020304',
    'cross'  : '5060708',
    'diamond' : '5678',
    'rtriangle' : '164',
    'cube'   : '1234i/Ai/Bi/Ci/41234',
    'star'   : '102030405060708',
    'star3d' : '1/02/03/04/05/06/07/08/0A/0B/0C/0D/0E/0F/0G/0H/0a/0b/0c/0d/0e/0f/0g/0h'
    }


def regularGrid(x0,x1,nx):
    """Create a regular grid between points x0 and x1.

    x0 and x1 are n-dimensional points (usually 1D, 2D or 3D).
    The space between x0 and x1 is divided in nx equal parts. nx should have
    the same dimension as x0 and x1.
    The result is a rectangular grid of coordinates in an array with
    shape ( nx[0]+1, nx[1]+1, ..., n ).
    """
    x0 = asarray(x0).ravel()
    x1 = asarray(x1).ravel()
    nx = asarray(nx).ravel()
    if x0.size != x1.size or nx.size != x0.size:
        raise ValueError,"Expected equally sized 1D arrays x0,x1,nx"
    if any(nx < 0):
        raise ValueError,"nx values should be >= 0"
    n = x0.size
    ind = indices(nx+1).reshape((n,-1))
    shape = append(tuple(nx+1),n)
    nx[nx==0] = 1
    jnd = nx.reshape((n,-1)) - ind
    ind = ind.transpose()
    jnd = jnd.transpose()
    return ( (x0*jnd + x1*ind) / nx ).reshape(shape)


def shape(name):
    """Return a Formex with one of the predefined named shapes.

    This is a convenience function returning a plex-2 Formex constructed
    from one of the patterns defined in the simple.Pattern dictionary.
    """
    return Formex(pattern(Pattern[name]))
    

def circle(a1=1.,a2=2.,a3=360.):
    """Return a Formex which is a unit circle at the origin in the x-y-plane.

    a1: dash angle in degrees, a2: modular angle in degrees, a3: total angle.
    a1 == a2 gives a full circle, a1 < a2 gives a dashed circle.
    If a3 < 360, the result is an arc.
    The default values give a dashed circle.
    Large angle values result in polygones. Thus circle(120.,120.) is an
    equilateral triangle.
    """
    n = int(round(a3/a2))
    a1 *= pi/180.
    return Formex([[[1.,0.,0.],[cos(a1),sin(a1),0.]]]).rosette(n,a2,axis=2,point=[0.,0.,0.])


def triangle():
    """An equilateral triangle with base [0,1] on axis 0"""
    return Formex([[[0.,0.,0.],[1.,0.,0.],[0.5,0.5*sqrt(3.),0.]]])


class Sphere2(Formex):
    """A sphere consisting of line elements.

    The sphere is modeled by a regular grid of nx longitude circles,
    ny latitude circles and their diagonals.
    """

    def __init__(self,nx,ny,r=1,bot=-90,top=90):
        """Construct a new Sphere2 object.

        A sphere with radius r is modeled by a regular grid of nx
        longitude circles, ny latitude circles and their diagonals.
        
        The 3 sets of lines can be distinguished by their property number:
        1: diagonals, 2: meridionals, 3: horizontals.
        
        The sphere caps can be cut off by specifying top and bottom latitude
        angles (measured in degrees from 0 at north pole to 180 at south pole.
        """
        base = Formex(pattern("543"),[1,2,3])     # single cell
        d = base.select([0]).replic2(nx,ny,1,1)   # all diagonals
        m = base.select([1]).replic2(nx,ny,1,1)   # all meridionals
        h = base.select([2]).replic2(nx,ny+1,1,1) # all horizontals
        grid = m+d+h
        s = float(top-bot) / ny
        F = grid.translate([0,bot/s,1]).spherical(scale=[360./nx,s,r])
        Formex.__init__(self,F.f,F.p)

        
class Sphere3(Formex):
    """A sphere consisting of surface triangles.

    The sphere is modeled by the triangles formed by a regular grid of
    nx longitude circles, ny latitude circles and their diagonals.
    """

    def __init__(self,nx,ny,r=1,bot=-90,top=90):
        """Construct a new Sphere3 object.

        A sphere with radius r is modeled by the triangles fromed by a regular
        grid of nx longitude circles, ny latitude circles and their diagonals.

        The two sets of triangles can be distinguished by their property number:
        1: horizontal at the bottom, 2: horizontal at the top.

        The sphere caps can be cut off by specifying top and bottom latitude
        angles (measured in degrees from 0 at north pole to 180 at south pole.
        """
        base = Formex( [[[0,0,0],[1,0,0],[1,1,0]],
                        [[1,1,0],[0,1,0],[0,0,0]]],
                       [1,2])
        grid = base.replic2(nx,ny,1,1)
        s = float(top-bot) / ny
        F = grid.translate([0,bot/s,1]).spherical(scale=[360./nx,s,r])
        Formex.__init__(self,F.f,F.p)
