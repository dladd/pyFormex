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

"""Isoparametric transformations"""

from formex import *


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


class Isopar(object):
    """A class representing an isoparametric transformation"""

    isodata = {
        'line2' : (1, ('1','x')),
        'line3' : (1, ('1','x','x*x')),
        'line4' : (1, ('1','x','x**2','x**3')),
        'tri3'  : (2, ('1','x','y')),
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


    def __init__(self,eltype,coords,oldcoords):
        """Create an isoparametric transformation.

        type is one of the keys in Isopar.isodata
        coords and oldcoords can be either arrays, Coords or Formex instances,
        but should be of equal shape, and match the number of atoms in the
        specified transformation type
        """
        ndim,atoms = Isopar.isodata[eltype]
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
        self.eltype = eltype
        self.trf = ab


    def transform(self,X):
        """Apply isoparametric transform to a set of coordinates.

        Returns a Coords array with same shape as X
        """
        ndim,atoms = Isopar.isodata[self.eltype]
        X = Coords(X)
        aa = build_matrix(atoms,X.x().ravel(),X.y().ravel(),X.z().ravel())
        xx = reshape(dot(aa,self.trf),X.shape)
        if ndim < 3:
            xx[...,ndim:] += X[...,ndim:]
        return xx


    def transformFormex(self,F):
        """Apply an isoparametric transform to a Formex.

        The result is a topologically equivalent Formex.
        """
        return Formex(self.transform(F.f))


def transformFormex(F,trf):
    return trf.transformFormex(F)
    
Formex.isopar = transformFormex

# End
