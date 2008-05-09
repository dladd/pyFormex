#!/usr/bin/env python
##
## This file is part of pyFormex 0.7.1 Release Fri May  9 08:39:30 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

"""mesh.py

A plugin providing some useful meshing functions.
"""

from numpy import *
from coords import *
from formex import *
from gui import actors
from simple import line

def createWedgeElements(S1,S2,div=1):
    """Create wedge elements between to triangulated surfaces.
    
    6-node wedge elements are created between two input surfaces (S1 and S2).
    The keyword div determines the number of created wedge element layers.
    Layers with equal thickness are created when an integer value is used for div.
    div can also be specified using a list, that defines the interpolation between the two surfaces.
    Consequently, this can be used to create layers with unequal thickness.
    For example, div=2 gives the same result as [0.,0.5,1.]
    """
    #check which surface lays on top
    n = S1.areaNormals()[1][0]
    if S2.coords[0].distanceFromPlane(S1.coords[0],n) < 0:
        S = S2.copy()
        S2 = S1.copy()
        S1 = S
    #determine the number of layers of wedge elements
    if type(div) == int:
        nlayers = div
    else:
        nlayers = shape(div)[0] - 1
   #create array containing the nodes of the wedge elements
    C1 = S1.coords
    C2 = S2.coords
    coordsWedge = Coords.interpolate(C1,C2,div).reshape(-1,3)
    #create array containing wedge connectivity
    ncoords = C1.shape[0]
    elems = S1.getElems()
    elemsWedge = array([]).astype(int)
    for i in range(nlayers):
        elemsLayer = append(elems,elems+ncoords,1).reshape(-1)
        elemsWedge = append(elemsWedge,elemsLayer,0)
        elems += ncoords
    return coordsWedge,elemsWedge.reshape(-1,6)


def sweepGrid(nodes,elems,path,scale=1.,angle=0.,a1=None,a2=None):
    """ Sweep a quadrilateral mesh along a path
    
    The path should be specified as a (n,2,3) Formex.
    The input grid (quadrilaterals) has to be specified with the nodes and elems and 
    can for example be created with the functions gridRectangle or gridBetween2Curves.
    This quadrilateral grid should be within the YZ-plane.
    The quadrilateral grid can be scaled and/or rotated along the path.
    
    There are three options for the first (a1) / last (a2) element of the path:
    1) None: No corresponding hexahedral elements
    2) 'last': The direction of the first/last element of the path is used to 
    direct the input grid at the start/end of the path
    3) specify a vector: This vector is used to direct the input grid at the start/end of the path
    
    The resulting hexahedral mesh is returned in terms of nodes and elems.
    """
    nodes = Formex(nodes.reshape(-1,1,3))
    n = nodes.shape()[0]
    s = path.shape()[0]
    sc = scale-1.
    a = angle
    
    if a1 != None:
        if a1 == 'last':
            nodes1 = nodes.rotate(actors.rotMatrix(path[0,1]-path[0,0])).translate(path[0,0])
        else:
            nodes1 = nodes.rotate(actors.rotMatrix(a1)).translate(path[0,0])
    else:
        nodes1 = Formex([[[0.,0.,0.]]])
    
    for i in range(s-1):
        r1 = vectorNormalize(path[i+1,1]-path[i+1,0])[1][0]
        r2 = vectorNormalize(path[i,1]-path[i,0])[1][0]
        r = r1+r2
        nodes1 += nodes.rotate(angle,0).scale(scale).rotate(actors.rotMatrix(r)).translate(path[i+1,0])
        scale = scale+sc
        angle = angle+a

    if a2 != None:    
        if a2 == 'last':
            nodes1 += nodes.rotate(angle,0).scale(scale).rotate(actors.rotMatrix(path[s-1,1]-path[s-1,0])).translate(path[s-1,1])
        else:
            nodes1 += nodes.rotate(angle,0).scale(scale).rotate(actors.rotMatrix(a2)).translate(path[s-1,1])
    
    if a1 == None:
        nodes1 = nodes1[1:]
        s = s-1
    if a2 == None:
        s = s-1

    elems0 = elems
    elems1 = append(elems0,elems+n,1)
    elems = elems1
    for i in range(s-1):
        elems = append(elems,elems1+(i+1)*n,0)
    if s == 0:
        elems = array([])
    
    return nodes1[:].reshape(-1,3),elems

### Some useful functions for creating quadrilateral meshes ############################    

def gridRectangle(n1,n2,width,height):
    print """
Function mesh.gridRectangle has been removed!
Its functionality can be obtained by simple.rectangle.
See the SweepBeam example.
"""

def gridBetween2Curves(curve1,curve2,n):
    print """
Function mesh.gridBetween2Curves has been removed!
Its functionality can be obtained by simple.rectangle.
See the SweepBeam example.
"""
