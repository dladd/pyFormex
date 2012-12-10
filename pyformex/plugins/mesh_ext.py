# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

"""Extended functionality of the Mesh class.

This module defines extended Mesh functionality which is considered to be
experimental, maybe incomplete or even buggy.

The functions in this module can be called as functions operating on a
Mesh object, but are also available as Mesh methods.
"""
from __future__ import print_function

from mesh import Mesh
from elements import elementType,_default_facetype
from formex import *
from utils import deprecation, warn


##############################################################################

#
# What is a ring
# What is returned?

def rings(self, sources=0, nrings=-1):
    """_
    It finds rings of elements connected to sources by a node.
    
    Sources can be a single element index (integer) or a list of element indices.
    A list of rings is returned, from zero (ie. the sources) to nrings.
    If nrings is negative (default), all rings are returned.
    """
#
# TODO: this should be made an example
#
    ## Example:
    
    ## S=Sphere(2)
    ## smooth()
    ## draw(S, mode='wireframe')
    ## drawNumbers(S, ontop=True, color='white')
    ## r=S.rings(sources=[2, 6], nrings=2)
    ## print(r)
    ## for i, ri in enumerate(r):
    ##     draw(S.select(ri).setProp(i), mode='smooth')
    
    if nrings == 0:
        return [array(sources)]
    r=self.frontWalk(startat=sources,maxval=nrings-1, frontinc=1)
    ar, rr= arange(len(r)), range(r.max()+1)
    return [ar[r==i] for i in rr ]

def correctNegativeVolumes(self):
    """Modify the connectivity of negative-volume elements to make
    positive-volume elements.

    Negative-volume elements (hex or tet with inconsistent face orientation)
    may appear by error during geometrical trnasformations
    (e.g. reflect, sweep, extrude, revolve).
    This function fixes those elements.
    Currently it only works with linear tet and hex.
    """
    vol=self.volumes()<0.
    if self.eltype.name()=='tet4':
        self.elems[vol]=self.elems[vol][:,  [0, 2, 1, 3]]
    if self.eltype.name()=='hex8':
        self.elems[vol]=self.elems[vol][:,  [4, 5, 6, 7, 0, 1, 2, 3]]
    return self

def scaledJacobian(self, scaled=True):
    """
    Returns a quality measure for volume meshes.
    
    If scaled if False, it returns the Jacobian at the corners of each element.
    If scaled is True, it returns a quality metrics, being
    the minumum value of the scaled Jacobian in each element (at one corner, 
    the Jacobian divided by the volume of a perfect brick).
    Each tet or hex element gives a value between -1 and 1. 
    Acceptable elements have a positive scaled Jacobian. However, good 
    quality requires a minimum of 0.2.
    Quadratic meshes are first converted to linear.
    If the mesh contain mainly negative Jacobians, it probably has negative
    volumes and can be fixed with the  correctNegativeVolumes.
    """
    ne = self.nelems()
    if self.eltype.name()=='hex20':
        self = self.convert('hex8')
    if self.eltype.name()=='tet10':
        self = self.convert('tet4')      
    if self.eltype.name()=='tet4':
        iacre=array([
        [[0, 1], [1, 2],[2, 0],[3, 2]],
        [[0, 2], [1, 0],[2, 1],[3, 1]],
        [[0, 3], [1, 3],[2, 3],[3, 0]],
        ], dtype=int)
        nc = 4
    if self.eltype.name()=='hex8':
        iacre=array([
        [[0, 4], [1, 5],[2, 6],[3, 7], [4, 7], [5, 4],[6, 5],[7, 6]],
        [[0, 1], [1, 2],[2, 3],[3, 0], [4, 5], [5, 6],[6, 7],[7, 4]],
        [[0, 3], [1, 0],[2, 1],[3, 2], [4, 0], [5, 1],[6, 2],[7, 3]],
        ], dtype=int)
        nc = 8
    acre = self.coords[self.elems][:, iacre]
    vacre = acre[:, :,:,1]-acre[:, :,:,0]
    cvacre = concatenate(vacre, axis=1)

    J = vectorTripleProduct(*cvacre).reshape(ne, nc)
    if not scaled: 
        return J
    else:
        normvol = prod(length(cvacre), axis=0).reshape(ne, nc)#volume of 3 nprmal edges
        Jscaled = J/normvol
        return Jscaled.min(axis=1)

# REMOVED in 0.9.0
## THIS NEEDS WORK ###
## surfacetype is also eltype ??
## it does not work (anymore) in many cases
## it makes more sense to just convert to TriSurface and take area
## def areas(self):
##     """_area of elements

##     For surface element the faces' area is returned.
##     For volume elements the sum of the faces'areas is returned.

##     """

##     #In case of quadratic faces, the face's area should be 
##     #the area inside the polygon of face vertices or 
##     #the area of the equivalent linear face?

##     ##this function would require some changes (here proposed inside the function as starting):
##     ##create a _default_surfacetype to create quad8 instead of hex8 ?maybe also a _default_volumetype to create tet4 instead of quad4 ?

##     def defaultSurfacetype(nplex):
##         """Default face type for a surface mesh with given plexitude.

##         For the most common cases of plexitudes, we define a default face
##         type. The full list of default types can be found in
##         mesh._default_facetype.
##         """
##         return _default_surfacetype.get(nplex,None)

##     import geomtools
##     nfacperel= len(self.eltype.faces[1])#nfaces per elem
##     mf=Mesh(self.coords, self.getFaces())#mesh of all faces
##     mf.eltype = elementType(defaultSurfacetype(mf.nplex()))
##     ntriperfac= mf.select([0]).convert('tri3').nelems()#how many tri per face
##     elfacarea = geomtools.areaNormals( mf.convert('tri3').toFormex()[:])[0].reshape(self.nelems(), nfacperel*ntriperfac)#elems'faces'areas
##     return elfacarea.sum(axis=1)#elems'areas


def area(self):
    """_Return the total area of the Mesh.

    For a Mesh with dimensionality 2, the total area of the Mesh is returned.
    For a Mesh with dimensionality 3, the total area of all the element faces
    is returned. Use Mesh.getBorderMesh().area() if you only want the total
    area of the border faces.
    For a Mesh with dimensionality < 2, 0 is returned.
    """
    try:
        return self.areas().sum()
    except:
        return 0.0


def partitionByAngle(self,**arg):
    """Partition a surface Mesh by the angle between adjacent elements.

    The Mesh is partitioned in parts bounded by the sharp edges in the
    surface. The arguments and return value are the same as in
    :meth:`TriSurface.partitionByAngle`.

    Currently this only works for 'tri3' and 'quad4' type Meshes.
    Also, the 'quad4' partitioning method currently only works correctly
    if the quads are nearly planar.
    """
    from plugins.trisurface import TriSurface
    if self.eltype.name() not in [ 'tri3', 'quad4' ]:
        raise ValueError, "partitionByAngle currently only works for 'tri3' and 'quad4' type Meshes."

    S = TriSurface(self.convert('tri3'))
    p = S.partitionByAngle(**arg)
    if self.eltype.name() == 'tri3':
        return p
    if self.eltype.name() == 'quad4':
        p = p.reshape(-1,2)
        if not (p[:,0] == p[:,1]).all():
            warn("The partitioning may be incorrect due to nonplanar 'quad4' elements")
        return p[:,0]


##############################################################################
#
# Initialize
#

def _auto_initialize():
    """Auto-initialize Mesh extensions.

    Calling this function will install some of the mesh functions
    defined in this modules as Mesh methods.
    This function is called when the module is loaded, so the functions
    installed here will always be available as Mesh methods just by
    importing the mesh_ext module.
    """
#    Mesh.areas = areas
    Mesh.area = area
    Mesh.rings = rings
    Mesh.correctNegativeVolumes = correctNegativeVolumes
    Mesh.scaledJacobian = scaledJacobian
    Mesh.partitionByAngle = partitionByAngle
    
_auto_initialize()


# End
