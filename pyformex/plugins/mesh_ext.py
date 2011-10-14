# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

from mesh import Mesh
from elements import elementType,_default_facetype
from formex import *
from utils import deprecation


##############################################################################
#
# These functions replace the standard Mesh.report method.
# They are here for demonstration purposes.
#
def report(self):
    """Create a report on the Mesh shape and size.

    The report contains the number of nodes, number of elements,
    plexitude, element type, bbox and size.
    """
    bb = self.bbox()
    return """
    
Shape: %s nodes, %s elems, plexitude %s
Eltype: %s (ndim=%s),
BBox: %s, %s
Size: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),self.eltype,self.eltype.ndim,bb[0],bb[1],bb[1]-bb[0])


def alt_report(self):
    """Create a report on the Mesh shape and size.

    The report contains the number of nodes, number of elements,
    plexitude, element type, bbox and size.
    """
    bb = self.bbox()
    return """
    
Number of nodes: %s
Number of elements: %s
Plexitude: %s
Eltype: %s
Dimensionality: %s
Min Coords: %s
Max Coords: %s
Size: %s
Area: %s
Volume: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),self.eltype,self.eltype.ndim,bb[1],bb[0],bb[1]-bb[0],self.area(),self.volume())

##############################################################################
#

#GDS connectivity functions valid for all mesh types, moved from trisurface.py
def nodeFront(self,startat=0,front_increment=1):
    """Generator function returning the frontal elements.

    startat is an element number or list of numbers of the starting front.
    On first call, this function returns the starting front.
    Each next() call returns the next front.
    """
    p = -ones((self.nelems()),dtype=int)
    if self.nelems() <= 0:
        return
    # Construct table of elements connected to each element
    adj = self.nodeAdjacency()

    # Remember nodes left for processing
    todo = ones((self.npoints(),),dtype=bool)
    elems = clip(asarray(startat),0,self.nelems())
    prop = 0
    while elems.size > 0:
        # Store prop value for current elems
        p[elems] = prop
        yield p

        prop += front_increment

        # Determine adjacent elements
        elems = unique(adj[elems])
        elems = elems[(elems >= 0) * (p[elems] < 0) ]
        if elems.size > 0:
            continue

        # No more elements in this part: start a new one
        elems = where(p<0)[0]
        if elems.size > 0:
            # Start a new part
            elems = elems[[0]]
            prop += 1

def walkNodeFront(self,startat=0,nsteps=-1,front_increment=1):
    for p in self.nodeFront(startat=startat,front_increment=front_increment):   
        if nsteps > 0:
            nsteps -= 1
        elif nsteps == 0:
            break
    return p

def partitionByNodeFront(self,firstprop=0,startat=0):
    """Detects different parts of the Mesh using a frontal method.

    okedges flags the edges where the two adjacent elems are to be
    in the same part of the Mesh.
    startat is a list of elements that are in the first part.
    The partitioning is returned as a property type array having a value
    corresponding to the part number. The lowest property number will be
    firstprop.
    """
    return firstprop +self.walkNodeFront( startat=startat,front_increment=0)

def partitionByConnection(self):
    """Detect the connected parts of a Mesh.

    The Mesh is partitioned in parts in which all elements are
    connected. Two elements are connected if it is possible to draw a
    continuous (poly)line from a point in one element to a point in
    the other element without leaving the Mesh.
    The partitioning is returned as a property type array having a value
    corresponding to the part number. The lowest property number will be
    firstprop.
    """
    return self.partitionByNodeFront()

def splitByConnection(self):
    """Split the Mesh into connected parts.

    Returns a list of Meshes that each form a connected part.
    """
    split = self.setProp(self.partitionByConnection()).splitProp()
    if split:
        return split.values()
    else:
        return [ self ]

def largestByConnection(self):
    """Return the largest connected part of the Mesh."""
    p = self.partitionByConnection()
    nparts = p.max()+1
    if nparts == 1:
        return self,nparts
    else:
        t = [ p == pi for pi in range(nparts) ]
        n = [ ti.sum() for ti in t ]
        w = array(n).argmax()
        return self.clip(t[w]),nparts

########################################

def rings(self, sources, nrings):
    """
    It finds the rings of elems connected to sources by node.
    
    Sources is a list of elem indices.
    A list of rings is returned, from zero (equal to sources) to step.
    If step is -1, all rings are returned.
    """
    r=self.walkNodeFront(startat=sources,nsteps=nrings, front_increment=1)
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
    if self.eltype.name()=='tet4': #Tet4 has Jacobian equal on every corner, but differnt scaled J
        vacre = concatenate(vacre[:, :, 0], axis=1).reshape(-1, 2, 3)
        J = vectorTripleProduct(*vacre).repeat(nc).reshape(ne, 4)
    else: 
        J = vectorTripleProduct(*cvacre).reshape(ne, nc)
    if not scaled: 
        return J
    else:
        normvol = prod(length(cvacre), axis=0).reshape(ne, nc)#volume of 3 nprmal edges
        Jscaled = J/normvol
        return Jscaled.min(axis=1)

## THIS NEEDS WORK ###
## surfacetype is also eltype ??

def areas(self):
    """area of elements

    For surface element the faces' area is returned.
    For volume elements the sum of the faces'areas is returned.

    """

    #In case of quadratic faces, the face's area should be 
    #the area inside the polygon of face vertices or 
    #the area of the equivalent linear face?

    ##this function would require some changes (here proposed inside the function as starting):
    ##create a _default_surfacetype to create quad8 instead of hex8 ?maybe also a _default_volumetype to create tet4 instead of quad4 ?

    def defaultSurfacetype(nplex):
        """Default face type for a surface mesh with given plexitude.

        For the most common cases of plexitudes, we define a default face
        type. The full list of default types can be found in
        mesh._default_facetype.
        """
        return _default_surfacetype.get(nplex,None)
    import geomtools
    nfacperel= len(self.eltype.faces[1])#nfaces per elem
    mf=Mesh(self.coords, self.getFaces())#mesh of all faces
    mf.eltype = elementType(defaultSurfacetype(mf.nplex()))
    ntriperfac= mf.select([0]).convert('tri3').nelems()#how many tri per face
    elfacarea = geomtools.areaNormals( mf.convert('tri3').toFormex()[:])[0].reshape(self.nelems(), nfacperel*ntriperfac)#elems'faces'areas
    return elfacarea.sum(axis=1)#elems'areas


def area(self):
    """Return the total area of the Mesh.

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
    Also, the 'quad4' partitioning method currently only works corectly
    if the quads are nearly planar.
    """
    if self.eltype.name() not in [ 'tri3', 'quad4' ]:
        raise ValueError, "partitionByAngle currently only works for 'tri3' and 'quad4' type Meshes."

    S = TriSurface(self.convert('tri3'))
    p = S.partitionByAngle(**arg)
    if self.eltype.name() == 'tri3':
        return p
    if self.eltype.name() == 'quad4':
        p = p.reshape(-1,2)
        if not (p[:,0] == p[:,1]).all():
            pf.warning("The partitioning may be incorrect due to nonplanar 'quad4' elements")
        return p[:,0]


# BV: This is not mesh specific and can probably be achieved by t1 * t2
@deprecation("Deprecated")
def tests(t):
    """Intersection of multiple test operations.
    
    t is a list of n 1D boolean list, obtained by n Mesh.test operations.
    A new 1D boolean list is returned.
    t1=M.test(nodes=...)
    t2=M.test(nodes=...)
    T=tests( [t1, t2] ) is the intersection if t1 and t2 and can be used for Mesh.clip(T)
    """
    return array(t, int).sum(axis=0)==len(t)


##############################################################################
#
# Initialize
#

def initialize():
    """Initialize the Mesh extensions.

    Calling this function will install some of the mesh functions
    defined in this modules as Mesh methods.
    """
    Mesh.report = alt_report


def _auto_initialize():
    """Auto-initialize Mesh extensions.

    Calling this function will install some of the mesh functions
    defined in this modules as Mesh methods.
    This function is called when the module is loaded, so the functions
    installed here will always be available as Mesh methods just by
    importing the mesh_ext module.
    """
    Mesh.report = report
    Mesh.areas = areas
    Mesh.area = area
    Mesh.nodeFront = nodeFront
    Mesh.walkNodeFront = walkNodeFront
    Mesh.partitionByNodeFront = partitionByNodeFront
    Mesh.partitionByConnection = partitionByConnection
    Mesh.splitByConnection = splitByConnection
    Mesh.largestByConnection = largestByConnection
    Mesh.rings = rings
    Mesh.correctNegativeVolumes = correctNegativeVolumes
    Mesh.scaledJacobian = scaledJacobian
    Mesh.partitionByAngle = partitionByAngle
    
_auto_initialize()


# End
