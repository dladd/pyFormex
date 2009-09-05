# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""mesh.py

Definition of the Mesh class for describing discrete geometrical models.
And some useful meshing functions to create such models.
"""

from numpy import *
from coords import *
from formex import *
from connectivity import *
import elements
from plugins.fe import mergeModels
from utils import deprecation

# Should be made a Coords method
def sweepCoords(self,path,origin=[0.,0.,0.],normal=0,avgdir=False,enddir=None):
    """ Sweep a Coords object along a path, returning a series of copies.

    origin and normal define the local path position and direction on the mesh.
    
    At each point of the curve, a copy of the Coords object is created, with
    its origin in the curve's point, and its normal along the curve's direction.
    In case of a PolyLine, directions are pointing to the next point by default.
    If avgdir==True, average directions are taken at the intermediate points.
    Missing end directions can explicitely be set by enddir, and are by default
    taken along the last segment.
    If the curve is closed, endpoints are treated as any intermediate point,
    and the user should normally not specify enddir. 

    The return value is a sequence of the transformed Coords objects.
    """
    points = path.coords
    if avgdir:
        directions = path.avgDirections()
    else:
         directions = path.directions()

    missing = points.shape[0] - directions.shape[0]
    if missing == 1:
        lastdir = (points[-1] - points[-2]).reshape(1,3)
        directions = concatenate([directions,lastdir],axis=0)
    elif missing == 2:
        lastdir = (points[-1] - points[-2]).reshape(1,3)
        firstdir = (points[1] - points[0]).reshape(1,3)
        directions = concatenate([firstdir,directions,lastdir],axis=0)

    if enddir:
        for i,j in enumerate([0,-1]):
            if enddir[i]:
                directions[j] = Coords(enddir[i])

    directions = normalize(directions)

    if type(normal) is int:
        normal = unitVector(normal)
    normal = Coords(resize(normal,directions.shape))
    normal = normalize(normal)
    angles,normals = vectorPairAreaNormals(directions,normal)
    w = where(angles==0.0)[0]
    normals[w] = [0.,0.,1.]
    angles = arcsin(angles)/rad
    
    base = self.translate(-Coords(origin))
    sequence = [ base.rotate(a,v).translate(p) for a,v,p in zip(angles,normals,points) ]

    return sequence


class Mesh(object):
    """A mesh is a discrete geometrical model consisting of nodes and elements.

    In the Mesh geometrical data model, coordinates of all points are gathered
    in a single twodimensional array 'coords' with shape (ncoords,3) and the
    individual geometrical elements are described by indices into the 'coords'
    array.
    This model has some advantages over the Formex data model, where all
    points of all element are stored by their coordinates:
    
    - compacter storage, because coordinates of coinciding points do not
      need to be repeated,
    - faster connectivity related algorithms.
    
    The downside is that geometry generating algorithms are far more complex
    and possibly slower.
    
    In pyFormex we therefore mostly use the Formex data model when creating
    geometry, but when we come to the point of exporting the geometry to
    file (and to other programs), a Mesh data model may be more adequate.

    The Mesh data model has at least the following attributes:
    
    - coords: (ncoords,3) shaped Coords array,
    - elems: (nelems,nplex) shaped array of int32 indices into coords. All
      values should be in the range 0 <= value < ncoords.
    - prop: array of element property numbers, default None.
    - eltype: string designing the element type, default None.
    """
    
    def __init__(self,coords=None,elems=None,prop=None,eltype=None):
        """Create a new Mesh from the specified data.

        data is either a tuple of (coords,elems) arrays, or an object having
        a 'toMesh()' method, which should return such a tuple.
        """
        self.coords = None
        self.elems = None
        self.prop = prop
        self.eltype = eltype

        if elems is None:
            if hasattr(coords,'toMesh'):
                # initialize from a single object
                coords,elems = coords.toMesh()
            elif type(coords) is tuple:
                coords,elems = coords

        try:
            self.coords = Coords(coords)
            self.elems = Connectivity(elems)
            if coords.ndim != 2 or coords.shape[-1] != 3 or elems.ndim != 2 or \
                   elems.max() >= coords.shape[0] or elems.min() < 0:
                raise ValueError,"Invalid mesh data"

        except:
            raise ValueError,"Invalid initialization data"


    def copy(self):
        """Return a copy using the same data arrays"""
        return Mesh(self.coords,self.elems,self.prop,self.eltype)


    def toFormex(self):
        """Convert a Mesh to a Formex.

        The Formex inherits the element property numbers and eltype from
        the Mesh. Node property numbers however can not be translated to
        the Formex data model.
        """
        return Formex(self.coords[self.elems],self.prop,eltype=self.eltype)


    def data(self):
        """Return the mesh data as a tuple (coords,elems)"""
        return self.coords,self.elems

    def nelems(self):
        return self.elems.shape[0]
    def nplex(self):
        return self.elems.shape[1]
    def ncoords(self):
        return self.coords.shape[0]
    npoints = ncoords
    def shape(self):
        return self.elems.shape
    def bbox(self):
        return self.coords.bbox()


    def report(self):
        bb = self.bbox()
        return """
Shape: %s nodes, %s elems, plexitude %s
BBox: %s, %s
Size: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),bb[1],bb[0],bb[1]-bb[0])

    def compact(self):
        """Renumber the mesh and remove unconnected nodes."""
        nodes = unique1d(self.elems)
        if nodes[-1] >= nodes.size:
            coords = self.coords[nodes]
            elems = reverseUniqueIndex(nodes)[self.elems]
            return Mesh(coords,elems,eltype=self.eltype)
        else:
            return self


    def extrude(self,n,step=1.,dir=0,autofix=True):
        """Extrude a Mesh in one of the axes directions.

        Returns a new Mesh obtained by extruding the given Mesh
        over n steps of length step in direction of axis dir.
        The returned Mesh has double plexitude of the original.

        This function is usually used to extrude points into lines,
        lines into surfaces and surfaces into volumes.
        By default it will try to fix the connectivity ordering where
        appropriate. If autofix is switched off, the connectivities
        are merely stacked, and the user may have to fix it himself.

        Currently, this function correctly transforms: point1 to line2,
        line2 to quad4, tri3 to wedge6, quad4 to hex8.
        """
        nplex = self.nplex()
        coord2 = self.coords.translate(dir,n*step)
        M = connectMesh(self,Mesh(coord2,self.elems),n)

        if autofix and nplex == 2:
            # fix node ordering for line2 to quad4 extrusions
            M.elems[:,-nplex:] = M.elems[:,-1:-(nplex+1):-1].copy()

        if autofix:
            M.eltype = { 6:'wedge6', 8:'hex8' }.get(M.nplex(),None)
        else:
            M.eltype = None

        return M


    def sweep(self,path,eltype=None,**kargs):
        """Sweep a mesh along a path, creating an extrusion"""
        seq = sweepCoords(self.coords,path,**kargs)
        ML = [ Mesh(x,self.elems) for x in seq ]
        return connectMeshSequence(ML,eltype=eltype)


    def convert(self,fromtype,totype):
        """Convert a mesh from element type fromtype to type totype.

        Currently defined conversions:
        'quad4' -> 'tri3'
        """
        fromtype = fromtype.capitalize()
        totype = totype.capitalize()
        try:
            conv = getattr(elements,fromtype).conversion[totype]
        except:
            raise ValueError,"Don't know how to convert from '%s' to '%s'" % (fromtype,totype)

        elems = self.elems[:,conv].reshape(-1,len(conv[0]))
        print elems.shape
        return Mesh(self.coords,elems)


    @classmethod
    def concatenate(clas,ML):
        """Concatenate a list of meshes of the same plexitude and eltype"""
        if len(set([ m.nplex() for m in ML ])) > 1 or len(set([ m.eltype for m in ML ])) > 1:
            raise ValueError,"Meshes are not of same type/plexitude"

        coords,elems = mergeModels([(m.coords,m.elems) for m in ML])
        elems = concatenate(elems,axis=0)
        return Mesh(coords,elems,eltype=ML[0].eltype)
        

def connectMesh(mesh1,mesh2,n=1,n1=None,n2=None,eltype=None):
    """Connect two meshes to form a hypermesh.
    
    mesh1 and mesh2 are two meshes with same topology (shape). 
    The two meshes are connected by a higher order mesh with n
    elements in the direction between the two meshes.
    n1 and n2 are node selection indices permitting a permutation of the
    nodes of the base sets in their appearance in the hypermesh.
    This can e.g. be used to achieve circular numbering of the hypermesh.
    """
    # For compatibility, allow meshes to be specified as tuples
    if type(mesh1) is tuple:
        mesh1 = Mesh(mesh1)
    if type(mesh2) is tuple:
        mesh2 = Mesh(mesh2)

    if mesh1.shape() != mesh2.shape():
        raise ValueError,"Meshes are not compatible"

    # compact the node numbering schemes
    mesh1 = mesh1.copy().compact()
    mesh2 = mesh2.copy().compact()

    # Create the interpolations of the coordinates
    x = Coords.interpolate(mesh1.coords,mesh2.coords,n).reshape(-1,3)

    nnod = mesh1.ncoords()
    nplex = mesh1.nplex()
    if n1 is None:
        n1 = range(nplex)
    if n2 is None:
        n2 = range(nplex)
    e1 = mesh1.elems[:,n1]
    e2 = mesh2.elems[:,n2] + nnod
    et = concatenate([e1,e2],axis=-1)
    e = concatenate([et+i*nnod for i in range(n)])
    return Mesh(x,e,eltype=eltype)


def connectMeshSequence(ML,loop=False,**kargs):
    MR = ML[1:]
    if loop:
        MR.append(ML[0])
    else:
        ML = ML[:-1]
    print [ type(Mi) for Mi in ML ]
    HM = [ connectMesh(Mi,Mj,**kargs) for Mi,Mj in zip (ML,MR) ]
    print [Mi.eltype for Mi in HM]
    return Mesh.concatenate(HM)


########### Deprecated #####################

@deprecation("\nUse mesh.connectMesh instead.")
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


@deprecation("\nUse mesh.sweepMesh instead.")
def sweepGrid(nodes,elems,path,scale=1.,angle=0.,a1=None,a2=None):
    """ Sweep a quadrilateral mesh along a path
    
    The path should be specified as a (n,2,3) Formex.
    The input grid (quadrilaterals) has to be specified with the nodes and
    elems and can for example be created with the functions gridRectangle or
    gridBetween2Curves.
    This quadrilateral grid should be within the YZ-plane.
    The quadrilateral grid can be scaled and/or rotated along the path.
    
    There are three options for the first (a1) / last (a2) element of the path:
    
    1) None: No corresponding hexahedral elements
    2) 'last': The direction of the first/last element of the path is used to 
       direct the input grid at the start/end of the path
    3) specify a vector: This vector is used to direct the input grid at the
       start/end of the path
    
    The resulting hexahedral mesh is returned in terms of nodes and elems.
    """
    nodes = Formex(nodes.reshape(-1,1,3))
    n = nodes.shape()[0]
    s = path.shape()[0]
    sc = scale-1.
    a = angle
    
    if a1 != None:
        if a1 == 'last':
            nodes1 = nodes.rotate(rotMatrix(path[0,1]-path[0,0])).translate(path[0,0])
        else:
            nodes1 = nodes.rotate(rotMatrix(a1)).translate(path[0,0])
    else:
        nodes1 = Formex([[[0.,0.,0.]]])
    
    for i in range(s-1):
        r1 = vectorNormalize(path[i+1,1]-path[i+1,0])[1][0]
        r2 = vectorNormalize(path[i,1]-path[i,0])[1][0]
        r = r1+r2
        nodes1 += nodes.rotate(angle,0).scale(scale).rotate(rotMatrix(r)).translate(path[i+1,0])
        scale = scale+sc
        angle = angle+a

    if a2 != None:    
        if a2 == 'last':
            nodes1 += nodes.rotate(angle,0).scale(scale).rotate(rotMatrix(path[s-1,1]-path[s-1,0])).translate(path[s-1,1])
        else:
            nodes1 += nodes.rotate(angle,0).scale(scale).rotate(rotMatrix(a2)).translate(path[s-1,1])
    
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


# End
