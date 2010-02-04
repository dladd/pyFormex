# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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

"""mesh.py

Definition of the Mesh class for describing discrete geometrical models.
And some useful meshing functions to create such models.
"""

from formex import *
from connectivity import Connectivity,reverseUniqueIndex
import elements
from utils import deprecation
#from collections import defaultdict


#################### This first section holds experimental stuff!! #####

# This should probably go to formex or coords module

def vectorRotation(vec1,vec2,upvec=[0.,0.,1.]):
    """Return a rotation matrix for rotating vector vec1 to vec2

    The rotation matrix will be such that the plane of vec2 and the
    rotated upvec will be parallel to the original upvec.

    This function is like :func:`arraytools.rotMatrix`, but allows the
    specification of vec1.
    The returned matrix should be used in postmultiplication to the Coords.
    """
    u = normalize(vec1)
    u1 = normalize(vec2)
    w = normalize(upvec)
    v = normalize(cross(w,u))
    w = normalize(cross(u,v))
    v1 = normalize(cross(w,u1))
    w1 = normalize(cross(u1,v1))
    mat1 = column_stack([u,v,w])
    mat2 = row_stack([u1,v1,w1])
    mat = dot(mat1,mat2)
    return mat


# Should probably be made a Coords method
# But that would make the coords module dependent on a plugin
def sweepCoords(self,path,origin=[0.,0.,0.],normal=0,upvector=2,avgdir=False,enddir=None):
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

    if type(upvector) is int:
        upvector = Coords(unitVector(upvector))

    base = self.translate(-Coords(origin))
    sequence = [ base.rotate(vectorRotation(normal,d,upvector)).translate(p)
                 for d,p in zip(directions,points)
                 ]
        
    return sequence


_default_eltype = {
    1 : 'point',
    2 : 'line2',
    3 : 'tri3',
    4 : 'quad4',
    6 : 'wedge6',
    8 : 'hex8',
    }


def defaultEltype(nplex):
    """Default element type for a mesh with given plexitude.

    """
    return _default_eltype.get(nplex,None)


########################################################################
## Mesh conversions ##
######################

_conversions_ = {
    'tri3': {
        'tri3-4' : [ ('v', 'tri6'), ],
        'tri6'   : [ ('m', [ (0,1), (1,2), (2,0) ]), ],
        'quad4'  : [ ('v', 'tri6'), ],
    },
    'tri6': {
        'tri3'   : [ ('s', [ (0,1,2) ]), ],
        'tri3-4' : [ ('s', [ (0,3,5),(3,1,4),(4,2,5),(3,4,5) ]), ],
        'quad4'  : [ ('m', [ (0,1,2), ]),
                     ('s', [ (0,3,6,5),(1,4,6,3),(2,5,6,4) ]),
                     ],
        },
    'quad4': {
        'tri3'   : 'tri3-u',
        'tri3-r' : [ ('r', ['tri3-u','tri3-d']), ],
        'tri3-u' : [ ('s', [ (0,1,2), (2,3,0) ]), ],
        'tri3-d' : [ ('s', [ (0,1,3), (2,3,1) ]), ],
        'tri3-x' : [ ('m', [ (0,1,2,3) ]),
                     ('s', [ (0,1,4),(1,2,4),(2,3,4),(3,0,4) ]),
                     ],
        'quad4-4': [ ('v', 'quad9'), ],
        'quad8'  : [ ('m', [ (0,1), (1,2), (2,3), (3,0) ]), ],
        'quad9'  : [ ('v', 'quad8'), ],
        },
    'quad8': {
        'quad9'  : [ ('m', [ (4,5,6,7) ]), ],
        },
    'quad9': {
        'quad8'  : [ ('s', [ (0,1,2,3,4,5,6,7) ]), ],
        'quad4'  : [ ('s', [ (0,1,2,3) ]), ],
        'quad4-4': [ ('s', [ (0,4,8,7),(4,1,5,8),(7,8,6,3),(8,5,2,6) ]), ],
        'tri3-d' : [ ('s', [ (0,4,7),(4,1,5),(5,2,6),(6,3,7),
                      (7,4,8),(4,5,8),(5,6,8),(6,7,8) ]), ],
        'tri3-x' : [ ('s', [ (0,4,8),(4,1,8),(1,5,8),(5,2,8),
                      (2,6,8),(6,3,8),(3,7,8),(7,0,8) ]), ],
        },
    }
   

##############################################################

class Mesh(object):
    """A mesh is a discrete geometrical model defined by nodes and elements.

    In the Mesh geometrical data model, coordinates of all points are gathered
    in a single twodimensional array 'coords' with shape (ncoords,3) and the
    individual geometrical elements are described by indices into the 'elems'
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

        if eltype is None:
            self.eltype = defaultEltype(self.nplex())
        else:
            self.eltype = eltype


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
    nnodes = ncoords
    npoints = ncoords
    def shape(self):
        return self.elems.shape
    def bbox(self):
        return self.coords.bbox()
    def center(self):
        return self.coords.center()
    
    def nedges(self):
        """Return the number of edges.

        Currently, the edges are not fused!
        """
        try:
            el = getattr(elements,self.eltype.capitalize())
            return self.nelems() * len(el.edges)
        except:
            return 0

    def centroids(self):
        """Return the centroids of all elements of the Formex.

        The centroid of an element is the point whose coordinates
        are the mean values of all points of the element.
        The return value is a Coords object with nelems points.
        """
        return self.coords[self.elems].mean(axis=1)
        

    def report(self):
        bb = self.bbox()
        return """
Shape: %s nodes, %s elems, plexitude %s
BBox: %s, %s
Size: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),bb[1],bb[0],bb[1]-bb[0])


    def fuse(self,**kargs):
        """Fuse the nodes of a Meshes.

        All nodes that are within the tolerance limits of each other
        are merged into a single node.  

        The merging operation can be tuned by specifying extra arguments
        that will be passed to :meth:`Coords:fuse`.
        """
        coords,index = self.coords.fuse(**kargs)
        return Mesh(coords,index[self.elems],self.prop,self.eltype)
    

    def compact(self):
        """Remove unconnected nodes and renumber the mesh."""
        #print "Compacting mesh with %s nodes and %s elems" % (self.ncoords(),self.nelems())
        #print self.elems
        nodes = unique1d(self.elems)
        #print nodes
        if nodes.shape[0] < self.ncoords() or nodes[-1] >= nodes.size:
            #print "Leaving %s nodes after compaction" % nodes.shape[0]
            coords = self.coords[nodes]
            if nodes[-1] >= nodes.size:
                elems = reverseUniqueIndex(nodes)[self.elems]
            else:
                elems = self.elems
            return Mesh(coords,elems,prop=self.prop,eltype=self.eltype)
        else:
            return self


    def select(self,selected,compact=False):
        """Return a mesh with selected elements from the original.

        - `selected`: an object that can be used as an index in the
          `elems` array, e.g. a list of element numbers.
        The coords are not compacted, unless compact=True is specify
        """
        prop = self.prop
        if prop:
            prop = prop[selected]
        elems = self.elems[selected]
        return Mesh(self.coords,elems,prop,self.eltype)


    def meanNodes(self,nodsel):
        """Create nodes from the existing nodes of a mesh.

        `nodsel` is a local node selector as in :meth:`selectNodes`
        Returns the mean coordinates of the points in the selector. 
        """
        elems = self.elems.selectNodes(nodsel)
        return self.coords[elems].mean(axis=1)


    def addNodes(self,newcoords,eltype=None):
        """Add new nodes to elements.

        `newcoords` is an `(nelems,nnod,3)` array of coordinates.
        Each element thus gets exactly `nnod` extra points and the result
        is a Mesh with plexitude self.nplex() + nnod.
        """
        newnodes = arange(newcoords.shape[0]).reshape(self.elems.shape[0],-1) + self.coords.shape[0]
        elems = Connectivity(concatenate([self.elems,newnodes],axis=-1))
        coords = Coords.concatenate([self.coords,newcoords])
        return Mesh(coords,elems,self.prop,eltype)


    def addMeanNodes(self,nodsel,eltype=None):
        """Add new nodes to elements by averaging existing ones.

        `nodsel` is a local node selector as in :meth:`selectNodes`
        Returns a Mesh where the mean coordinates of the points in the
        selector are added to each element, thus increasing the plexitude
        by the length of the items in the selector.
        The new element type should be set to correct value.
        """
        newcoords = self.meanNodes(nodsel)
        return self.addNodes(newcoords,eltype)


    def selectNodes(self,nodsel,eltype):
        """Return a mesh with subsets of the original nodes.

        `nodsel` is an object that can be converted to a 1-dim or 2-dim
        array. Examples are a tuple of local node numbers, or a list
        of such tuples all having the same length.
        Each row of `nodsel` holds a list of local node numbers that
        should be retained in the new connectivity table.
        """
        elems = self.elems.selectNodes(nodsel)
        prop = self.prop
        if prop is not None:
            prop = column_stack([prop]*len(nodsel)).reshape(-1)
        return Mesh(self.coords,elems,prop=prop,eltype=eltype)   
    

    def convert(self,totype):
        fromtype = self.eltype

        strategy = _conversions_[fromtype].get(totype,None)

        while not type(strategy) is list:
            # This allows for aliases in the conversion database
            strategy = _conversions_[fromtype].get(strategy,None)
            if strategy is None:
                raise ValueError,"Don't know how to convert %s -> %s" % (fromtype,totype)

        # 'r' and 'v' steps can only be the first and only step
        steptype,stepdata = strategy[0]
        if steptype == 'r':
            # Randomly convert elements to one of the types in list
            return self.convertRandom(stepdata)
        elif steptype == 'v':
            return self.convert(stepdata).convert(totype)

        # Execute a strategy
        mesh = self
        totype = totype.split('-')[0]
        for step in strategy:
            #print "STEP: %s" % str(step)
            steptype,stepdata = step

            if steptype == 'm':
                mesh = mesh.addMeanNodes(stepdata,totype)
                
            elif steptype == 's':
                mesh = mesh.selectNodes(stepdata,totype)

            else:
                raise ValueError,"Unknown conversion step type '%s'" % steptype

        return mesh


    def randomSplit(self,n):
        """Split a mesh in n parts, distributing the elements randomly."""
        sel = random.randint(0,n,(self.nelems()))
        return [ self.select(sel==i) for i in range(n) if i in sel ]


    def convertRandom(self,choices):
        """Convert choosing randomly between choices"""
        ml = self.randomSplit(len(choices))
        ml = [ m.convert(c) for m,c in zip(ml,choices) ]
        prop = self.prop
        if prop:
            prop = concatenate([m.prop for m in ml])
        elems = concatenate([m.elems for m in ml],axis=0)
        eltype = set([m.eltype for m in ml])
        if len(eltype) > 1:
            raise RuntimeError,"Invalid choices for random conversions"
        eltype = eltype.pop()
        return Mesh(self.coords,elems,prop,eltype)
 

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
            M.eltype = defaultEltype(M.nplex())

        return M


    def sweep(self,path,autofix=True,**kargs):
        """Sweep a mesh along a path, creating an extrusion

        Returns a new Mesh obtained by sweeping the given Mesh
        over a path.
        The returned Mesh has double plexitude of the original.
        The operation is similar to the extrude() method, but the path
        can be any 3D curve.
        
        This function is usually used to extrude points into lines,
        lines into surfaces and surfaces into volumes.
        By default it will try to fix the connectivity ordering where
        appropriate. If autofix is switched off, the connectivities
        are merely stacked, and the user may have to fix it himself.

        Currently, this function correctly transforms: point1 to line2,
        line2 to quad4, tri3 to wedge6, quad4 to hex8.
        """
        nplex = self.nplex()
        seq = sweepCoords(self.coords,path,**kargs)
        ML = [ Mesh(x,self.elems) for x in seq ]
        M = connectMeshSequence(ML)
        #print M

        if autofix and nplex == 2:
            # fix node ordering for line2 to quad4 extrusions
            M.elems[:,-nplex:] = M.elems[:,-1:-(nplex+1):-1].copy()

        if autofix:
            M.eltype = defaultEltype(M.nplex())

        return M


    @classmethod
    def concatenate(clas,meshes,**kargs):
        """Concatenate a list of meshes of the same plexitude and eltype

        Merging of the nodes can be tuned by specifying extra arguments
        that will be passed to :meth:`Coords:fuse`.
        """
        nplex = set([ m.nplex() for m in meshes ])
        if len(nplex) > 1:
            raise ValueError,"Cannot concatenate meshes with different plexitude: %s" % str(nplex)
        eltype = set([ m.eltype for m in meshes if m.eltype is not None ])
        if len(eltype) > 1:
            raise ValueError,"Cannot concatenate meshes with different eltype: %s" % str(eltype)
        if len(eltype) == 1:
            eltype = eltype.pop()
        else:
            eltype = None
            
        prop = [m.prop for m in meshes]
        if None in prop:
            prop = None
        else:
            prop = concatenate(prop)
            
        coords,elems = mergeMeshes(meshes,**kargs)
        elems = concatenate(elems,axis=0)
        #print coords,elems,prop,eltype
        return Mesh(coords,elems,prop,eltype)
 
########### Functions #####################


def mergeNodes(nodes,**kargs):
    """Merge all the nodes of a list of node sets.

    Each item in nodes is a Coords array.
    The return value is a tuple with:
    
    - the coordinates of all unique nodes,
    - a list of indices translating the old node numbers to the new.

    The merging operation can be tuned by specifying extra arguments
    that will be passed to :meth:`Coords:fuse`.
    """
    coords = Coords(concatenate([x for x in nodes],axis=0))
    coords,index = coords.fuse(**kargs)
    n = array([0] + [ x.npoints() for x in nodes ]).cumsum()
    ind = [ index[f:t] for f,t in zip(n[:-1],n[1:]) ]
    return coords,ind


def mergeMeshes(meshes,**kargs):
    """Merge all the nodes of a list of Meshes.

    Each item in meshes is a Mesh instance.
    The return value is a tuple with:

    - the coordinates of all unique nodes,
    - a list of elems corresponding to the input list,
      but with numbers referring to the new coordinates.

    The merging operation can be tuned by specifying extra arguments
    that will be passed to :meth:`Coords:fuse`.
    """
    coords = [ m.coords for m in meshes ]
    elems = [ m.elems for m in meshes ]
    coords,index = mergeNodes(coords,**kargs)
    return coords,[Connectivity(i[e]) for i,e in zip(index,elems)]


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
    #print([Mi.eltype for Mi in ML])
    MR = ML[1:]
    if loop:
        MR.append(ML[0])
    else:
        ML = ML[:-1]
    HM = [ connectMesh(Mi,Mj,**kargs) for Mi,Mj in zip (ML,MR) ]
    #print([Mi.eltype for Mi in HM])
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
