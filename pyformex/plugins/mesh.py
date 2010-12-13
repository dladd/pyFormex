# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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

"""Finite element meshes in pyFormex.

This module defines the Mesh class, which can be used to describe discrete
geometrical models like those used in Finite Element models.
It also contains some useful functions to create such models.
"""

from formex import *
from connectivity import Connectivity
import elements
from utils import deprecation
from geometry import Geometry
from simple import regularGrid


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
def sweepCoords(self,path,origin=[0.,0.,0.],normal=0,upvector=2,avgdir=False,enddir=None,scalex=None,scaley=None):
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
    
    At each point of the curve, the original Coords object can be scaled in x
    and y direction by specifying scalex and scaley. The number of values
    specified in scalex and scaly should be equal to the number of points on
    the curve.

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
        
    if scalex is not None:
        if len(scalex) != points.shape[0]:
            raise ValueError,"The number of scale values in x-direction differs from the number of copies that will be created."
    else:
        scalex = ones(points.shape[0])
        
    if scaley is not None:
        if len(scaley) != points.shape[0]:
            raise ValueError,"The number of scale values in y-direction differs from the number of copies that will be created."
    else:
        scaley = ones(points.shape[0])
    
    base = self.translate(-Coords(origin))
    sequence = [ base.scale([scx,scy,1.]).rotate(vectorRotation(normal,d,upvector)).translate(p)
                 for scx,scy,d,p in zip(scalex,scaley,directions,points)
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

    For the most common cases of plexitudes, we define a default element
    type. The full list of default types can be found in
    plugins.mesh._default_eltype.
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
        'quad8'  : [ ('m', [ (0,1), (1,2), (2,3), (3,0) ]), ],
        'quad4-4': [ ('v', 'quad9'), ],
        'quad9'  : [ ('v', 'quad8'), ],
        },
    'quad8': {
        'tri3'   : [ ('v', 'quad9'), ],
        'tri3-v' : [ ('s', [ (0,4,7),(1,5,4),(2,6,5),(3,7,6),(5,6,4),(7,4,6) ]), ],
        'tri3-h' : [ ('s', [ (0,4,7),(1,5,4),(2,6,5),(3,7,6),(4,5,7),(6,7,5) ]), ],
        'quad4'  : [ ('s', [ (0,1,2,3) ]), ],
        'quad4-4': [ ('v', 'quad9'), ],
        'quad9'  : [ ('m', [ (4,5,6,7) ]), ],
        },
    'quad9': {
        'quad8'  : [ ('s', [ (0,1,2,3,4,5,6,7) ]), ],
        'quad4'  : [ ('v', 'quad8'), ],
        'quad4-4': [ ('s', [ (0,4,8,7),(4,1,5,8),(7,8,6,3),(8,5,2,6) ]), ],
        'tri3'   : 'tri3-d',
        'tri3-d' : [ ('s', [ (0,4,7),(4,1,5),(5,2,6),(6,3,7),
                      (7,4,8),(4,5,8),(5,6,8),(6,7,8) ]), ],
        'tri3-x' : [ ('s', [ (0,4,8),(4,1,8),(1,5,8),(5,2,8),
                      (2,6,8),(6,3,8),(3,7,8),(7,0,8) ]), ],
        },
    'wedge6': {
        'tet4'  : [ ('s', [ (0,1,2,3),(1,2,3,4),(2,3,4,5) ]), ],
        },
    'hex8': {
        'wedge6': [ ('s', [ (0,1,2,4,5,6),(2,3,0,6,7,4) ]), ],
        'tet4'  : [ ('s', [ (0,1,2,5),(2,3,0,7),(5,7,6,2),(7,5,4,0),(0,5,2,7) ]), ],
        'hex20' : [ ('m', [ (0,1), (1,2), (2,3), (3,0),
                            (4,5), (5,6), (6,7), (7,4),
                            (0,4), (1,5), (2,6), (3,7), ]), ],
        },
    'hex20': {
        'hex8'  : [ ('s', [ (0,1,2,3,4,5,6,7) ]), ],
        'tet4'  : [ ('v', 'hex8'), ],
        },
    }


# degenerate patterns
_reductions_ = {
    'hex8': {
        'wedge6' : [
            ([[0,1],[4,5]], [0,2,3,4,6,7]),
            ([[1,2],[5,6]], [0,1,3,4,5,7]),
            ([[2,3],[6,7]], [0,1,2,4,5,6]),
            ([[3,0],[7,4]], [0,1,2,4,5,6]),
            ([[0,1],[3,2]], [0,4,5,3,7,6]),
            ([[1,5],[2,6]], [0,4,5,3,7,6]),
            ([[5,4],[6,7]], [0,4,1,3,7,2]),
            ([[4,0],[7,3]], [0,5,1,3,6,2]),
            ([[0,3],[1,2]], [0,7,4,1,6,5]),
            ([[3,7],[2,6]], [0,3,4,1,2,5]),
            ([[7,4],[6,5]], [0,3,4,1,2,5]),
            ([[4,0],[5,1]], [0,3,7,1,2,6]),
            ],
        },
    }
   

##############################################################

class Mesh(Geometry):
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
    
    If eltype is None, a default eltype is derived from the plexitude, by
    calling the defaultEltype function. For plexitudes without default type,
    or if the default type is not the wanted element type, the user should
    specify the element type himself.

    A Mesh can be initialized by its attributes (coords,elems,prop,eltype)
    or by a single geometric object that provides a toMesh() method.
    """
    ###################################################################
    ## DEVELOPERS: ATTENTION
    ##
    ## Because the TriSurface is derived from Mesh, all methods which
    ## return a Mesh and will also work correctly on a TriSurface,
    ## should use self.__class__ to return the proper class, and they 
    ## should specify the prop and eltype arguments using keywords
    ## (because only the first two arguments match).
    ## See the copy() method for an example.
    ###################################################################
    
    def __init__(self,coords=None,elems=None,prop=None,eltype=None):
        """Initialize a new Mesh."""
        self.coords = self.elems = self.prop = self.eltype = None
        self.ndim = -1
        self.nodes = self.edges = self.faces = self.cells = None

        if coords is None:
            # Create an empty Mesh object
            #print "EMPTY MESH"
            return

        if elems is None:
            if hasattr(coords,'toMesh'):
                # initialize from a single object
                coords,elems = coords.toMesh()
            elif type(coords) is tuple:
                # SHOULD WE KEEP THIS ???
                coords,elems = coords

        try:
            self.coords = Coords(coords)
            if self.coords.ndim != 2:
                raise ValueError,"\nExpected 2D coordinate array, got %s" % self.coords.ndim
            self.elems = Connectivity(elems)
            if self.elems.size > 0 and (
                self.elems.max() >= self.coords.shape[0] or
                self.elems.min() < 0):
                raise ValueError,"\nInvalid connectivity data: some node number(s) not in coords array"
        except:
            raise

        self.setProp(prop)

        if eltype is None:
            self.eltype = defaultEltype(self.nplex())
        else:
            # THis should check that the eltype is known
            self.eltype = eltype


    def _set_coords(self,coords):
        """Replace the current coords with new ones.

        Returns a Mesh or subclass exactly like the current except
        for the position of the coordinates.
        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            return self.__class__(coords,self.elems,prop=self.prop,eltype=self.eltype)
        else:
            raise ValueError,"Invalid reinitialization of %s coords" % self.__class__


    def setProp(self,prop=None):
        """Create or destroy the property array for the Mesh.

        A property array is a rank-1 integer array with dimension equal
        to the number of elements in the Mesh.
        You can specify a single value or a list/array of integer values.
        If the number of passed values is less than the number of elements,
        they wil be repeated. If you give more, they will be ignored.
        
        If a value None is given, the properties are removed from the Mesh.
        """
        if prop is None:
            self.prop = None
        else:
            prop = array(prop).astype(Int)
            self.prop = resize(prop,(self.nelems(),))
        return self


    def getProp(self):
        """Return the properties as a numpy array (ndarray)"""
        return self.prop


    def maxProp(self):
        """Return the highest property value used, or None"""
        if self.prop is None:
            return None
        else:
            return self.prop.max()


    def propSet(self):
        """Return a list with unique property values."""
        if self.prop is None:
            return None
        else:
            return unique(self.prop)


    def copy(self):
        """Return a copy using the same data arrays"""
        # SHOULD THIS RETURN A DEEP COPY?
        return self.__class__(self.coords,self.elems,prop=self.prop,eltype=self.eltype)


    def toFormex(self):
        """Convert a Mesh to a Formex.

        The Formex inherits the element property numbers and eltype from
        the Mesh. Node property numbers however can not be translated to
        the Formex data model.
        """
        return Formex(self.coords[self.elems],self.prop,self.eltype)

    
    def ndim(self):
        return 3
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

    def nedges(self):
        """Return the number of edges.

        This returns the number of rows that would be in getEdges(),
        without actually constructing the edges.
        The edges are not fused!
        """
        try:
            el = getattr(elements,self.eltype.capitalize())
            return self.nelems() * len(el.edges)
        except:
            return 0
    


    def centroids(self):
        """Return the centroids of all elements of the Mesh.

        The centroid of an element is the point whose coordinates
        are the mean values of all points of the element.
        The return value is a Coords object with nelems points.
        """
        return self.coords[self.elems].mean(axis=1)


    
    def getCoords(self):
        """Get the coords data.

        Returns the full array of coordinates stored in the Mesh object.
        Note that this may contain points that are not used in the mesh.
        :meth:`compact` will remove the unused points.
        """
        return self.coords

    
    def getElems(self):
        """Get the elems data.

        Returns the element connectivity data as stored in the object.
        """
        return self.elems


    def getLowerEntitiesSelector(self,level=-1,unique=False):
        """Get the entities of a lower dimensionality.

        If the element type is defined in the :mod:`elements` module,
        this returns a Connectivity table with the entities of a lower
        dimensionality. The full list of entities with increasing
        dimensionality  0,1,2,3 is::

            ['points', 'edges', 'faces', 'cells' ]

        If level is negative, the dimensionality returned is relative
        to that of the caller. If it is positive, it is taken absolute.
        Thus, for a Mesh with a 3D element type, getLowerEntities(-1)
        returns the faces, while for a 2D element type, it returns the edges.
        For bothe meshes however,  getLowerEntities(+1) returns the edges.

        By default, all entities for all elements are returned and common
        entities will appear multiple times. Specifying unique=True will 
        return only the unique ones.

        The return value may be an empty table, if the element type does
        not have the requested entities (e.g. the 'point' type).
        If the eltype is not defined, or the requested entity level is
        outside the range 0..3, the return value is None.
        """
        try:
            el = getattr(elements,self.eltype.capitalize())
        except:
            return None

        if level < 0:
            level = el.ndim + level

        if level < 0 or level > 3:
            return None

        attr = ['points', 'edges', 'faces', 'cells'][level]
        return array(getattr(el,attr))


    def getLowerEntities(self,level=-1,unique=False):
        """Get the entities of a lower dimensionality.

        If the element type is defined in the :mod:`elements` module,
        this returns a Connectivity table with the entities of a lower
        dimensionality. The full list of entities with increasing
        dimensionality  0,1,2,3 is::

            ['points', 'edges', 'faces', 'cells' ]

        If level is negative, the dimensionality returned is relative
        to that of the caller. If it is positive, it is taken absolute.
        Thus, for a Mesh with a 3D element type, getLowerEntities(-1)
        returns the faces, while for a 2D element type, it returns the edges.
        For both meshes however,  getLowerEntities(+1) returns the edges.

        By default, all entities for all elements are returned and common
        entities will appear multiple times. Specifying unique=True will 
        return only the unique ones.

        The return value may be an empty table, if the element type does
        not have the requested entities (e.g. the 'point' type).
        If the eltype is not defined, or the requested entity level is
        outside the range 0..3, the return value is None.
        """
        sel = self.getLowerEntitiesSelector(level)
        ent = self.elems.selectNodes(sel)
        if unique:
            ent = ent.removeDoubles()

        return ent


    def getNodes(self):
        """Return the set of unique node numbers in the Mesh.

        This returns only the node numbers that are effectively used in
        the connectivity table. For a compacted Mesh, it is equal to
        ```arange(self.nelems)```.

        This function also stores the result internally so that future
        requests can return it without the need for computing.
        """
        if self.nodes is None:
            self.nodes  = unique(self.elems)
        return self.nodes


    def getPoints(self):
        """Return the nodal coordinates of the Mesh.

        This returns only those points that are effectively used in
        the connectivity table. For a compacted Mesh, it is equal to
        the coords attribute.
        """
        return self.coords[self.getNodes()]
        

    def getEdges(self):
        """Return the unique edges of all the elements in the Mesh.

        This is a convenient function to create a table with the element
        edges. It is equivalent to ```self.getLowerEntities(1,unique=True)```

        This function also stores the result internally so that future
        requests can return it without the need for computing.
        """
        if self.edges is None:
            self.edges = self.getLowerEntities(1,unique=True)
        return self.edges
        

    def getFaces(self):
        """Return the unique faces of all the elements in the Mesh.

        This is a convenient function to create a table with the element
        faces. It is equivalent to ```self.getLowerEntities(2,unique=True)```

        This function also stores the result internally so that future
        requests can return it without the need for computing.
        """
        from trisurface import TriSurface
        if self.__class__ == TriSurface:
            import warnings
            warnings.warn('warn_trisurface_getfaces')

        if self.faces is None:
            self.faces = self.getLowerEntities(2,unique=True)
        return self.faces
        

    def getCells(self):
        """Return the cells of the elements.

        This is a convenient function to create a table with the element
        cells. It is equivalent to ```self.getLowerEntities(3,unique=True)```

        This function also stores the result internally so that future
        requests can return it without the need for computing.
        """
        if self.cells is None:
            self.cells = self.getLowerEntities(3,unique=True)
        return self.cells
    
    
    ## def getEdges(self):
    ##     """Get the edges data."""
    ##     if self.edges is None:
    ##         self.faces,self.edges = self.elems.untangle()
    ##     return self.edges

    
    def getFaceEdges(self):
        """Get the faces' edge numbers."""
        if self.face_edges is None:
            self.face_edges,self.edges2 = self.elems.untangle()
        return self.face_edges


    def getBorder(self,return_indices=False):
        """Return the border of the Mesh.

        This returns a Connectivity table with the border of the Mesh.
        The border entities are of a lower hierarchical level than the
        mesh itself. These entities become part of the border if they
        are connected to only one element.

        If return_indices==True, it returns also an (nborder,2) index
        for inverse lookup of the higher entity (column 0) and its local
        border part number (column 1).

        The returned Connectivity can be used together with the
        Mesh.coords to construct a Mesh of the border geometry.
        See also :meth:`getBorderMesh`.
        """
        sel = self.getLowerEntitiesSelector(-1)
        hi,lo = self.elems.insertLevel(sel)
        hiinv = hi.inverse()
        ncon = (hiinv>=0).sum(axis=1)
        isbrd = (ncon<=1)   # < 1 should not occur 
        brd = lo[isbrd]
        if not return_indices:
            return brd
        
        # return indices where the border elements come from
        binv = hiinv[isbrd]
        enr = binv[binv >= 0]  # element number
        a = hi[enr]
        b = arange(lo.shape[0])[isbrd].reshape(-1,1)
        fnr = where(a==b)[1]   # local border part number
        return brd,column_stack([enr,fnr])


    def getBorderMesh(self,compact=True):
        """Return a Mesh with the border elements.

        Returns a Mesh representing the border of the Mesh.
        The returned Mesh is of the next lower hierarchical level.
        If the Mesh has property numbers, the border elements inherit
        the property of the element to which they belong.

        By default, the resulting Mesh is compacted. Compaction can be
        switched off by setting `compact=False`.
        """
        if self.prop==None:
            M = Mesh(self.coords,self.getBorder())

        else:
            brd,indices = self.getBorder(return_indices=True)
            enr = indices[:,0]
            M = Mesh(self.coords,brd,prop=self.prop[enr])

        if compact:
            M._compact()
        return M



    def reverse(self):
        """Return a Mesh where all elements have been reversed.

        Reversing an element means reversing the order of its points.
        This is equivalent to::
        
          Mesh(self.coords,self.elems[:,::-1])
          
        """
        return self.__class__(self.coords,self.elems[:,::-1],prop=self.prop,eltype=self.eltype)


#############################################################################    
    # ?? DOES THIS WORK FOR *ANY* MESH ??
    # What with a mesh of points, lines, ...
    def getAngles(self, angle_spec=Deg):
        """Returns the angles in Deg or Rad between the edges of a mesh.
        
        The returned angles are shaped  as (nelems, n1faces, n1vertices),
        where n1faces are the number of faces in 1 element and the number
        of vertices in 1 face.
        """
        mf = self.coords[self.getFaces()]
        el = getattr(elements,self.eltype.capitalize())
        v = mf - roll(mf,-1,axis=1)
        v=normalize(v)
        v1=-roll(v,+1,axis=1)
        angfac= arccos( dotpr(v, v1) )/angle_spec
        return angfac.reshape(self.nelems(),len(el.faces), len(el.faces[0]))

    # BV: This needs clean up
    def neighborsByNode(self, elsel=None):
        """_For each element index in the list elsel,

        it returns the list of neighbor elements (connected by one node at
        least). If elsel is None, the neighbors of all elements are
        calculated, but it is computationally expensive for big meshes.
        """
        if elsel==None:
            elsel=range(self.nelems())
        fnf = self.elems.inverse()#faces touched by node
        fnf = fnf[self.elems[elsel]]#face, nodes belonging to face, faces touched by nodes)
        ff = fnf.reshape(fnf.shape[0], fnf.shape[1]*fnf.shape[2] )#(faces touched faces)
        #add -1 so everyone has at least once -1
        ff = concatenate([ff, -ones([ff.shape[0]  ],  dtype=int).reshape(-1, 1)   ], 1)
        #take unique on each row and remove the -1
        uf = [unique(fi)[1:] for fi in ff]
        #remove the face-i from the neighboors of face-i
        return [ uf[i][uf[i]!=i] for i in range(len(uf)) ]


    def report(self):
        """Create a report on the Mesh shape and size.

        The report contains the number of nodes, number of elements,
        plexitude, bbox and size.
        """
        bb = self.bbox()
        return """
Shape: %s nodes, %s elems, plexitude %s
BBox: %s, %s
Size: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),bb[1],bb[0],bb[1]-bb[0])


    def __str__(self):
        """Format a Mesh in a string.

        This creates a detailed string representation of a Mesh,
        containing the report() and the lists of nodes and elements.
        """
        return self.report() + "Coords:\n" + self.coords.__str__() +  "Elems:\n" + self.elems.__str__()


    def fuse(self,**kargs):
        """Fuse the nodes of a Meshes.

        All nodes that are within the tolerance limits of each other
        are merged into a single node.  

        The merging operation can be tuned by specifying extra arguments
        that will be passed to :meth:`Coords:fuse`.
        """
        coords,index = self.coords.fuse(**kargs)
        return self.__class__(coords,index[self.elems],prop=self.prop,eltype=self.eltype)
    

    # Since this is used in only a few places, we could
    # throw it away and only use compact()
    def _compact(self):
        """Remove unconnected nodes and renumber the mesh.

        Beware! This function changes the object in place and therefore
        returns nothing. It is mostly intended for internal use.
        Normal users should use compact().
        """
        nodes = unique(self.elems)
        if nodes.size == 0:
            self.__init__([],[])
        
        elif nodes.shape[0] < self.ncoords() or nodes[-1] >= nodes.size:
            coords = self.coords[nodes]
            if nodes[-1] >= nodes.size:
                elems = inverseUniqueIndex(nodes)[self.elems]
            else:
                elems = self.elems
            self.__init__(coords,elems,prop=self.prop,eltype=self.eltype)
    

    def compact(self):
        """Remove unconnected nodes and renumber the mesh.

        Returns a mesh where all nodes that are not used in any
        element have been removed, and the nodes are renumbered to
        a compacter scheme.
        """
        nodes = unique(self.elems)
        if nodes.size == 0:
            self.__init__([],[])
        
        elif nodes.shape[0] < self.ncoords() or nodes[-1] >= nodes.size:
            coords = self.coords[nodes]
            if nodes[-1] >= nodes.size:
                elems = inverseUniqueIndex(nodes)[self.elems]
            else:
                elems = self.elems
            self.__class__(coords,elems,prop=self.prop,eltype=self.eltype)

        return self


    def select(self,selected,compact=True):
        """Return a Mesh only holding the selected elements.

        - `selected`: an object that can be used as an index in the
          `elems` array, e.g. a list of (integer) element numbers,
          or a boolean array with the same length as the `elems` array.

        - `compact`: boolean. If True (default), the returned Mesh will be
          compacted, i.e. the unused nodes are removed and the nodes are
          renumbered from zero. If False, returns the node set and numbers
          unchanged. 
          
        Returns a Mesh (or subclass) with only the selected elements.
        
        See `cselect` for the complementary operation.
        """
        if self.__class__ == Mesh:
            import warnings
            warnings.warn('warn_mesh_select_default_compacted')
        M = self.__class__(self.coords,self.elems[selected],eltype=self.eltype)
        if self.prop is not None:
            M.setProp(self.prop[selected])
        if compact:
            M._compact()
        return M


    def cselect(self,selected,compact=True):
        """Return a mesh without the selected elements.

        - `selected`: an object that can be used as an index in the
          `elems` array, e.g. a list of (integer) element numbers,
          or a boolean array with the same length as the `elems` array.

        - `compact`: boolean. If True (default), the returned Mesh will be
          compacted, i.e. the unused nodes are removed and the nodes are
          renumbered from zero. If False, returns the node set and numbers
          unchanged. 
          
        Returns a Mesh with all but the selected elements.
        
        This is the complimentary operation of `select`.
        """
        selected = asarray(selected)
        if selected.dtype==bool:
            return self.select(selected==False,compact=compact)
        else:
            wi = range(self.nelems())
            wi = delete(wi,selected)
            return self.select(wi,compact=compact)


    def meanNodes(self,nodsel):
        """Create nodes from the existing nodes of a mesh.

        `nodsel` is a local node selector as in :meth:`selectNodes`
        Returns the mean coordinates of the points in the selector as
        `(nelems*nnod,3)` array of coordinates, where nnod is the length
        of the node selector. 
        """
        elems = self.elems.selectNodes(nodsel)
        return self.coords[elems].mean(axis=1)


    def addNodes(self,newcoords,eltype=None):
        """Add new nodes to elements.

        `newcoords` is an `(nelems,nnod,3)` or`(nelems*nnod,3)` array of
        coordinates. Each element gets exactly `nnod` extra nodes from this
	array. The result is a Mesh with plexitude `self.nplex() + nnod`.
        """
        newcoords = newcoords.reshape(-1,3)
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

    
    def withProp(self,val):
        """Return a Mesh which holds only the elements with property val.

        val is either a single integer, or a list/array of integers.
        The return value is a Mesh holding all the elements that
        have the property val, resp. one of the values in val.
        The returned Mesh inherits the matching properties.
        
        If the Mesh has no properties, a copy with all elements is returned.
        """
        if self.prop is None:
            return self.__class__(self.coords,self.elems,eltype=self.eltype)
        elif type(val) == int:
            return self.__class__(self.coords,self.elems[self.prop==val],prop=val,eltype=self.eltype)
        else:
            t = zeros(self.prop.shape,dtype=bool)
            for v in asarray(val).flat:
                t += (self.prop == v)
            return self.__class__(self.coords,self.elems[t],prop=self.prop[t],eltype=self.eltype)
        
    
    def withoutProp(self, val):
        """Return a Mesh without the elements with property val.

        This is the complementary method of Mesh.withProp().
        val is either a single integer, or a list/array of integers.
        The return value is a Mesh holding all the elements that do not
        have the property val, resp. one of the values in val.
        The returned Mesh inherits the matching properties.
        
        If the Mesh has no properties, a copy with all elements is returned.
        """
        wi = range(len(self.propSet()))
        wi = delete(wi, val)
        return self.withProp(wi)


    def splitProp(self):
        """Partition a Mesh according to its propery values.

        Returns a dict with the property values as keys and the
        corresponding partitions as values. Each value is a Mesh instance.
        It the Mesh has no props, an empty dict is returned.
        """
        if self.prop is None:
            return {}
        else:
            return dict([(p,self.withProp(p)) for p in self.propSet()])


    def splitRandom(self,n,compact=True):
        """Split a mesh in n parts, distributing the elements randomly.

        Returns a list of n Mesh objects, constituting together the same
        Mesh as the original. The elements are randomly distributed over
        the subMeshes.

        By default, the Meshes are compacted. Compaction may be switched
        off for efficiency reasons.
        """
        sel = random.randint(0,n,(self.nelems()))
        return [ self.select(sel==i,compact=compact) for i in range(n) if i in sel ]
    

    def convert(self,totype):
        """Convert a Mesh to another element type.

        Converting a Mesh from one element type to another can only be
        done if both element types are of the same dimensionality.
        Thus, 3D elements can only be converted to 3D elements.

        The conversion is done by splitting the elements in smaller parts
        and/or by adding new nodes to the elements.

        Not all conversions between elements of the same dimensionality
        are possible. The possible conversion strategies are implemented
        in a table. New strategies may be added however.

        The return value is a Mesh of the requested element type, representing
        the same geometry (possibly approximatively) as the original mesh.
        
        If the requested conversion is not implemented, an error is raised.
        """
        
        fromtype = self.eltype
        if totype == fromtype:
            return self

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


    def convertRandom(self,choices):
        """Convert choosing randomly between choices

        """
        ml = self.splitRandom(len(choices),compact=False)
        ml = [ m.convert(c) for m,c in zip(ml,choices) ]
        prop = self.prop
        if prop is not None:
            prop = concatenate([m.prop for m in ml])
        elems = concatenate([m.elems for m in ml],axis=0)
        eltype = set([m.eltype for m in ml])
        if len(eltype) > 1:
            raise RuntimeError,"Invalid choices for random conversions"
        eltype = eltype.pop()
        return Mesh(self.coords,elems,prop,eltype)


    def reduceDegenerate(self,eltype=None):
        """Reduce degenerate elements to lower plexitude elements.

        This will try to reduce the degenerate elements of the mesh to elements
        of a lower plexitude. If a target element type is given, only the matching
        recuce scheme is tried. Else, all the target element types for which
        a reduce scheme from the Mesh eltype is available, will be tried.

        The result is a list of Meshes of which the last one contains the
        elements that could not be reduced and may be empty.
        Property numbers propagate to the children. 
        """
        strategies = _reductions_.get(self.eltype,{})
        if eltype is not None:
            s = strategies.get(eltype,[])
            if s:
                strategies = {eltype:s}
            else:
                strategies = {}
        if not strategies:
            return [self]

        m = self
        ML = []

        for eltype in strategies:
            #print "REDUCE TO %s" % eltype

            elems = []
            prop = []
            for conditions,selector in strategies[eltype]:
                e = m.elems
                cond = array(conditions)
                #print "TRYING",cond
                #print e
                w = (e[:,cond[:,0]] == e[:,cond[:,1]]).all(axis=1)
                #print "Matching elems: %s" % where(w)[0]
                sel = where(w)[0]
                if len(sel) > 0:
                    elems.append(e[sel][:,selector])
                    if m.prop is not None:
                        prop.append(m.prop[sel])
                    # remove the reduced elems from m
                    m = m.select(~w,compact=False)

                    if m.nelems() == 0:
                        break

            if elems:
                elems = concatenate(elems)
                if prop:
                    prop = concatenate(prop)
                else:
                    prop = None
                #print elems
                #print prop
                ML.append(Mesh(m.coords,elems,prop,eltype))

            if m.nelems() == 0:
                break

        ML.append(m)

        return ML


    def splitDegenerate(self,autofix=True):
        """Split a Mesh in degenerate and non-degenerate elements.

        If autofix is True, the degenerate elements will be tested against
        known degeneration patterns, and the matching elements will be
        transformed to non-degenerate elements of a lower plexitude.

        The return value is a list of Meshes. The first holds the
        non-degenerate elements of the original Mesh. The last holds
        the remaining degenerate elements.
        The intermediate Meshes, if any, hold elements
        of a lower plexitude than the original. These may still contain
        degenerate elements.
        """
        deg = self.elems.testDegenerate()
        M0 = self.select(~deg,compact=False)
        M1 = self.select(deg,compact=False)
        if autofix:
            ML = [M0] + M1.reduceDegenerate()
        else:
            ML = [M0,M1]
            
        return ML


    def renumber(self,order='elems'):
        """Renumber the nodes of a Mesh in the specified order.

        order is an index with length equal to the number of nodes. The
        index specifies the node number that should come at this position.
        Thus, the order values are the old node numbers on the new node
        number positions.

        order can also be a predefined value that will generate the node
        index automatically:
        
        - 'elems': the nodes are number in order of their appearance in the
          Mesh connectivity.
        """
        if order == 'elems':
            order = renumberIndex(self.elems)
        newnrs = inverseUniqueIndex(order)
        return Mesh(self.coords[order],newnrs[self.elems],prop=self.prop,eltype=self.eltype)
 

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


    def revolve(self,n,axis=0,angle=360.,around=None,autofix=True):
        """Revolve a Mesh around an axis.

        Returns a new Mesh obtained by revolving the given Mesh
        over an angle around an axis in n steps, while extruding
        the mesh from one step to the next.

        This function is usually used to extrude points into lines,
        lines into surfaces and surfaces into volumes.
        By default it will try to fix the connectivity ordering where
        appropriate. If autofix is switched off, the connectivities
        are merely stacked, and the user may have to fix it himself.

        Currently, this function correctly transforms: point1 to line2,
        line2 to quad4, tri3 to wedge6, quad4 to hex8.
        """
        nplex = self.nplex()
        angles = arange(n+1) * angle / n
        coordL = [ self.coords.rotate(angle=a,axis=axis,around=around) for a in angles ]
        ML = [ Mesh(x,self.elems) for x in coordL ]

        n1 = n2 = eltype = None

        if autofix and nplex == 2:
            # fix node ordering for line2 to quad4 revolutions
            n1 = [0,1]
            n2 = [1,0]

        if autofix:
            eltype = defaultEltype(2*self.nplex())

        CL = [ connectMesh(m1,m2,1,n1,n2,eltype) for (m1,m2) in zip(ML[:-1],ML[1:]) ]
        return Mesh.concatenate(CL)


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
        return clas(coords,elems,prop=prop,eltype=eltype)

 
    # Test and clipping functions
    

    def test(self,nodes='all',dir=0,min=None,max=None,atol=0.):
        """Flag elements having nodal coordinates between min and max.

        This function is very convenient in clipping a Mesh in a specified
        direction. It returns a 1D integer array flagging (with a value 1 or
        True) the elements having nodal coordinates in the required range.
        Use where(result) to get a list of element numbers passing the test.
        Or directly use clip() or cclip() to create the clipped Mesh
        
        The test plane can be defined in two ways, depending on the value of dir.
        If dir == 0, 1 or 2, it specifies a global axis and min and max are
        the minimum and maximum values for the coordinates along that axis.
        Default is the 0 (or x) direction.

        Else, dir should be compaitble with a (3,) shaped array and specifies
        the direction of the normal on the planes. In this case, min and max
        are points and should also evaluate to (3,) shaped arrays.
        
        nodes specifies which nodes are taken into account in the comparisons.
        It should be one of the following:
        
        - a single (integer) point number (< the number of points in the Formex)
        - a list of point numbers
        - one of the special strings: 'all', 'any', 'none'
        
        The default ('all') will flag all the elements that have all their
        nodes between the planes x=min and x=max, i.e. the elements that
        fall completely between these planes. One of the two clipping planes
        may be left unspecified.
        """
        if min is None and max is None:
            raise ValueError,"At least one of min or max have to be specified."

        f = self.coords[self.elems]
        if type(nodes)==str:
            nod = range(f.shape[1])
        else:
            nod = nodes

        if type(dir) == int:
            if not min is None:
                T1 = f[:,nod,dir] > min
            if not max is None:
                T2 = f[:,nod,dir] < max
        else:
            if min is not None:
                T1 = f.distanceFromPlane(min,dir) > -atol
            if max is not None:
                T2 = f.distanceFromPlane(max,dir) < atol

        if min is None:
            T = T2
        elif max is None:
            T = T1
        else:
            T = T1 * T2

        if len(T.shape) == 1:
            return T
        
        if nodes == 'any':
            T = T.any(axis=1)
        elif nodes == 'none':
            T = (1-T.any(1)).astype(bool)
        else:
            T = T.all(axis=1)
        return T


    def clip(self,t,compact=False):
        """Return a Mesh with all the elements where t>0.

        t should be a 1-D integer array with length equal to the number
        of elements of the Mesh.
        The resulting Mesh will contain all elements where t > 0.
        """
        return self.select(t>0,compact=compact)


    def cclip(self,t,compact=False):
        """This is the complement of clip, returning a Mesh where t<=0.
        
        """
        return self.select(t<=0,compact=compact)


    def clipAtPlane(self,p,n,nodes='any',side='+'):
        """Return the Mesh clipped at plane (p,n).

        This is a convenience function returning the part of the Mesh
        at one side of the plane (p,n)
        """
        if side == '-':
            n = -n
        return self.clip(self.test(nodes=nodes,dir=n,min=p))


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
        ##areaNormals cannot be imported from trisurface. Move it to mesh.py or upper?
        
        _default_surfacetype = {
        3 : 'tri3',
        4 : 'quad4',
        6 : 'tri6',
        8 : 'quad8',
        9 : 'quad9',
        }
        def defaultSurfacetype(nplex):
            """Default face type for a surface mesh with given plexitude.
        
            For the most common cases of plexitudes, we define a default face
            type. The full list of default types can be found in
            plugins.mesh._default_facetype.
            """
            return _default_surfacetype.get(nplex,None)
        def areaNormals(x):
            """Compute the area and normal vectors of a collection of triangles.
        
            x is an (ntri,3,3) array of coordinates.
        
            Returns a tuple of areas,normals.
            The normal vectors are normalized.
            The area is always positive.
            """
            area,normals = vectorPairAreaNormals(x[:,1]-x[:,0],x[:,2]-x[:,1])
            area *= 0.5
            return area,normals
            
        nfacperel= len(getattr(elements,self.eltype.capitalize()).faces)#nfaces per elem
        mf=Mesh(self.coords, self.getFaces(unique=False))#mesh of all faces
        mf.eltype = defaultSurfacetype(mf.nplex())
        ntriperfac= mf.select([0]).convert('tri3').nelems()#how many tri per face
        elfacarea= areaNormals( mf.convert('tri3').toFormex()[:])[0].reshape(self.nelems(), nfacperel*ntriperfac)#elems'faces'areas
        return elfacarea.sum(axis=1)#elems'areas
    

    def volumes(self):
        """Return the signed volume of all the mesh elements

        For a 'tet4' tetraeder Mesh, the volume of the elements is calculated
        as 1/3 * surface of base * height.

        For other Mesh types the volumes are calculated by first splitting
        the elements into tetraeder elements.

        The return value is an array of float values with length equal to the
        number of elements.
        If the Mesh conversion to tetraeder does not succeed, the return
        value is None.
        """
        try:
            M = self.convert('tet4')
            mult = M.nelems() // self.nelems()
            if mult*self.nelems() !=  M.nelems():
                raise ValueError,"Conversion to tet4 Mesh produces nonunique split paterns"
            f = M.coords[M.elems]
            vol = 1./6. * vectorTripleProduct(f[:, 1]-f[:, 0], f[:, 2]-f[:, 1], f[:, 3]-f[:, 0])
            if mult > 1:
                vol = vol.reshape(-1,mult).sum(axis=1)
            return vol
        
        except:
            return None


    def volume(self):
        """Return the total volume of a Mesh.

        If the Mesh can not be converted to tet4 type, 0 is returned
        """
        try:
            return self.volumes().sum()
        except:
            return 0.0
    

    # ?? IS THIS DEFINED FOR *ANY* MESH ??
    def equiAngleSkew(self):
        """Returns the equiAngleSkew of the elements, a mesh quality parameter .
       
      It quantifies the skewness of the elements: normalize difference between
      the worst angle in each element and the ideal angle (angle in the face 
      of an equiangular element, qe).
      """
        eang=self.getAngles(Deg)
        eangsh= eang.shape
        eang= eang.reshape(eangsh[0], eangsh[1]*eangsh[2])
        eang.max(axis=1), eang.min(axis=1)
        eangMax, eangmin=eang.max(axis=1), eang.min(axis=1)
        el = getattr(elements,self.eltype.capitalize())
        nedginface= len( el.faces[0] )
        qe=180.*(nedginface-2.)/nedginface
        extremeAngles= [ (eangMax-qe)/(180.-qe), (qe-eangmin)/qe ]
        return array(extremeAngles).max(axis=0)


    def actor(self,**kargs):

        if self.nelems() == 0:
            return None
        
        from gui.actors import GeomActor
        return GeomActor(self,**kargs)


    ################ DEPRECATED ###############
    @deprecation("Mesh.unselect is deprecated. Use Mesh.cselect instead")
    def unselect(self,*args,**kargs):
        return self.cselect(*args,**kargs)


######################## Functions #####################


def mergeNodes(nodes,fuse=True,**kargs):
    """Merge all the nodes of a list of node sets.

    Each item in nodes is a Coords array.
    The return value is a tuple with:
    
    - the coordinates of all unique nodes,
    - a list of indices translating the old node numbers to the new.

    The merging operation can be tuned by specifying extra arguments
    that will be passed to :meth:`Coords.fuse`.
    """
    coords = Coords(concatenate([x for x in nodes],axis=0))
    if fuse:
        coords,index = coords.fuse(**kargs)
    else:
        index = arange(coords.shape[0])
    n = array([0] + [ x.npoints() for x in nodes ]).cumsum()
    ind = [ index[f:t] for f,t in zip(n[:-1],n[1:]) ]
    return coords,ind


def mergeMeshes(meshes,fuse=True,**kargs):
    """Merge all the nodes of a list of Meshes.

    Each item in meshes is a Mesh instance.
    The return value is a tuple with:

    - the coordinates of all unique nodes,
    - a list of elems corresponding to the input list,
      but with numbers referring to the new coordinates.

    The merging operation can be tuned by specifying extra arguments
    that will be passed to :meth:`Coords:fuse`.
    Setting fuse=False will merely concatenate all the mesh.coords, but
    not fuse them.
    """
    coords = [ m.coords for m in meshes ]
    elems = [ m.elems for m in meshes ]
    coords,index = mergeNodes(coords,fuse,**kargs)
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
    mesh1 = mesh1.compact()
    mesh2 = mesh2.compact()

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
    if type(n)!=int:n=len(n)-1
    e = concatenate([et+i*nnod for i in range(n)])
    return Mesh(x,e,eltype=eltype).setProp(mesh1.prop)
        
# define this also as a Mesh method
Mesh.connect = connectMesh
def connectQuadraticMesh(mesh1,mesh2,n=1, eltype='Hex20'):
    """currently works for Quad8 only.

    Connect two Quad8 meshes to form a Hex20 mesh.
    """
    #this is a proposal. Is it better to implement the conversion in the _conversions_ ?
    h16=connectMesh(mesh1,mesh2,n=n)
    #now a conversion hex16 to hex20
    h20=h16.addMeanNodes( [ (0,8), (1,9), (2,10), (3,11) ],'hex20')
    h20.elems=h20.elems[:, [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19]]
    #finally fuse (needed after addMeanNodes) and renumber
    return h20

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



## def structuredHexGrid(dx, dy, dz, isophex='hex64'):
##     """_it builds a structured hexahedral grid with nodes and elements both numbered in a structured way: first along z, then along y,and then along x. The resulting hex cells are oriented along z. This function is the equivalent of simple.rectangularGrid but for a mesh. Additionally, dx,dy,dz can be either integers or div (1D list or array). In case of list/array, first and last numbers should be 0.0 and 1.0 if the desired grid has to be inside the region 0.,0.,0. to 1.,1.,1.
##     If isopHex is specified, a convenient set of control points for the isoparametric transformation hex64 is also returned.
##     TODO: include other options to get the control points for other isoparametric transformation for hex."""
##     sgx, sgy, sgz=dx, dy, dz
##     if type(dx)!=int:sgx=len(dx)-1
##     if type(dy)!=int:sgy=len(dy)-1
##     if type(dz)!=int:sgz=len(dz)-1
##     n3=regularGrid([0., 0., 0.],[1., 1., 1.],[sgx, sgy, sgz])
##     if type(dx)!=int:n3[..., 0]=array(dx).reshape(-1, 1, 1)
##     if type(dy)!=int:n3[..., 1]=array(dy).reshape(-1,  1)
##     if type(dz)!=int:n3[..., 2]=array(dz).reshape(-1)
##     nyz=(sgy+1)*(sgz+1)
##     xh0= array([0, nyz, nyz+sgz+1,0+sgz+1 ])
##     xh0= concatenate([xh0, xh0+1], axis=1)#first cell
##     hz= array([xh0+j for j in range(sgz)])#z column
##     hzy= array([hz+(sgz+1)*j for j in range(sgy)])#zy 2D rectangle
##     hzyx=array([hzy+nyz*k for k in range(sgx)]).reshape(-1, 8)#zyx 3D
##     if isophex=='hex64': return Coords(n3.reshape(-1, 3)), hzyx.reshape(-1, 8), regularGrid([0., 0., 0.], [1., 1., 1.], [3, 3, 3]).reshape(-1, 3)#control points for the hex64 applied to a basic struct hex grid
##     else: return Coords(n3.reshape(-1, 3)), hzyx.reshape(-1, 8)




# BV:
# While this function seems to make sense, it should be avoided
# The creator of the mesh normally KNOWS the correct connectivity,
# and should immediately fix it, instead of calculating it from
# coordinate data
def correctHexMeshOrientation(hm):
    """_hexahedral elements have an orientation.

    Some geometrical transformation (e.g. reflect) may produce
    inconsistent orientation, which results in negative (signed)
    volume of the hexahedral (triple product).
    This function fixes the hexahedrals without orientation.
    """
    from formex import vectorTripleProduct
    hf=hm.coords[hm.elems]
    tp=vectorTripleProduct(hf[:, 1]-hf[:, 0], hf[:, 2]-hf[:, 1], hf[:, 4]-hf[:, 0])# from formex.py
    hm.elems[tp<0.]=hm.elems[tp<0.][:,  [4, 5, 6, 7, 0, 1, 2, 3]]
    return hm


# End
