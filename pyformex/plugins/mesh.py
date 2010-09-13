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
    'wedge6': {
        'tet4'  : [ ('s', [ (0,1,2,4),(4,6,5,2) ]), ],
        },
    'hex8': {
        'wedge6': [ ('s', [ (0,1,2,4,5,6),(2,3,0,6,7,4) ]), ],
        'tet4'  : [ ('s', [ (0,1,2,5),(2,3,0,7),(5,7,6,2),(7,5,4,0),(0,5,2,7) ]), ],
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
    
    If eltype is None, a default eltype is deived from the plexitude.

    A Mesh can be initialized by its attributes (coords,elems,prop,eltype)
    or by a single geometric object that provides a toMesh() method.
    """
    def __init__(self,coords=None,elems=None,prop=None,eltype=None):
        """Initialize a new Mesh."""
        self.coords = self.elems = self.prop = self.eltype = None

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
            self.eltype = eltype


    def setCoords(self,coords):
        """Replace the current coords with new ones.

        Returns a Mesh exactly like the current except for the position
        of the coordinates.
        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            return Mesh(coords,self.elems,self.prop,self.eltype)
        else:
            raise ValueError,"Invalid reinitialization of Mesh coords"


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
        return Mesh(self.coords,self.elems,self.prop,self.eltype)


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
        """Get the coords data."""
        return self.coords

    
    def getElems(self):
        """Get the elems data."""
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
        For bothe meshes however,  getLowerEntities(+1) returns the edges.

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


    def getEdges(self,unique=False):
        """Return the edges of the elements.

        This is a convenient function to create a table with the element
        edges. It is equivalent to  ```self.getLowerEntities(1,unique)```
        """
        return self.getLowerEntities(1,unique)


    def getFaces(self,unique=False):
        """Return the faces of the elements.

        This is a convenient function to create a table with the element
        faces. It is equivalent to ```self.getLowerEntities(2,unique)```
        """
        return self.getLowerEntities(2,unique)


    # ?? DOES THIS WORK FOR *ANY* MESH ??
    def getAngles(self, angle_spec=Deg):
        """Returns the angles in Deg or Rad between the edges of a mesh.
        
        The returned angles are shaped  as (nelems, n1faces, n1vertices),
        where n1faces are the number of faces in 1 element and the number of vertices in 1 face.
        """
        mf=self.getCoords()[self.getFaces()]
        el = getattr(elements,self.eltype.capitalize())
        v = mf - roll(mf,-1,axis=1)
        v=normalize(v)
        v1=-roll(v,+1,axis=1)
        angfac= arccos( dotpr(v, v1) )*180./math.pi
        return angfac.reshape(self.nelems(),len(el.faces), len(el.faces[0]))


    def getBorder(self):
        """Return the border of the Mesh.

        This returns a Connectivity table with the border of the Mesh.
        The border entities are of a lower jierarchical level than the
        mesh itself. This Connectivity can be used together with the
        Mesh coords to construct a Mesh of the border geometry.
        See also getBorderMesh.
        """
        sel = self.getLowerEntitiesSelector(-1)
        hi,lo = self.elems.insertLevel(sel)
        hiinv = hi.inverse()
        ncon = (hiinv>=0).sum(axis=1)
        brd = (ncon<=1)
        brd = lo[brd]
        return brd


    def getBorderMesh(self):
        """Return a Mesh with the border elements.

        Returns a Mesh representing the border of the Mesh.
        The new Mesh is of the next lower hierarchical level.
        """
        return Mesh(self.coords,self.getBorder())


    def report(self):
        bb = self.bbox()
        return """
Shape: %s nodes, %s elems, plexitude %s
BBox: %s, %s
Size: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),bb[1],bb[0],bb[1]-bb[0])


    def __str__(self):
        return self.report() + "Coords:\n" + self.coords.__str__() +  "Elems:\n" + self.elems.__str__()


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
        """Remove unconnected nodes and renumber the mesh.

        Beware! This function changes the object in place.
        """
        nodes = unique1d(self.elems)
        if nodes.size == 0:
            self.__init__([],[])
        
        elif nodes.shape[0] < self.ncoords() or nodes[-1] >= nodes.size:
            coords = self.coords[nodes]
            if nodes[-1] >= nodes.size:
                elems = inverseUniqueIndex(nodes)[self.elems]
            else:
                elems = self.elems
            self.__init__(coords,elems,prop=self.prop,eltype=self.eltype)

        return self


    def select(self,selected):
        """Return a mesh with selected elements from the original.

        - `selected`: an object that can be used as an index in the
          `elems` array, e.g. a list of element numbers.
          
        Returns a Mesh with only the selected elements.
        The returned mesh is not compacted.
        """
        if len(self.elems) == 0:
            return self
        prop = self.prop
        if prop is not None:
            prop = prop[selected]
        elems = self.elems[selected]
        return Mesh(self.coords,elems,prop,self.eltype)


    def unselect(self, unselected):
        """Return a mesh without the unselected elements.
        """
        wi=ones([self.nelems()])
        wi[unselected]=0
        return self.clip(wi)


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

    
    def withProp(self,val):
        """Return a Mesh which holds only the elements with property val.

        val is either a single integer, or a list/array of integers.
        The return value is a Mesh holding all the elements that
        have the property val, resp. one of the values in val.
        The returned Mesh inherits the matching properties.
        
        If the Mesh has no properties, a copy with all elements is returned.
        """
        if self.prop is None:
            return Mesh(self.coords,self.elems,eltype=self.eltype)
        elif type(val) == int:
            return Mesh(self.coords,self.elems[self.prop==val],val,self.eltype)
        else:
            t = zeros(self.prop.shape,dtype=bool)
            for v in asarray(val).flat:
                t += (self.prop == v)
            return Mesh(self.coords,self.elems[t],self.prop[t],self.eltype)
            

    def splitProp(self):
        """Partition aMesh according to its prop values.

        Returns a dict with the prop values as keys and the corresponding
        partitions as values. Each value is a Mesh instance.
        It the Mesh has no props, an empty dict is returned.
        """
        if self.prop is None:
            return {}
        else:
            return dict([(p,self.withProp(p)) for p in self.propSet()])
    

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


    def splitRandom(self,n):
        """Split a mesh in n parts, distributing the elements randomly."""
        sel = random.randint(0,n,(self.nelems()))
        return [ self.select(sel==i) for i in range(n) if i in sel ]


    def convertRandom(self,choices):
        """Convert choosing randomly between choices"""
        ml = self.splitRandom(len(choices))
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
                    m = m.select(~w)

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
        M0 = self.select(~deg)
        M1 = self.select(deg)
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

 
    # Test and clipping functions
    

    def test(self,nodes='all',dir=0,min=None,max=None,atol=0.):
        """Flag elements having nodal coordinates between min and max.

        This function is very convenient in clipping a TriSurface in a specified
        direction. It returns a 1D integer array flagging (with a value 1 or
        True) the elements having nodal coordinates in the required range.
        Use where(result) to get a list of element numbers passing the test.
        Or directly use clip() or cclip() to create the clipped TriSurface
        
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


    def clip(self,t):
        """Return a TriSurface with all the elements where t>0.

        t should be a 1-D integer array with length equal to the number
        of elements of the TriSurface.
        The resulting TriSurface will contain all elements where t > 0.
        """
        return self.select(t>0)


    def cclip(self,t):
        """This is the complement of clip, returning a TriSurface where t<=0.
        
        """
        return self.select(t<=0)


    def clipAtPlane(self,p,n,nodes='any',side='+'):
        """Return the Mesh clipped at plane (p,n).

        This is a convenience function returning the part of the Mesh
        at one side of the plane (p,n)
        """
        if side == '-':
            n = -n
        return self.clip(self.test(nodes=nodes,dir=n,min=p))


    def volumes(self):
        """Return the signed volume of all the mesh elements
        by splitting into tet (tet volume is 1/3 * base *height). 
        """
        if self.eltype=='tet4': 
            f=self.coords[self.elems]
            return 1./6. * vectorTripleProduct(f[:, 1]-f[:, 0], f[:, 2]-f[:, 1], f[:, 3]-f[:, 0])
        #if self.eltype=='wedge6': return self.convert('tet4').volumes().reshape(-1, 3).sum(axis=1)#the conversion of the wedge6 to tet4 may be wrong!
        if self.eltype=='hex8': return self.convert('tet4').volumes().reshape(-1, 5).sum(axis=1)


    def volume(self):
        """Return the total volume of a Mesh.
        """
        return self.volumes().sum()


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
    if type(n)!=int:n=len(n)-1
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



def structuredHexGrid(dx, dy, dz, isophex='hex64'):
    """it builds a structured hexahedral grid with nodes and elements both numbered in a structured way: first along z, then along y,and then along x. The resulting hex cells are oriented along z. This function is the equivalent of simple.rectangularGrid but for a mesh. Additionally, dx,dy,dz can be either integers or div (1D list or array). In case of list/array, first and last numbers should be 0.0 and 1.0 if the desired grid has to be inside the region 0.,0.,0. to 1.,1.,1.
    If isopHex is specified, a convenient set of control points for the isoparametric transformation hex64 is also returned.
    TODO: include other options to get the control points for other isoparametric transformation for hex."""
    sgx, sgy, sgz=dx, dy, dz
    if type(dx)!=int:sgx=len(dx)-1
    if type(dy)!=int:sgy=len(dy)-1
    if type(dz)!=int:sgz=len(dz)-1
    n3=regularGrid([0., 0., 0.],[1., 1., 1.],[sgx, sgy, sgz])
    if type(dx)!=int:n3[..., 0]=array(dx).reshape(-1, 1, 1)
    if type(dy)!=int:n3[..., 1]=array(dy).reshape(-1,  1)
    if type(dz)!=int:n3[..., 2]=array(dz).reshape(-1)
    nyz=(sgy+1)*(sgz+1)
    xh0= array([0, nyz, nyz+sgz+1,0+sgz+1 ])
    xh0= concatenate([xh0, xh0+1], axis=1)#first cell
    hz= array([xh0+j for j in range(sgz)])#z column
    hzy= array([hz+(sgz+1)*j for j in range(sgy)])#zy 2D rectangle
    hzyx=array([hzy+nyz*k for k in range(sgx)]).reshape(-1, 8)#zyx 3D
    if isophex=='hex64': return Coords(n3.reshape(-1, 3)), hzyx.reshape(-1, 8), regularGrid([0., 0., 0.], [1., 1., 1.], [3, 3, 3]).reshape(-1, 3)#control points for the hex64 applied to a basic struct hex grid
    else: return Coords(n3.reshape(-1, 3)), hzyx.reshape(-1, 8)




# BV:
# While this function seems to make sense, it should be avoided
# The creator of the mesh normally KNOWS the correct connectivity,
# and should immediately fix it, instead of calculating it from
# coordinate data
# 

def correctHexMeshOrientation(hm):
    """hexahedral elements have an orientation. Some geometrical transformation (e.g. reflect) may produce inconsistent orientation, which results in negative (signed) volume of the hexahedral (triple product). This function fixes the hexahedrals without orientation. """
    from formex import vectorTripleProduct
    hf=hm.coords[hm.elems]
    tp=vectorTripleProduct(hf[:, 1]-hf[:, 0], hf[:, 2]-hf[:, 1], hf[:, 4]-hf[:, 0])# from formex.py
    hm.elems[tp<0.]=hm.elems[tp<0.][:,  [4, 5, 6, 7, 0, 1, 2, 3]]
    return hm


# End
