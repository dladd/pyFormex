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
from elements import elementType,elementName
from utils import deprecation
from geometry import Geometry
from simple import regularGrid


#################### This first section holds experimental stuff!! #####

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
    - eltype: an Element subclass or a string designing the element type,
      default None.
    
    If eltype is None, a default eltype is derived from the plexitude, by
    calling the elements.elementType function.
    For plexitudes without default type,
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
        self.elem_edges = self.eadj = None
        self.conn = self.econn = self.fconn = None 
        
        if coords is None:
            # Create an empty Mesh object
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

        self.setType(eltype)
        self.setProp(prop)


    def _set_coords(self,coords):
        """Replace the current coords with new ones.

        Returns a Mesh or subclass exactly like the current except
        for the position of the coordinates.
        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            return self.__class__(coords,self.elems,prop=self.prop,eltype=self.eltype)
        else:
            raise ValueError,"Invalid reinitialization of %s coords" % self.__class__


    def setType(self,eltype=None):
        """Set the eltype from a character string.

        This function allows the user to change the element type of the Mesh.
        The input is a character string with the name of one of the element
        defined in elements.py. The function will only allow to set a type
        matching the plexitude of the Mesh.

        This method is seldom needed, because the applications should
        normally set the element type at creation time.
        """
        self.eltype = elementType(eltype,self.nplex())
        return self
    

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


    def __getitem__(self,i):
        """Allows elements to be addressed as self[i].

        self[i] returns an array with the coordinates of element i
        self[i][j] then will return the coordinates of node j of element i
        """
        return self.coords[self.elems[i]]
    

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
        return Formex(self.coords[self.elems],self.prop,self.eltype.name())

    
    def ndim(self):
        return 3
    def ngrade(self):
        return self.eltype.ndim
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

    def info(self):
        return "coords" + str(self.coords.shape) + "; elems" + str(self.elems.shape)
    

    def nedges(self):
        """Return the number of edges.

        This returns the number of rows that would be in getEdges(),
        without actually constructing the edges.
        The edges are not fused!
        """
        try:
            return self.nelems() * self.eltype.nedges()
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
        For both meshes however,  getLowerEntities(+1) returns the edges.

        By default, all entities for all elements are returned and common
        entities will appear multiple times. Specifying unique=True will 
        return only the unique ones.

        The return value may be an empty table, if the element type does
        not have the requested entities (e.g. the 'point' type).
        If the eltype is not defined, or the requested entity level is
        outside the range 0..3, the return value is None.
        """
        if level < 0:
            level = self.eltype.ndim + level

        if level < 0 or level > 3:
            return None

        attr = ['points', 'edges', 'faces', 'cells'][level]
        return array(getattr(self.eltype,attr))


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
        requests can return it without the need for computing it again.
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
        edges. It is equivalent to ```self.getLowerEntities(1,unique=True)```,
        but this also stores the result internally so that future
        requests can return it without the need for computing it again.
        """
        if self.edges is None:
            self.edges = self.getLowerEntities(1,unique=True)
        return self.edges
        

    def getFaces(self):
        """Return the unique faces of all the elements in the Mesh.

        This is a convenient function to create a table with the element
        faces. It is equivalent to ```self.getLowerEntities(2,unique=True)```,
        but this also stores the result internally so that future
        requests can return it without the need for computing it again.
        """
        from plugins.trisurface import TriSurface
        if self.__class__ == TriSurface:
            import warnings
            warnings.warn('warn_trisurface_getfaces')

        if self.faces is None:
            self.faces = self.getLowerEntities(2,unique=True)
        return self.faces
        

    def getCells(self):
        """Return the cells of the elements.

        This is a convenient function to create a table with the element
        cells. It is equivalent to ```self.getLowerEntities(3,unique=True)```,
        but this also stores the result internally so that future
        requests can return it without the need for computing it again.
        """
        if self.cells is None:
            self.cells = self.getLowerEntities(3,unique=True)
        return self.cells


    def getElemEdges(self):
        """Defines the elements in function of its edges.

        This returns a Connectivity table with the elements defined in
        function of the edges. It is equivalent to
        ```self.elems.insertLevel(self.getLowerEntitiesSelector(1))```
        but it also stores the definition of the edges and the returned
        element to edge connectivity.
        """
        if self.elem_edges is None:
            sel = self.getLowerEntitiesSelector(1)
            self.elem_edges,self.edges = self.elems.insertLevel(sel)
        return self.elem_edges


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
    # Adjacency #
    

    def nodeConnections(self):
        """Find and store the elems connected to nodes."""
        if self.conn is None:
            self.conn = self.elems.inverse()
        return self.conn
    

    def nNodeConnected(self):
        """Find the number of elems connected to nodes."""
        return (self.nodeConnections() >=0).sum(axis=-1)


    def edgeConnections(self):
        """Find and store the elems connected to edges."""
        if self.econn is None:
            self.econn = self.getElemEdges().inverse()
        return self.econn


    def nEdgeConnected(self):
        """Find the number of elems connected to edges."""
        return (self.edgeConnections() >=0).sum(axis=-1)


    def nodeAdjacency(self):
        """Find the elems adjacent to each elem via one or more nodes."""
        return self.elems.adjacency()


    def nNodeAdjacent(self):
        """Find the number of elems which are adjacent by node to each elem."""
        return (self.nodeAdjacency() >=0).sum(axis=-1)


    def edgeAdjacency(self):
        """Find the elems adjacent to elems via an edge."""
        return self.getElemEdges().adjacency()


    def nEdgeAdjacent(self):
        """Find the number of adjacent elems."""
        return (self.edgeAdjacency() >=0).sum(axis=-1)


    # BV: Should be made dependent on getFaces
    # or a function??
    # ?? DOES THIS WORK FOR *ANY* MESH ??
    # What with a mesh of points, lines, ...
    # Also, el.faces can contain items of different length
    @deprecation("The use of this function is discouraged!")
    def getAngles(self, angle_spec=Deg):
        """_Returns the angles in Deg or Rad between the edges of a mesh.
        
        The returned angles are shaped  as (nelems, n1faces, n1vertices),
        where n1faces are the number of faces in 1 element and the number
        of vertices in 1 face.
        """
        #mf = self.coords[self.getFaces()]
        mf = self.coords[self.getLowerEntities(2,unique=False)]
        v = mf - roll(mf,-1,axis=1)
        v=normalize(v)
        v1=-roll(v,+1,axis=1)
        angfac= arccos( clip(dotpr(v, v1), -1., 1. ))/angle_spec
        el = self.eltype
        return angfac.reshape(self.nelems(),len(el.faces), len(el.faces[0]))


    # BV: REMOVED node2nodeAdjacency:
    #
    # Either use
    #  - self.elems.nodeAdjacency() (gives nodes connected by elements)
    #  - self.getEdges().nodeAdjacency() (gives nodes connected by edges)


    # BV: name is way too complex
    # should not be a mesh method, but of some MeshValue class? Field?
    def avgNodalScalarOnAdjacentNodes(self, val, iter=1):
        """_Smooth nodal scalar values by averaging over adjacent nodes iter times.
        
        Nodal scalar values (val is a 1D array of self.ncoords() scalar values )
        are averaged over adjacent nodes an number of time (iter)
        in order to provide a smoothed mapping. 
        """
        
        if iter==0: return val
        nadjn=self.getEdges().adjacency(kind='n')
        nadjn=[x[x>=0] for x in nadjn]
        lnadjn=[len(i) for i in nadjn]
        lmax= max( lnadjn )
        adjmatrix=zeros([self.ncoords(), lmax], float)
        avgval=val
        for i in range(iter):
            for i in range( self.ncoords()  ):
                adjmatrix[i, :len( nadjn[i]) ]=avgval[ nadjn[i]  ]
            avgval= sum(adjmatrix, axis=1)/lnadjn
        return avgval


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


    def matchCoords(self,mesh,**kargs):
        """Match nodes of Mesh with nodes of self.

        This is a convenience function equivalent to::

           self.coords.match(mesh.coords,**kargs)

        See also :meth:`Coords.match`
        """
        return self.coords.match(mesh.coords,**kargs)
        
    
    def matchElemsCentroids(self, mesh,**kargs):
        """Match elems of Mesh with elems of self.
        
        self and Mesh are same eltype meshes
        and are both without Doubles.
        
        Elems are matched by their centroids.
        """
        c=Mesh(self.centroids(), arange(self.nelems() ))
        mc=Mesh(mesh.centroids(), arange(mesh.nelems() ))
        return c.matchCoords(mc,**kargs)
        

    # BV: I'm not sure that we need this. Looks like it can or should
    # be replaced with a method applied on the BorderMesh
    #~ FI It has been tested on quad4-quad4, hex8-quad4, tet4-tri3
    def matchFaces(self,mesh):
        """Match faces of mesh with faces of self.
            
        self and Mesh can be same eltype meshes or different eltype but of the 
        same hierarchical type (i.e. hex8-quad4 or tet4 - tri3) 
        and are both without Doubles.
            
        Returns the indices array of the elems of self that matches
        the faces of mesh
        """
        sel = self.getLowerEntitiesSelector(2)
        hi,lo = self.elems.insertLevel(sel)
        hiinv = hi.inverse()
        fm=Mesh(self.coords,self.getFaces())
        mesh=Mesh(mesh.coords,mesh.getFaces())
        c=fm.matchElemsCentroids(mesh)
        hiinv = hiinv[c]
        enr =  unique(hiinv[hiinv >= 0])  # element number
        return enr
    
    
    @deprecation("Mesh.findCoincidentNodes is deprecated. Use Coords.match or Mesh.matchCoords. Beware for order of arguments!")
    def findCoincidentCoords(self,mesh,**kargs):
        return mesh.coords.match(self.coords)


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
            return self.__class__([],[])
        
        elif nodes.shape[0] < self.ncoords() or nodes[-1] >= nodes.size:
            coords = self.coords[nodes]
            if nodes[-1] >= nodes.size:
                elems = inverseUniqueIndex(nodes)[self.elems]
            else:
                elems = self.elems
            return self.__class__(coords,elems,prop=self.prop,eltype=self.eltype)
        
        else:
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
        #print "OLD plex = %s, NEW plex = %s, eltype = %s" % (self.nplex(),elems.nplex(),eltype)
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


    def selectNodes(self,nodsel,eltype=None):
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
        ps=self.propSet()
        if type(val)==int:
            t=ps==val
        else:
            t=sum([ps==v for v in val], axis=0)
        return self.withProp(ps[t==0])


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

        if elementName(totype) == self.eltype.name():
            return self

        strategy = self.eltype.conversions.get(totype,None)

        while not type(strategy) is list:
            # This allows for aliases in the conversion database
            strategy = self.eltype.conversions.get(strategy,None)
            
            if strategy is None:
                raise ValueError,"Don't know how to convert %s -> %s" % (self.eltype,totype)

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


    def removeDegenerate(self,eltype=None):
        """Remove the degenerate elements from a Mesh.

        Returns a Mesh with all degenerate elements removed.
        """
        deg = self.elems.testDegenerate()
        M0 = self.select(~deg,compact=False)
        return M0


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
        return self.__class__(self.coords[order],newnrs[self.elems],prop=self.prop,eltype=self.eltype)


    def renumberElems(self,order='nodes'):
        """Renumber the elements of a Mesh.

        Parameters:

        - `order`: either a 1-D integer array with a permutation of
          ``arange(self.nelems())``, specifying the requested order, or one of
          the following predefined strings:

          - 'nodes': order the elements in increasing node number order.
          - 'random': number the elements in a random order.
          - 'reverse': number the elements in. 

        Returns:
          A Mesh equivalent with self but with the elements ordered as specified.

        See also: :meth:`Connectivity.reorder`
        """
        order = self.elems.reorder(order)
        return self.__class__(self.coords,self.elems[order],prop=self.prop[order],eltype=self.eltype)
 

    def extrude(self,n,step=1.,dir=0,eltype=None):
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
        #~ nplex = self.nplex()
        el = self.eltype
        coord2 = self.coords.translate(dir,n*step)
        M = connectMesh(self,Mesh(coord2,self.elems,eltype=el),n,eltype=eltype)
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
            eltype = elementType(nplex=2*self.nplex())

        CL = [ connectMesh(m1,m2,1,n1,n2,eltype) for (m1,m2) in zip(ML[:-1],ML[1:]) ]
        return Mesh.concatenate(CL)

#FI TODO remove autofix flag check if already works with the new connectMesh
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
        ML = [ Mesh(x,self.elems,prop=self.prop) for x in seq ]
        M = connectMeshSequence(ML)

        if autofix and nplex == 2:
            # fix node ordering for line2 to quad4 extrusions
            M.elems[:,-nplex:] = M.elems[:,-1:-(nplex+1):-1].copy()

        if autofix:
            M.eltype = elementType(nplex=M.nplex())

        return M


    def __add__(self,other):
        """Return the sum of two Meshes.

        The sum of the Meshes is simply the concatenation thereof.
        It allows us to write simple expressions as M1+M2 to concatenate
        the Meshes M1 and M2. Both meshes should be of the same plexitude
        and have the same eltype.
        """
        return Mesh.concatenate([self,other])
    

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
    #this is define for every linear surface or linear volume mesh.
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
        f= self.eltype.getFaces()
        nedginface=len( array(f[ f.keys()[0] ], int)[0] )
        qe=180.*(nedginface-2.)/nedginface
        extremeAngles= [ (eangMax-qe)/(180.-qe), (qe-eangmin)/qe ]
        return array(extremeAngles).max(axis=0)


    def actor(self,**kargs):

        if self.nelems() == 0:
            return None
        
        from gui.actors import GeomActor
        return GeomActor(self,**kargs)


    # BV: removed in 0.8.4
    ## ################ DEPRECATED ###############
    ## @deprecation("Mesh.unselect is deprecated. Use Mesh.cselect instead")
    ## def unselect(self,*args,**kargs):
    ##     return self.cselect(*args,**kargs)


######################## Functions #####################


def mergeNodes(nodes,fuse=True,**kargs):
    """Merge all the nodes of a list of node sets.

    Merging the nodes creates a single Coords object containing all nodes,
    and the indices to find the points of the original node sets in the
    merged set.

    Parameters:

    - `nodes`: a list of Coords objects, all having the same shape, except
      possibly for their first dimension
    - `fuse`: if True (default), coincident (or very close) points will
      be fused to a single point
    - `**kargs`: keyword arguments that are passed to the fuse operation

    Returns:
    
    - a Coords with the coordinates of all (unique) nodes,
    - a list of indices translating the old node numbers to the new. These
      numbers refer to the serialized Coords.

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


def connectMesh(mesh1,mesh2,div=1,n1=None,n2=None,eltype=None):
    """Connect two meshes to form a hypermesh.

    Parameters:

    - `mesh1`, `mesh2`: Mesh objects with the same element type and shape
      (number of elements and plexitude).
      The two Meshes usually also have the same topology.
      Both Meshes are connected to form a hypermesh. The plexitude of the
      new Mesh is two times that of the original Meshes.
    - `div`: Either an integer, or a sequence of numbers (usually between 0.0
      and 1.0). This parameter has the same meaning as in `Coords.interpolate`.
      number of
      elements in the direction between the two Meshes is determined by . 
    n1 and n2 are node selection indices permitting a permutation of the
    nodes of the base sets in their appearance in the hypermesh.
    This can e.g. be used to achieve circular numbering of the hypermesh.
    """
    import warnings
    warnings.warn("Beware, connectMesh is currently under revision and may be broken!")
    # For compatibility, allow meshes to be specified as tuples
    if type(mesh1) is tuple:
        mesh1 = Mesh(mesh1)
    if type(mesh2) is tuple:
        mesh2 = Mesh(mesh2)

    if mesh1.eltype != mesh2.eltype or mesh1.shape() != mesh2.shape():
        raise ValueError,"Meshes are not compatible"

    # get the eltype of the base Mesh, the extruded eltype and reordering
    eltyp0 = mesh1.eltype
    print "BASE ELTYPE = %s" % eltyp0
    eltyp1,reorder = eltyp0.extruded
    print "EXTRUDED ELTYPE = %s" % eltyp1
    

    # compact the node numbering schemes
    mesh1 = mesh1.compact()
    mesh2 = mesh2.compact()

    # Create the interpolations of the coordinates
    if type(div) == int:
        div = arange(div+1) / float(div)
    else:
        div = array(div).ravel()
    x = Coords.interpolate(mesh1.coords,mesh2.coords,div).reshape(-1,3)

    nnod = mesh1.ncoords()
    nplex = mesh1.nplex()
    if n1 is None:
        n1 = range(nplex)
    if n2 is None:
        n2 = range(nplex)
    e1 = mesh1.elems[:,n1]
    e2 = mesh2.elems[:,n2] + nnod
    et = concatenate([e1,e2],axis=-1)
    e = concatenate([et+i*nnod for i in range(div.size-1)])
    # Reorder nodes if necessary
    if reorder:
        print "REORDER NODES",reorder
        e = e[:,reorder]
    eM = Mesh(x,e,eltype=eltyp1).setProp(mesh1.prop)

    # convert to proper eltype
    if eltype:
        print eltype
        eM = eM.convert(eltype)

    return eM

        
# define this also as a Mesh method
#~ Mesh.connect= connectMesh

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
