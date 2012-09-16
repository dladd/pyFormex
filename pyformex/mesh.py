# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
#
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

from coords import *
from formex import Formex
from connectivity import Connectivity
from elements import elementType
from utils import deprecation
from geometry import Geometry
from simple import regularGrid
   

##############################################################

class Mesh(Geometry):
    """A Mesh is a discrete geometrical model defined by nodes and elements.

    In the Mesh geometrical data model, the coordinates of all the points
    are gathered in a single twodimensional array with shape (ncoords,3).
    The individual geometrical elements are then described by indices into
    the coordinates array.
    
    This model has some advantages over the Formex data model (which stores
    all the points of all the elements by their coordinates):
    
    - a more compact storage, because coordinates of coinciding
      points are not repeated,
    - faster connectivity related algorithms.
    
    The downside is that geometry generating algorithms are far more complex
    and possibly slower.
    
    In pyFormex we therefore mostly use the Formex data model when creating
    geometry, but when we come to the point of exporting the geometry to
    file (and to other programs), a Mesh data model may be more adequate.

    The Mesh data model has at least the following attributes:
    
    - coords: (ncoords,3) shaped Coords object, holding the coordinates of
      all points in the Mesh;
    - elems: (nelems,nplex) shaped Connectivity object, defining the elements
      by indices into the Coords array. All values in elems should be in the
      range 0 <= value < ncoords.
    - prop: an array of element property numbers, default None.
    - eltype: an element type (a subclass of :class:`Element`) or the name
      of an Element type, or None (default).
      If eltype is None, the eltype of the elems Connectivity table is used,
      and if that is missing, a default eltype is derived from the plexitude,
      by a call to :func:`elements.elementType`.
      In most cases the eltype can be set automatically.
      The user can override the default value, but an error will occur if
      the element type does not exist or does not match the plexitude.
      
    A Mesh can be initialized by its attributes (coords,elems,prop,eltype)
    or by a single geometric object that provides a toMesh() method.

    If only an element type is provided, a unit sized single element Mesh
    of that type is created. Without parameters, an empty Mesh is created.
    
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

    def _formex_transform(func):
        """Perform a Formex transformation on the .coords attribute of the object.

        This is a decorator function. It should be used only for Formex methods
        which are not Geometry methods as well.
        """
        formex_func = getattr(Formex,func.__name__)
        def newf(self,*args,**kargs):
            """Performs the Formex %s transformation on the coords attribute"""
            F = Formex(self.coords).formex_func(self.coords,*args,**kargs)
            return self._set_coords(coords_func(self.coords,*args,**kargs))
        newf.__name__ = func.__name__
        newf.__doc__ = coords_func.__doc__
        return newf

    
    def __init__(self,coords=None,elems=None,prop=None,eltype=None):
        """Initialize a new Mesh."""
        self.coords = self.elems = self.prop = self.eltype = None
        self.ndim = -1
        self.nodes = self.edges = self.faces = self.cells = None
        self.elem_edges = self.eadj = None
        self.conn = self.econn = self.fconn = None 
        
        if coords is None:
            if eltype is None:
                # Create an empty Mesh object
                return

            else:
                # Create unit Mesh of specified type
                el = elementType(eltype)
                coords = el.vertices
                elems = el.getElement()

        if elems is None:
            # A single object was specified instead of (coords,elems) pair
            try:
                # initialize from a single object
                if isinstance(coords,Mesh):
                    M = coords
                else:
                    M = coords.toMesh()
                coords,elems = M.coords,M.elems
            except:
                raise ValueError,"No `elems` specified and the first argument can not be converted to a Mesh."

        try:
            self.coords = Coords(coords)
            if self.coords.ndim != 2:
                raise ValueError,"\nExpected 2D coordinate array, got %s" % self.coords.ndim
            self.elems = Connectivity(elems)
            if self.elems.size > 0 and (
                self.elems.max() >= self.coords.shape[0] or
                self.elems.min() < 0):
                raise ValueError,"\nInvalid connectivity data: some node number(s) not in coords array (min=%s, max=%s, ncoords=%s)" % (self.elems.min(),self.elems.max(),self.coords.shape[0])
        except:
            raise

        self.setType(eltype)
        self.setProp(prop)



    def __getattribute__(self,name):
        """This is temporarily here to warn people about the eltype removal"""
        if name == 'eltype':
            warn("depr_mesh_eltype")

        # Default behaviour
        return object.__getattribute__(self, name)


    def _set_coords(self,coords):
        """Replace the current coords with new ones.

        Returns a Mesh or subclass exactly like the current except
        for the position of the coordinates.
        """
        if isinstance(coords,Coords) and coords.shape == self.coords.shape:
            return self.__class__(coords,self.elems,prop=self.prop,eltype=self.elType())
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
        # For compatibility reasons, the eltype is set as an attribute
        # of both the Mesh and the Mesh.elems attribute
        if eltype is None and hasattr(self.elems,'eltype'):
            eltype = self.elems.eltype
        self.eltype = self.elems.eltype = elementType(eltype,self.nplex())
        return self


    def elType(self):
        """Return the element type of the Mesh.

        """
        if self.elems.eltype is not None:
            return self.elems.eltype
        else:
            return self.eltype


    def elName(self):
        """Return the element name of the Mesh.

        """
        return self.elType().name()
    

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
        """Return element i of the Mesh.

        This allows addressing element i of Mesh M as M[i].
        The return value is an array with the coordinates of all the points
        of the element. M[i][j] then will return the coordinates of node j
        of element i.
        This also allows to change the individual coordinates or nodes, by
        an assignment like  M[i][j] = [1.,0.,0.].
        """
        return self.coords[self.elems[i]]
    

    def __setitem__(self,i,val):
        """Change element i of the Mesh.

        This allows changing all the coordinates of an element by direct
        assignment such as M[i] = [[1.,0.,0.], ...]. The user should make sure
        that the data match the plexitude of the element. 
        """
        self.coords[i] = val


    def __getstate__(self):
        import copy
        state = copy.copy(self.__dict__)
        # Store the element type by name,
        # This is needed because of the way ElementType is initialized
        # Maybe we should change that.
        # The setstate then needs to set the elementType
        # And this needs also to be done in Connectivity, if it has an eltype
        try:
            state['eltype'] = state['eltype'].name()
        except:
            state['eltype'] = None
        return state


    def __setstate__(self,state):
        """Set the object from serialized state.
        
        This allows to read back old pyFormex Project files where the Mesh
        class did not set element type yet.
        """
        if 'eltype' in state:
            state['eltype'] = elementType(state['eltype'])
            self.__dict__.update(state)
        else:
            # Old Project files did not save element type
            self.setType(self.eltype)
    

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
        return self.__class__(self.coords,self.elems,prop=self.prop,eltype=self.elType())


    def toFormex(self):
        """Convert a Mesh to a Formex.

        The Formex inherits the element property numbers and eltype from
        the Mesh. Node property numbers however can not be translated to
        the Formex data model.
        """
        return Formex(self.coords[self.elems],self.prop,self.elName())


    def toSurface(self):
        """Convert a Mesh to a Surface.

        If the plexitude of the mesh is 3, returns a TriSurface equivalent
        with the Mesh. Else, an error is raised.
        """
        from plugins.trisurface import TriSurface
        if self.nplex() == 3:
            return TriSurface(self)
        else:
            raise ValueError,"Only plexitude-3 Meshes can be converted to TriSurface. Got plexitude %s" % self.nplex()

            
    def ndim(self):
        return 3
    def level(self):
        return self.elType().ndim
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
            return self.nelems() * self.elType().nedges()
        except:
            return 0


    def info(self):
        """Return short info about the Mesh.

        This includes only the shape of the coords and elems arrays.
        """
        return "coords" + str(self.coords.shape) + "; elems" + str(self.elems.shape)

    def report(self,full=True):
        """Create a report on the Mesh shape and size.

        The report always contains the number of nodes, number of elements,
        plexitude, dimensionality, element type, bbox and size.
        If full==True(default), it also contains the nodal coordinate
        list and element connectivity table. Because the latter can be rather
        bulky, they can be switched off. (Though numpy will limit the printed
        output).

        TODO: We should add an option here to let numpy print the full tables.
        """
        bb = self.bbox()
        s = """
Mesh: %s nodes, %s elems, plexitude %s, ndim %s, eltype: %s
  BBox: %s, %s
  Size: %s
""" % (self.ncoords(),self.nelems(),self.nplex(),self.level(),self.elName(),bb[0],bb[1],bb[1]-bb[0])

        if full:
            s += "Coords:\n" + self.coords.__str__() +  "\nElems:\n" + self.elems.__str__()
        return s


    def __str__(self):
        """Format a Mesh in a string.

        This creates a detailed string representation of a Mesh,
        containing the report() and the lists of nodes and elements.
        """
        return self.report(False)


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


#######################################################################
    ## Entity selection and mesh traversal ##


    @deprecation("Mesh.getLowerEntitiesSelector is deprecated. Use Element.getEntities instead.")
    def getLowerEntitiesSelector(self,level=-1):
        """Get the entities of a lower dimensionality.

        """
        return self.elType().getEntities(level)



    # BV: This should probably be removed,
    # If needed, add a unique=False to Connectivity.insertLEvel 
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
        sel = self.elType().getEntities(level)
        ent = self.elems.selectNodes(sel)
        ent.eltype = sel.eltype
        if unique:
            warn("depr_mesh_getlowerentities_unique")
            ent = ent.removeDuplicate()

        return ent


    def getNodes(self):
        """Return the set of unique node numbers in the Mesh.

        This returns only the node numbers that are effectively used in
        the connectivity table. For a compacted Mesh, it is equivalent to
        ``arange(self.nelems)``.
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
        edges. It is equivalent to ``self.getLowerEntities(1,unique=True)``,
        but this also stores the result internally so that future
        requests can return it without the need for computing it again.
        """
        if self.edges is None:
            self.edges = self.elems.insertLevel(1)[1]
        return self.edges
        

    def getFaces(self):
        """Return the unique faces of all the elements in the Mesh.

        This is a convenient function to create a table with the element
        faces. It is equivalent to ``self.getLowerEntities(2,unique=True)``,
        but this also stores the result internally so that future
        requests can return it without the need for computing it again.
        """
        if self.faces is None:
            self.faces = self.elems.insertLevel(2)[1]
        return self.faces
        

    def getCells(self):
        """Return the cells of the elements.

        This is a convenient function to create a table with the element
        cells. It is equivalent to ``self.getLowerEntities(3,unique=True)``,
        but this also stores the result internally so that future
        requests can return it without the need for computing it again.
        """
        if self.cells is None:
            self.cells = self.elems.insertLevel(3)[1]
        return self.cells


    def getElemEdges(self):
        """Defines the elements in function of its edges.

        This returns a Connectivity table with the elements defined in
        function of the edges. It returns the equivalent of
        ``self.elems.insertLevel(self.elType().getEntities(1))``
        but as a side effect it also stores the definition of the edges
        and the returned element to edge connectivity in the attributes
        `edges`, resp. `elem_edges`.
        """
        if self.elem_edges is None:
            self.elem_edges,self.edges = self.elems.insertLevel(1)
        return self.elem_edges

    
    def getFreeEntities(self,level=-1,return_indices=False):
        """Return the border of the Mesh.

        Returns a Connectivity table with the free entities of the
        specified level of the Mesh. Free entities are entities
        that are only connected with a single element.

        If return_indices==True, also returns an (nentities,2) index
        for inverse lookup of the higher entity (column 0) and its local
        lower entity number (column 1).
        """
        hi,lo = self.elems.insertLevel(level)
        if hi.size == 0:
            if return_indices:
                return Connectivity(),[]
            else:
                return Connectivity()
            
        hiinv = hi.inverse()
        ncon = (hiinv>=0).sum(axis=1)
        isbrd = (ncon<=1)   # < 1 should not occur 
        brd = lo[isbrd]
        #
        # WE SET THE eltype HERE, BECAUSE THE INDEX OPERATION ABOVE
        # LOOSES THE eltype
        # SHOULD BE FIXED, BUT NEEDS TO BE CHECKED !!! BV
        #  ### BV uncommented for checking, report if not working ###
        # brd.eltype = sel.eltype
        if brd.eltype is None:
            raise ValueError,"THIS ERROR SOHULD NOT OCCUR! CONTACT MAINTAINERS!"
        if not return_indices:
            return brd
        
        # return indices where the border elements come from
        binv = hiinv[isbrd]
        enr = binv[binv >= 0]  # element number
        a = hi[enr]
        b = arange(lo.shape[0])[isbrd].reshape(-1,1)
        fnr = where(a==b)[1]   # local border part number
        return brd,column_stack([enr,fnr])


    def getFreeEntitiesMesh(self,level=-1,compact=True):
        """Return a Mesh with lower entities.

        Returns a Mesh representing the lower entities of the specified
        level. If the Mesh has property numbers, the lower entities inherit
        the property of the element to which they belong.

        By default, the resulting Mesh is compacted. Compaction can be
        switched off by setting `compact=False`.
        """
        if self.prop==None:
            M = Mesh(self.coords,self.getFreeEntities(level=level))

        else:
            brd,indices = self.getFreeEntities(return_indices=True,level=level)
            enr = indices[:,0]
            M = Mesh(self.coords,brd,prop=self.prop[enr])
            # THIS SEEMS SUPERFLUOUS
            # M.setType(brd.eltype)

        if compact:
            M = M.compact()
        return M


    def getBorder(self,return_indices=False):
        """Return the border of the Mesh.

        This returns a Connectivity table with the border of the Mesh.
        The border entities are of a lower hierarchical level than the
        mesh itself. These entities become part of the border if they
        are connected to only one element.

        If return_indices==True, it returns also an (nborder,2) index
        for inverse lookup of the higher entity (column 0) and its local
        border part number (column 1).

        This is a convenient shorthand for ::

          self.getFreeEntities(level=-1,return_indices=return_indices)
        """
        return self.getFreeEntities(level=-1,return_indices=return_indices)


    def getBorderMesh(self,compact=True):
        """Return a Mesh with the border elements.

        The returned Mesh is of the next lower hierarchical level and
        contains all the free entitites of that level.
        If the Mesh has property numbers, the border elements inherit
        the property of the element to which they belong.

        By default, the resulting Mesh is compacted. Compaction can be
        switched off by setting `compact=False`.

        This is a convenient shorthand for ::

          self.getFreeEntitiesMesh(level=-1,compact=compact)
        """
        return self.getFreeEntitiesMesh(level=-1,compact=compact)


    def getBorderElems(self):
        """Return the elements that are on the border of the Mesh.

        This returns a list with the numbers of the elements that are on the
        border of the Mesh. Elements are considered to be at the border if they
        contain at least one complete element of the border Mesh (i.e. an
        element of the first lower hierarchical level). Thus, in a volume Mesh,
        elements only touching the border by a vertex or an edge are not
        considered border elements.
        """
        brd,ind = self.getBorder(True)
        return unique(ind[:,0])


    def getBorderNodes(self):
        """Return the nodes that are on the border of the Mesh.

        This returns a list with the numbers of the nodes that are on the
        border of the Mesh.
        """
        brd = self.getBorder()
        return unique(brd)


    def peel(self):
        """Return a Mesh with the border elements removed.

        This is a convenient shorthand for ::

          self.cselect(self.getBorderElems())
        """
        return self.cselect(self.getBorderElems())


    def getFreeEdgesMesh(self,compact=True):
        """Return a Mesh with the free edge elements.

        The returned Mesh is of the hierarchical level 1 (no mather what
        the level of the parent Mesh is) and contains all the free entitites
        of that level.
        If the Mesh has property numbers, the border elements inherit
        the property of the element to which they belong.

        By default, the resulting Mesh is compacted. Compaction can be
        switched off by setting `compact=False`.

        This is a convenient shorthand for ::

          self.getFreeEntitiesMesh(level=1,compact=compact)
        """
        return self.getFreeEntitiesMesh(level=1,compact=compact)


#############################################################################
    # Adjacency #

    def adjacency(self,level=0,diflevel=-1):
        """Create an element adjacency table.

        Two elements are said to be adjacent if they share a lower
        entity of the specified level.
        The level is one of the lower entities of the mesh.

        Parameters:

        - `level`: hierarchy of the geometric items connecting two elements:
          0 = node, 1 = edge, 2 = face. Only values of a lower hierarchy than
          the elements of the Mesh itself make sense.
        - `diflevel`: if >= level, and smaller than the hierarchy of
          self.elems, elements that have a connection of this level are removed.
          Thus, in a Mesh with volume elements, self.adjacency(0,1) gives the
          adjacency of elements by a node but not by an edge.

        Returns an Adjacency with integers specifying for each element
        its neighbours connected by the specified geometrical subitems.
        """
        if diflevel > level:
            return self.adjacency(level).diff(self.adjacency(diflevel))

        if level == 0:
            elems = self.elems
        else:
            elems,lo = self.elems.insertLevel(level)
        return elems.adjacency()


    def frontWalk(self,level=0,startat=0,frontinc=1,partinc=1,maxval=-1):
        """Visit all elements using a frontal walk.

        In a frontal walk a forward step is executed simultanuously from all
        the elements in the current front. The elements thus reached become
        the new front. An element can be reached from the current element if
        both are connected by a lower entity of the specified level. Default
        level is 'point'.

        Parameters:

        - `level`: hierarchy of the geometric items connecting two elements:
          0 = node, 1 = edge, 2 = face. Only values of a lower hierarchy than
          the elements of the Mesh itself make sense. There are no
          connections on the upper level.

        The remainder of the parameters are like in
        :meth:`Connectivity.frontWalk`.

        Returns an array of integers specifying for each element in which step
        the element was reached by the walker.
        """
        return self.adjacency(level).frontWalk(startat=startat,frontinc=frontinc,partinc=partinc,maxval=maxval)
    

    def maskedEdgeFrontWalk(self,mask=None,startat=0,frontinc=1,partinc=1,maxval=-1):
        """Perform a front walk over masked edge connections.

        This is like frontWalk(level=1), but allows to specify a mask to
        select the edges that are used as connectors between elements.

        Parameters:

        - `mask`: Either None or a boolean array or index flagging the nodes
          which are to be considered connectors between elements. If None,
          all nodes are considered connections.

        The remainder of the parameters are like in
        :meth:`Connectivity.frontWalk`.
        """
        hi,lo = self.elems.insertLevel(1)
        adj = hi.adjacency(mask=mask)
        return adj.frontWalk(startat=startat,frontinc=frontinc,partinc=partinc,maxval=maxval)


    # BV: DO WE NEED THE nparts ?
    def partitionByConnection(self,level=0,startat=0,sort='number',nparts=-1):
        """Detect the connected parts of a Mesh.

        The Mesh is partitioned in parts in which all elements are
        connected. Two elements are connected if it is possible to draw a
        continuous (poly)line from a point in one element to a point in
        the other element without leaving the Mesh.
        The partitioning is returned as a integer array having a value
        for ech element corresponding to the part number it belongs to.
       
        By default the parts are sorted in decreasing order of the number
        of elements. If you specify nparts, you may wish to switch off the
        sorting by specifying sort=''.
        """
        p = self.frontWalk(level=level,startat=startat,frontinc=0,partinc=1,maxval=nparts)
        if sort=='number':
            p = sortSubsets(p)
        #
        # TODO: add weighted sorting methods
        #
        return p


    def splitByConnection(self,level=0,startat=0,sort='number'):
        """Split the Mesh into connected parts.

        Returns a list of Meshes that each form a connected part.
        By default the parts are sorted in decreasing order of the number
        of elements. 
        """
        p = self.partitionByConnection(level=level,startat=startat,sort=sort)
        split = self.setProp(p).splitProp()
        if split:
            return split.values()
        else:
            return [ self ]


    def largestByConnection(self,level=0):
        """Return the largest connected part of the Mesh.

        This is equivalent with, but more efficient than ::

          self.splitByConnection(level)[0]
        """
        p = self.partitionByConnection(level=level)
        return self.withProp(0)


    def growSelection(self,sel,mode='node',nsteps=1):
        """Grow a selection of a surface.

        `p` is a single element number or a list of numbers.
        The return value is a list of element numbers obtained by
        growing the front `nsteps` times.
        The `mode` argument specifies how a single frontal step is done:

        - 'node' : include all elements that have a node in common,
        - 'edge' : include all elements that have an edge in common.
        """
        level = {'node':0,'edge':1}[mode]
        p = self.frontWalk(level=level,startat=sel,maxval=nsteps)
        return where(p>=0)[0]  


###########################################################################

    #
    #  IDEA: Should we move these up to Connectivity ?
    #        That would also avoid some possible problems 
    #        with storing conn and econn
    #

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


    #
    # Are these really needed? better use adjacency(level)
    # 
    #
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


    def nonManifoldNodes(self):
        """Return the non-manifold nodes of a Mesh.

        Non-manifold nodes are nodes where subparts of a mesh of level >= 2
        are connected by a node but not by an edge.

        Returns an integer array with a sorted list of non-manifold node
        numbers. Possibly empty (always if the dimensionality of the Mesh
        is lower than 2). 
        """
        if self.level() < 2:
            return []
                
        ML = self.splitByConnection(1,sort='')
        nm = [ intersect1d(Mi.elems,Mj.elems) for Mi,Mj in combinations(ML,2) ]
        return unique(concat(nm))


    def nonManifoldEdges(self):
        """Return the non-manifold edges of a Mesh.

        Non-manifold edges are edges where subparts of a mesh of level 3
        are connected by an edge but not by an face.

        Returns an integer array with a sorted list of non-manifold edge
        numbers. Possibly empty (always if the dimensionality of the Mesh
        is lower than 3).

        As a side effect, this constructs the list of edges in the object.
        The definition of the nonManifold edges in tgerms of the nodes can
        thus be got from ::

          self.edges[self.nonManifoldEdges()]
        """
        if self.level() < 3:
            return []

        elems = self.getElemEdges() 
        p = self.partitionByConnection(2,sort='')
        eL = [ elems[p==i] for i in unique(p) ] 
        nm = [ intersect1d(ei,ej) for ei,ej in combinations(eL,2) ]
        return unique(concat(nm))


    def nonManifoldEdgeNodes(self):
        """Return the non-manifold edge nodes of a Mesh.

        Non-manifold edges are edges where subparts of a mesh of level 3
        are connected by an edge but not by an face.

        Returns an integer array with a sorted list of numbers of nodes
        on the non-manifold edges.
        Possibly empty (always if the dimensionality of the Mesh
        is lower than 3). 
        """
        if self.level() < 3:
            return []
                
        ML = self.splitByConnection(2,sort='')
        nm = [ intersect1d(Mi.elems,Mj.elems) for Mi,Mj in combinations(ML,2) ]
        return unique(concat(nm))


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


    def fuse(self,**kargs):
        """Fuse the nodes of a Meshes.

        All nodes that are within the tolerance limits of each other
        are merged into a single node.  

        The merging operation can be tuned by specifying extra arguments
        that will be passed to :meth:`Coords:fuse`.
        """
        coords,index = self.coords.fuse(**kargs)
        return self.__class__(coords,index[self.elems],prop=self.prop,eltype=self.elType())


    def matchCoords(self,mesh,**kargs):
        """Match nodes of Mesh with nodes of self.

        This is a convenience function equivalent to::

           self.coords.match(mesh.coords,**kargs)

        See also :meth:`Coords.match`
        """
        return self.coords.match(mesh.coords,**kargs)
        
    
    def matchCentroids(self, mesh,**kargs):
        """Match elems of Mesh with elems of self.
        
        self and Mesh are same eltype meshes
        and are both without duplicates.
        
        Elems are matched by their centroids.
        """
        c = Mesh(self.centroids(), arange(self.nelems() ))
        mc = Mesh(mesh.centroids(), arange(mesh.nelems() ))
        return c.matchCoords(mc,**kargs)


    # BV: I'm not sure that we need this. Looks like it can or should
    # be replaced with a method applied on the BorderMesh
    #~ FI It has been tested on quad4-quad4, hex8-quad4, tet4-tri3
    def matchFaces(self,mesh):
        """Match faces of mesh with faces of self.
            
        self and Mesh can be same eltype meshes or different eltype but of the 
        same hierarchical type (i.e. hex8-quad4 or tet4 - tri3) 
        and are both without duplicates.
            
        Returns the indices array of the elems of self that matches
        the faces of mesh
        """
        sel = self.elType().getEntities(2)
        hi,lo = self.elems.insertLevel(sel)
        hiinv = hi.inverse()
        fm=Mesh(self.coords,self.getFaces())
        mesh=Mesh(mesh.coords,mesh.getFaces())
        c=fm.matchCentroids(mesh)
        hiinv = hiinv[c]
        enr =  unique(hiinv[hiinv >= 0])  # element number
        return enr
    

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
            return self.__class__(coords,elems,prop=self.prop,eltype=self.elType())
        
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
        M = self.__class__(self.coords,self.elems[selected],eltype=self.elType())
        if self.prop is not None:
            M.setProp(self.prop[selected])
        if compact:
            M = M.compact()
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
        return self.select(complement(selected,self.nelems()),compact=compact)


    def avgNodes(self,nodsel,wts=None):
        """Create average nodes from the existing nodes of a mesh.

        `nodsel` is a local node selector as in :meth:`selectNodes`
        Returns the (weighted) average coordinates of the points in the
        selector as `(nelems*nnod,3)` array of coordinates, where
        nnod is the length of the node selector.
        `wts` is a 1-D array of weights to be attributed to the points.
        Its length should be equal to that of nodsel.
        """
        elems = self.elems.selectNodes(nodsel)
        return self.coords[elems].average(wts=wts,axis=1)


    # The following is equivalent to avgNodes(self,nodsel,wts=None)
    # But is probably more efficient
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
            return self.__class__(self.coords,self.elems,eltype=self.elType())
        elif array(val).size == 1:
            return self.__class__(self.coords,self.elems[self.prop==val],prop=val,eltype=self.elType())
        else:
            t = zeros(self.prop.shape,dtype=bool)
            for v in asarray(val).flat:
                t += (self.prop == v)
            return self.__class__(self.coords,self.elems[t],prop=self.prop[t],eltype=self.elType())
        
    
    def withoutProp(self, val):
        """Return a Mesh without the elements with property `val`.

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


    def connectedTo(self,nodes):
        """Return a Mesh with the elements connected to the specified node(s).

        `nodes`: int or array_like, int.

        Returns a Mesh with all the elements from the original that contain
        at least one of the specified nodes.
        """
        return self.select(self.elems.connectedTo(nodes))


    def notConnectedTo(self, nod):
        """Return a Mesh with the elements not connected to the given node(s).

        `nodes`: int or array_like, int.

        Returns a Mesh with all the elements from the original that do not
        contain any of the specified nodes.
        """
        return self.select(self.elems.notConnectedTo(nod))


    def splitProp(self):
        """Partition a Mesh according to its property values.

        Returns a dict with the property values as keys and the
        corresponding partitions as values. Each value is a Mesh instance.
        It the Mesh has no props, an empty dict is returned.
        """
        if self.prop is None:
            return {}
        else:
            return dict([(p,self.withProp(p)) for p in self.propSet()])


    def splitRandom(self,n,compact=True):
        """Split a Mesh in n parts, distributing the elements randomly.

        Returns a list of n Mesh objects, constituting together the same
        Mesh as the original. The elements are randomly distributed over
        the subMeshes.

        By default, the Meshes are compacted. Compaction may be switched
        off for efficiency reasons.
        """
        sel = random.randint(0,n,(self.nelems()))
        return [ self.select(sel==i,compact=compact) for i in range(n) if i in sel ]


###########################################################################
    ## simple mesh transformations ##

    def reverse(self):
        """Return a Mesh where all elements have been reversed.

        Reversing an element has the following meaning:

        - for 1D elements: reverse the traversal direction,
        - for 2D elements: reverse the direction of the positive normal,
        - for 3D elements: reverse inside and outside directions of the
          element's border surface

        The :meth:`reflect` method by default calls this method to undo
        the element reversal caused by the reflection operation. 
        """
        utils.warn('warn_mesh_reverse')

        if hasattr(self.elType(),'reversed'):
            elems = self.elems[:,self.elType().reversed]
        else:
            elems = self.elems[:,::-1]
        return self.__class__(self.coords,elems,prop=self.prop,eltype=self.elType())

            
    def reflect(self,dir=0,pos=0.0,reverse=True,**kargs):
        """Reflect the coordinates in one of the coordinate directions.

        Parameters:

        - `dir`: int: direction of the reflection (default 0)
        - `pos`: float: offset of the mirror plane from origin (default 0.0)
        - `reverse`: boolean: if True, the :meth:`Mesh.reverse` method is
          called after the reflection to undo the element reversal caused
          by the reflection of its coordinates. This will in most cases have
          the desired effect. If not however, the user can set this to False
          to skip the element reversal.
        """
        if 'autofix' in kargs:
            utils.warn("The `autofix` parameter of Mesh.reflect has been renamed to `reverse`.")
            reverse = kargs['autofix']
            
        if reverse is None:
            reverse = True
            utils.warn("warn_mesh_reflect")
        
        M = Geometry.reflect(self,dir=dir,pos=pos)
        if reverse:
            M = M.reverse()
        return M
    

    def convert(self,totype,fuse=False):
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

        .. warning:: Conversion strategies that add new nodes may produce
          double nodes at the common border of elements. The :meth:`fuse`
          method can be used to merge such coincident nodes. Specifying
          fuse=True will also enforce the fusing. This option become the
          default in future.
        """
        #
        # totype is a string !
        #
        
        if elementType(totype) == self.elType():
            return self
        
        strategy = self.elType().conversions.get(totype,None)

        while not type(strategy) is list:
            # This allows for aliases in the conversion database
            strategy = self.elType().conversions.get(strategy,None)
            
            if strategy is None:
                raise ValueError,"Don't know how to convert %s -> %s" % (self.elName(),totype)

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

            if steptype == 'a':
                mesh = mesh.addMeanNodes(stepdata,totype)
                
            elif steptype == 's':
                mesh = mesh.selectNodes(stepdata,totype)

            else:
                raise ValueError,"Unknown conversion step type '%s'" % steptype

        if fuse:
            mesh = mesh.fuse()
        return mesh


    def convertRandom(self,choices):
        """Convert choosing randomly between choices

        Returns a Mesh obtained by converting the current Mesh by a
        randomly selected method from the available conversion type
        for the current element type.
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


    ## TODO:
    ## - We should add prop inheritance here
    ## - mesh_wts and mesh_els functions should be moved to elements.py
    def subdivide(self,*ndiv,**kargs):
        """Subdivide the elements of a Mesh.

        Parameters:

        - `ndiv`: specifies the number (and place) of divisions (seeds)
          along the edges of the elements. Accepted type and value depend
          on the element type of the Mesh. Currently implemented:

        - 'tri3': ndiv is a single int value specifying the number of
          divisions (of equal size) for each edge.

        - 'quad4': ndiv is a sequence of two int values nx,ny, specifying
          the number of divisions along the first, resp. second
          parametric direction of the element

        - `fuse`: bool, if True (default), the resulting Mesh is completely
          fused. If False, the Mesh is only fused over each individual
          element of the original Mesh.

        Returns a Mesh where each element is replaced by a number of
        smaller elements of the same type.

        .. note:: This is currently only implemented for Meshes of type 'tri3'
          and 'quad4' and for the derived class 'TriSurface'.
        """
        elname = self.elName()
        try:
            mesh_wts = globals()[elname+'_wts']
            mesh_els = globals()[elname+'_els']
        except:
            raise ValueError,"Can not subdivide element of type '%s'" % elname

        wts = mesh_wts(*ndiv)
        els = mesh_els(*ndiv)
        X = self.coords[self.elems]
        U = dot(wts,X).transpose([1,0,2]).reshape(-1,3)
        e = concatenate([els+i*wts.shape[0] for i in range(self.nelems())])
        M = self.__class__(U,e,eltype=self.elType())
        if kargs.get('fuse',True):
            M = M.fuse()
        return M


    def reduceDegenerate(self,eltype=None):
        """Reduce degenerate elements to lower plexitude elements.

        This will try to reduce the degenerate elements of the mesh to elements
        of a lower plexitude. If a target element type is given, only the
        matching reduce scheme is tried.
        Else, all the target element types for which
        a reduce scheme from the Mesh eltype is available, will be tried.

        The result is a list of Meshes of which the last one contains the
        elements that could not be reduced and may be empty.
        Property numbers propagate to the children. 
        """
        #
        # This duplicates a lot of the functionality of
        # Connectivity.reduceDegenerate
        # But this is really needed to keep the properties
        #
        
        if self.nelems() == 0:
            return [self]
        
        try:
            strategies = self.elType().degenerate
        except:
            return [self]
        
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

            elems = []
            prop = []
            for conditions,selector in strategies[eltype]:
                e = m.elems
                cond = array(conditions)
                w = (e[:,cond[:,0]] == e[:,cond[:,1]]).all(axis=1)
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
        return self.select(~deg,compact=False)


    def removeDuplicate(self,permutations=True):
        """Remove the duplicate elements from a Mesh.

        Duplicate elements are elements that consist of the same nodes,
        by default in no particular order. Setting permutations=False will
        only consider elements with the same nodes in the same order as
        duplicates.
        
        Returns a Mesh with all duplicate elements removed.
        """
        ind,ok = self.elems.testDuplicate(permutations)
        return self.select(ind[ok])


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
        - 'random': the nodes are numbered randomly.
        - 'front': the nodes are numbered in order of their frontwalk.
        """
        if order == 'elems':
            order = renumberIndex(self.elems)
        elif order == 'random':
            order = arange(self.nnodes())
            random.shuffle(order)
        elif order == 'front':
            adj = self.elems.adjacency('n')
            p = adj.frontWalk()
            order = p.argsort()
        newnrs = inverseUniqueIndex(order)
        return self.__class__(self.coords[order],newnrs[self.elems],prop=self.prop,eltype=self.elType())


    def reorder(self,order='nodes'):
        """Reorder the elements of a Mesh.

        Parameters:

        - `order`: either a 1-D integer array with a permutation of
          ``arange(self.nelems())``, specifying the requested order, or one of
          the following predefined strings:

          - 'nodes': order the elements in increasing node number order.
          - 'random': number the elements in a random order.
          - 'reverse': number the elements in reverse order. 

        Returns a Mesh equivalent with self but with the elements ordered as specified.

        See also: :meth:`Connectivity.reorder`
        """
        order = self.elems.reorder(order)
        return self.__class__(self.coords,self.elems[order],prop=self.prop[order],eltype=self.elType())


    # for compatibility:
    renumberElems = reorder

##############################################################
    #
    # Connection, Extrusion, Sweep, Revolution
    #

    def connect(self,coordslist,div=1,degree=1,loop=False,eltype=None):
        """Connect a sequence of toplogically congruent Meshes into a hypermesh.

        Parameters:

        - `coordslist`: either a list of Coords instances, all having the same
          shape as self.coords, or a single Mesh instance whose `coords`
          attribute has the same shape.

          If it is a list of Coords, consider a
          list of Meshes obtained by combining each Coords object with the
          connectivity table, element type and property numbers of the current
          Mesh. The return value then is the hypermesh obtained by connecting
          each consecutive slice of (degree+1) of these meshes. The hypermesh
          has a dimensionality that is one higher than the original Mesh (i.e.
          points become lines, lines become surfaces, surfaces become volumes).
          The resulting elements will be of the given `degree` in the
          direction of the connection. Notice that the coords of the current
          Mesh are not used, unless these coords are explicitely included into
          the specified `coordslist`. In many cases `self.coords` will be the
          first item in `coordslist`, but it could occur anywhere in the list
          or even not at all. The number of Coords items in the list should
          be a multiple of `degree` plus 1.

          Specifying a single Mesh instead of a list of Coords is
          just a convenience for the often occurring situation of connecting
          a Mesh (self) with another one (mesh) having the same connectivity:
          in this case the list of Coords will automatically be set to
          ``[ self.coords, mesh.coords ]``. The `degree` should be 1 in this
          case.

        - `degree`: degree of the connection. Currently only degree 1 and 2
          are supported.

          - If degree is 1, every Coords from the `coordslist`
            is connected with hyperelements of a linear degree in the
            connection direction.

          - If degree is 2, quadratic hyperelements are
            created from one Coords item and the next two in the list.
            Note that all Coords items should contain the same number of nodes,
            even for higher order elements where the intermediate planes
            contain less nodes.

        - `loop`: if True, the connections with loop around the list and
          connect back to the first. This is accomplished by adding the first
          Coords item back at the end of the list.

        - `div`: Either an integer, or a sequence of float numbers (usually
          in the range ]0.0..1.0]). This should only
          be used for degree==1.

          With this parameter the generated elements
          can be further subdivided along the connection direction.
          If an int is given, the connected elements will be divided
          into this number of elements along the connection direction. If a
          sequence of float numbers is given, the numbers specify the relative
          distance along the connection direction where the elements should
          end. If the last value in the sequence is not equal to 1.0, there
          will be a gap between the consecutive connections. 

        - `eltype`: the element type of the constructed hypermesh. Normally,
          this is set automatically from the base element type and the
          connection degree. If a different element type is specified,
          a final conversion to the requested element type is attempted.
        """
        if type(coordslist) is list:
            clist = coordslist
        elif isinstance(coordslist,Mesh):
            utils.warn("warn_mesh_connect")
            clist = [ self.coords, coordslist.coords ]
            if degree == 2:
                raise ValueError,"This only works for linear connection"
            ##     xm = 0.5 * (clist[0]+clist[1])
            ##     clist.insert(1, xm) 
        else:
            raise ValueError,"Invalid coordslist argument"

        if sum([c.shape != self.coords.shape for c in clist]):
            raise ValueError,"Incompatible coordinate sets"

        # implement loop parameter
        if loop:
            clist.append(clist[0])

        if (len(clist)-1) % degree != 0:
            raise ValueError,"Invalid length of coordslist (%s) for degree %s." % (len(clist),degree)

        # set divisions
        ## div = unitDivisor(degree*div,start=1)
        div = unitDivisor(div,start=1)

        # For higher order non-lagrangian elements the procedure could be
        # optimized by first compacting the coords and elems.
        # Instead we opted for the simpler method of adding the maximum
        # number of nodes, and then selecting the used ones.
        # A final compact() throws out the unused points.

        # Concatenate the coordinates
        x = [ Coords.interpolate(xi,xj,div).reshape(-1,3) for xi,xj in zip(clist[:-1:degree],clist[degree::degree]) ]
        x = Coords.concatenate(clist[:1] + x)
        
        # Create the connectivity table
        nnod = self.ncoords()
        nrep = (x.shape[0]//nnod - 1) // degree
        e = extrudeConnectivity(self.elems,nnod,degree)
        e = replicConnectivity(e,nrep,nnod*degree)
        
        # Create the Mesh
        M = Mesh(x,e).setProp(self.prop)
        # convert to proper eltype
        if eltype:
            M = M.convert(eltype)
        return M


    # deprecated in 0.8.5
    def connectSequence(self,coordslist,div=1,degree=1,loop=False,eltype=None):
        utils.deprec("Mesh.connectSequence is deprecated: use Mesh.connect instead")
        return self.connect(coordslist,div=div,degree=degree,loop=loop,eltype=eltype)


    def extrude(self,n,step=1.,dir=0,degree=1,eltype=None):
        """Extrude a Mesh in one of the axes directions.

        Returns a new Mesh obtained by extruding the given Mesh
        over `n` steps of length `step` in direction of axis `dir`.
          
        """
        print "Extrusion over %s steps of length %s" % (n,step)
        x = [ self.coords.trl(dir,i*n*step/degree) for i in range(1,degree+1) ]
        print bbox(x)
        return self.connect([self.coords] + x,n*degree,degree=degree,eltype=eltype)
        #return self.connect(self.trl(dir,n*step),n*degree,degree=degree,eltype=eltype)


    def revolve(self,n,axis=0,angle=360.,around=None,loop=False,eltype=None):
        """Revolve a Mesh around an axis.

        Returns a new Mesh obtained by revolving the given Mesh
        over an angle around an axis in n steps, while extruding
        the mesh from one step to the next.
        This extrudes points into lines, lines into surfaces and surfaces
        into volumes.
        """
        angles = arange(n+1) * angle / n
        seq = [ self.coords.rotate(angle=a,axis=axis,around=around) for a in angles ]
        return self.connect(seq,loop=loop,eltype=eltype)


    def sweep(self,path,eltype=None,**kargs):
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

        Currently, this function produces the correct element type, but
        the geometry .
        """
        seq = sweepCoords(self.coords,path,**kargs)
        return self.connect(seq,eltype=eltype)


    def smooth(self, iterations=1, lamb=0.5, k=0.1, edg=True, exclnod=[], exclelem=[]):
        """Return a smoothed mesh.
        
        Smoothing algorithm based on lowpass filters.
        
        If edg is True, the algorithm tries to smooth the
        outer border of the mesh seperately to reduce mesh shrinkage.
        
        Higher values of k can reduce shrinkage even more
        (up to a point where the mesh expands),
        but will result in less smoothing per iteration.
        
        - `exclnod`: It contains a list of node indices to exclude from the smoothing.
          If exclnod is 'border', all nodes on the border of the mesh will
          be unchanged, and the smoothing will only act inside.
          If exclnod is 'inner', only the nodes on the border of the mesh will
          take part to the smoothing.
        
        - `exclelem`: It contains a list of elements to exclude from the smoothing.
          The nodes of these elements will not take part to the smoothing.
          If exclnod and exclelem are used at the same time the union of them
          will be exluded from smoothing.
        """
        if iterations < 1: 
            return self
        
        if lamb*k == 1:
            raise ValueError,"Cannot assign values of lamb and k which result in lamb*k==1"
        
        mu = -lamb/(1-k*lamb)
        adj = self.getEdges().adjacency(kind='n')
        incl = resize(True, self.ncoords())
        
        if exclnod == 'border':
            exclnod = unique(self.getBorder())
            k = 0. #k can be zero because it cannot shrink
            edg = False #there is no border edge
        if exclnod == 'inner':
            exclnod = delete(arange(self.ncoords()), unique(self.getBorder()))
        exclelemnod = unique(self.elems[exclelem])
        exclude=array(unique(concatenate([exclnod, exclelemnod])), dtype = int)

        incl[exclude] = False
        
        if edg:
            externals = resize(False,self.ncoords())
            expoints = unique(self.getFreeEntities())
            if len(expoints) != self.ncoords():
                externals[expoints] = True
                a = adj[externals].ravel()
                inpoints = delete(range(self.ncoords()), expoints)
                for i in range(len(a)):
                    if a[i] in inpoints:
                        a[i]=-2
                adj[externals] = a.reshape(adj[externals].shape)
            else:
                message('Failed to recognize external points.\nShrinkage may be considerable.')
        w = ones(adj.shape,dtype=float)
        w[adj<0] = 0.
        w /= (adj>=0).sum(-1).reshape(-1,1)
        w = w.reshape(adj.shape[0],adj.shape[1],1)
        c = self.coords.copy()
        for i in range(iterations):
            c[incl] = (1.-lamb)*c[incl] + lamb*(w[incl]*c[adj][incl]).sum(1)
            c[incl] = (1.-mu)*c[incl] + mu*(w[incl]*c[adj][incl]).sum(1)
        return self.__class__(c, self.elems, prop=self.prop, eltype=self.elType())


    def __add__(self,other):
        """Return the sum of two Meshes.

        The sum of the Meshes is simply the concatenation thereof.
        It allows us to write simple expressions as M1+M2 to concatenate
        the Meshes M1 and M2. Both meshes should be of the same plexitude
        and have the same eltype.
        The result will be of the same class as self (either a Mesh or a
        subclass thereof).
        """
        return self.__class__.concatenate([self,other])
    

    @classmethod
    def concatenate(clas,meshes,**kargs):
        """Concatenate a list of meshes of the same plexitude and eltype

        All Meshes in the list should have the same plexitude.
        Meshes with plexitude are ignored though, to allow empty
        Meshes to be added in.
        
        Merging of the nodes can be tuned by specifying extra arguments
        that will be passed to :meth:`Coords:fuse`.

        If any of the meshes has property numbers, the resulting mesh will
        inherit the properties. In that case, any meshes without properties
        will be assigned property 0.
        If all meshes are without properties, so will be the result.

        This is a class method, and should be invoked as follows::

          Mesh.concatenate([mesh0,mesh1,mesh2])
        """
        def _force_prop(m):
            if m.prop is None:
                return zeros(m.nelems(),dtype=Int)
            else:
                return m.prop
            
        meshes = [ m for m in meshes if m.nplex() > 0 ]
        nplex = set([ m.nplex() for m in meshes ])
        if len(nplex) > 1:
            raise ValueError,"Cannot concatenate meshes with different plexitude: %s" % str(nplex)
        eltype = set([ m.eltype for m in meshes if m.eltype is not None ])
        if len(eltype) > 1:
            raise ValueError,"Cannot concatenate meshes with different eltype: %s" % [ e.name() for e in eltype ]
        if len(eltype) == 1:
            eltype = eltype.pop()
        else:
            eltype = None

        # Keep the available props
        prop = [m.prop for m in meshes if m.prop is not None]
        if len(prop) == 0:
            prop = None
        elif len(prop) < len(meshes):
            prop = concatenate([_force_prop(m) for m in meshes])
        else:
            prop = concatenate(prop)
            
        coords,elems = mergeMeshes(meshes,**kargs)
        elems = concatenate(elems,axis=0)
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

        if array(dir).size == 1:
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

        if len(T.shape) > 1:
            # We have results for more than 1 node per element
            if nodes == 'any':
                T = T.any(axis=1)
            elif nodes == 'none':
                T = ~T.any(axis=1)
            else:
                T = T.all(axis=1)

        return asarray(T)


    def clip(self,t,compact=True):
        """Return a Mesh with all the elements where t>0.

        t should be a 1-D integer array with length equal to the number
        of elements of the Mesh.
        The resulting Mesh will contain all elements where t > 0.
        """
        return self.select(t>0,compact=compact)


    def cclip(self,t,compact=True):
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

        Computes the volume of Mesh of dimensionality 3 by converting it
        first to a 'tet4' type Mesh. If the Mesh dimensionality is less than 3
        or the Mesh can not be converted to type 'tet4', 0.0 is returned.
        """
        try:
            return self.volumes().sum()
        except:
            return 0.0
    

    def actor(self,**kargs):

        if self.nelems() == 0:
            return None
        
        from gui.actors import GeomActor
        return GeomActor(self,**kargs)



    #
    #  NEEDS CLEANUP BEFORE BEING REINSTATED
    #  =====================================
    #

    # BV: Should be made dependent on getFaces
    # or become a function??
    # ?? DOES THIS WORK FOR *ANY* MESH ??
    # What with a mesh of points, lines, ...
    # Also, el.faces can contain items of different plexitude
    # Maybe define this for 2D meshes only?
    # Formulas for both linear and quadratic edges
    # For quadratic: Option to select tangential or chordal directions
    #
    #### Currently reinstated in trisurface.py ! 
    
    ## def getAngles(self, angle_spec=Deg):
    ##     """_Returns the angles in Deg or Rad between the edges of a mesh.
        
    ##     The returned angles are shaped  as (nelems, n1faces, n1vertices),
    ##     where n1faces are the number of faces in 1 element and the number
    ##     of vertices in 1 face.
    ##     """
    ##     #mf = self.coords[self.getFaces()]
    ##     mf = self.coords[self.getLowerEntities(2,unique=False)]
    ##     v = mf - roll(mf,-1,axis=1)
    ##     v=normalize(v)
    ##     v1=-roll(v,+1,axis=1)
    ##     angfac= arccos( clip(dotpr(v, v1), -1., 1. ))/angle_spec
    ##     el = self.elType()
    ##     return angfac.reshape(self.nelems(),len(el.faces), len(el.faces[0]))


    ## This is dependent on removed getAngles
    ## # ?? IS THIS DEFINED FOR *ANY* MESH ??
    ## #this is define for every linear surface or linear volume mesh.
    ## def equiAngleSkew(self):
    ##     """Returns the equiAngleSkew of the elements, a mesh quality parameter .
       
    ##   It quantifies the skewness of the elements: normalize difference between
    ##   the worst angle in each element and the ideal angle (angle in the face 
    ##   of an equiangular element, qe).
    ##   """
    ##     eang=self.getAngles(Deg)
    ##     eangsh= eang.shape
    ##     eang= eang.reshape(eangsh[0], eangsh[1]*eangsh[2])
    ##     eangMax, eangmin=eang.max(axis=1), eang.min(axis=1)        
    ##     f= self.elType().faces[1]
    ##     nedginface=len( array(f[ f.keys()[0] ], int)[0] )
    ##     qe=180.*(nedginface-2.)/nedginface
    ##     extremeAngles= [ (eangMax-qe)/(180.-qe), (qe-eangmin)/qe ]
    ##     return array(extremeAngles).max(axis=0)

######################## Functions #####################

# BV: THESE SHOULD GO TO connectivity MODULE

def extrudeConnectivity(e,nnod,degree):
    """_Extrude a Connectivity to a higher level Connectivity.

    e: Connectivity
    nnod: a number > highest node number
    degree: degree of extrusion (currently 1 or 2)

    The extrusion adds `degree` planes of nodes, each with an node increment
    `nnod`, to the original Connectivity `e` and then selects the target nodes
    from it as defined by the e.eltype.extruded[degree] value.
    
    Currently returns an integer array, not a Connectivity!
    """
    try:
        eltype,reorder = e.eltype.extruded[degree]
    except:
        try:
            eltype = e.eltype.name()
        except:
            eltype = None
        raise ValueError,"I don't know how to extrude a Connectivity of eltype '%s' in degree %s" % (eltype,degree)
    # create hypermesh Connectivity
    e = concatenate([e+i*nnod for i in range(degree+1)],axis=-1)
    # Reorder nodes if necessary
    if len(reorder) > 0:
        e = e[:,reorder]
    return Connectivity(e,eltype=eltype)


def replicConnectivity(e,n,inc):
    """_Repeat a Connectivity with increasing node numbers.

    e: a Connectivity
    n: integer: number of copies to make
    inc: integer: increment in node numbers for each copy

    Returns the concatenation of n connectivity tables, which are each copies
    of the original e, but having the node numbers increased by inc.
    """
    return Connectivity(concatenate([e+i*inc for i in range(n)]),eltype=e.eltype)
    

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
    return coords,[Connectivity(i[e],eltype=e.eltype) for i,e in zip(index,elems)]

# 
# Local utilities: move these to elements.py ??
#

def tri3_wts(ndiv):
    n = ndiv+1
    seeds = arange(n)
    pts = concatenate([
        column_stack([seeds[:n-i],[i]*(n-i)])
        for i in range(n)])
    pts = column_stack([ndiv-pts.sum(axis=-1),pts])
    return pts / float(ndiv)

def tri3_els(ndiv):
    n = ndiv+1
    els1 = [ row_stack([ array([0,1,n-j]) + i for i in range(ndiv-j) ]) + j * n - j*(j-1)/2 for j in range(ndiv) ]
    els2 = [ row_stack([ array([1,1+n-j,n-j]) + i for i in range(ndiv-j-1) ]) + j * n - j*(j-1)/2 for j in range(ndiv-1) ]
    elems = row_stack(els1+els2)
    
    return elems

def quad4_wts(nx,ny):
    x1 = arange(nx+1)
    y1 = arange(ny+1)
    x0 = nx-x1
    y0 = ny-y1
    pts = dstack([outer(y0,x0),outer(y0,x1),outer(y1,x1),outer(y1,x0)]).reshape(-1,4)
    return pts / float(nx*ny)

def quad4_els(nx,ny):
    n = nx+1
    els = [ row_stack([ array([0,1,n+1,n]) + i for i in range(nx) ]) + j * n for j in range(ny) ]
    return row_stack(els)
 


########### Deprecated #####################


# BV:
# While this function seems to make sense, it should be avoided
# The creator of the mesh normally KNOWS the correct connectivity,
# and should immediately fix it, instead of calculating it from
# coordinate data
# It could be used in a general Mesh checking/fixing utility

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


# deprecated in 0.8.5
def connectMeshSequence(ML,loop=False,**kargs):
    utils.deprec("connectMeshSequence is deprecated: use Mesh.connect instead")
    MR = ML[1:]
    if loop:
        MR.append(ML[0])
    else:
        ML = ML[:-1]
    HM = [ Mi.connect(Mj,**kargs) for Mi,Mj in zip (ML,MR) ]
    return Mesh.concatenate(HM)


# End
