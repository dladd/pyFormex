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

"""A class and functions for handling nodal connectivity.

This module defines a specialized array class for representing nodal
connectivity. This is e.g. used in mesh models, where geometry is
represented by a set of numbered points (nodes) and the geometric elements
are described by refering to the node numbers.
In a mesh model, points common to adjacent elements are unique, and
adjacency of elements can easily be detected from common node numbers. 
"""

from arraytools import *
from utils import deprecation
from messages import _future_deprecation


# BV: Should we make an InverseConnectivity class?


############################################################################
##
##   class Connectivity
##
#########################
#

class Connectivity(ndarray):
    """A class for handling element/node connectivity.

    A connectivity object is a 2-dimensional integer array with all
    non-negative values. Each row of the array defines an element by listing
    the numbers of its lower entity types. A typical use is a :class:`Mesh`
    object, where each element is defined in function of its nodes.
    While in a Mesh the word 'node' will normally refer to a geometrical
    point, here we will use 'node' for the lower entity whatever its nature
    is. It doesn't even have to be a geometrical entity.

    The current implementation limits a Connectivity object to numbers that
    are smaller than 2**31.
    
    In a row (element), the same node number may occur more than once, though
    usually all numbers in a row are different. Rows containing duplicate
    numbers are called `degenerate` elements.
    Rows containing the same node sets, albeit different permutations thereof,
    are called 'double's.

    A new Connectivity object is created with the following syntax ::
    
      Connectivity(data=[],dtyp=None,copy=False,nplex=0)

    Parameters:
    
    - `data`: should be compatible with an integer array with shape
      `(nelems,nplex)`, where `nelems` is the number of elements and
      `nplex` is the plexitude of the elements.
    - `dtype`: can be specified to force an integer type but is set by
      default from the passed `data`. 
    - `copy`: can be set True to force copying the data. By default, the
      specified data will be used without copying, if possible.
    - `nplex`: can be specified to force a check on the plexitude of the
      data, or to set the plexitude for an empty Connectivity.
      An error will be raised if the specified data do not match the
      specified plexitude.

    Example:

    >>> print Connectivity([[0,1,2],[0,1,3],[0,3,2],[0,5,3]])
    [[0 1 2]
     [0 1 3]
     [0 3 2]
     [0 5 3]]
      
    """
    #
    # :DEV
    # Because we have a __new__ constructor here and no __init__,
    # we have to list the arguments explicitely in the docstring above.
    #
    def __new__(self,data=[],dtyp=None,copy=False,nplex=0,allow_negative=False,eltype=None):
        """Create a new Connectivity object."""

        if eltype is None:
            try:
                eltype = data.eltype
            except:
                eltype = None
        
        # Turn the data into an array, and copy if requested
        ar = array(data, dtype=dtyp, copy=copy)
        if ar.ndim < 2:
            if nplex > 0:
                ar = ar.reshape(-1,nplex)
            else:
                ar = ar.reshape(-1,1)
                
        elif ar.ndim > 2:
            raise ValueError,"Expected 2-dim data"

        # Make sure dtype is an int type
        if ar.dtype.kind != 'i':
            ar = ar.astype(Int)
 
        # Check values
        if ar.size > 0:
            maxval = ar.max()
            if maxval > 2**31-1 or (ar.min() < 0 and not allow_negative):
                raise ValueError,"Negative or too large positive value in data"
            if nplex > 0 and ar.shape[1] != nplex:
                raise ValueError,"Expected data of plexitude %s" % nplex
        else:
            maxval = -1
            ar = ar.reshape(0,nplex)
            
        # Transform 'subarr' from an ndarray to our new subclass.
        ar = ar.view(self)

        ## # Other data
        ar.inv = None   # inverse index
        ar.maxval = maxval

        if eltype is not None:
            ar.eltype = eltype

        return ar


    def nelems(self):
        """Return the number of elements in the Connectivity table.

        Example:

          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2],[0,5,3]]).nelems()
          4
        """
        return self.shape[0]

    
    def nplex(self):
        """Return the plexitude of the elements in the Connectivity table.

        Example:

          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2],[0,5,3]]).nplex()
          3
        """
        return self.shape[1]


    def report(self):
        """Format a Connectivity table"""
        s = "Conn %s" % (self.shape,)
        if hasattr(self,'eltype'):
            s += ", eltype=%s" % self.eltype
        s += '\n'
        return s + ndarray.__str__(self)

############### Detecting degenerates and duplicates ##############


    def testDegenerate(self):
        """Flag the degenerate elements (rows).

        A degenerate element is a row which contains at least two
        equal values. 

        Returns: a boolean array with shape (self.nelems(),).
          The True values flag the degenerate rows.

        Example:
        
          >>> Connectivity([[0,1,2],[0,1,1],[0,3,2]]).testDegenerate()
          array([False,  True, False], dtype=bool)
          
        """
        srt = asarray(self.copy())
        srt.sort(axis=1)
        return (srt[:,:-1] == srt[:,1:]).any(axis=1)
        

    def listDegenerate(self):
        """Return a list with the numbers of the degenerate elements.

        Example:
        
          >>> Connectivity([[0,1,2],[0,1,1],[0,3,2]]).listDegenerate()
          array([1])
          
        """
        return arange(self.nelems())[self.testDegenerate()]


    def listNonDegenerate(self):
        """Return a list with the numbers of the non-degenerate elements.

        Example:
        
          >>> Connectivity([[0,1,2],[0,1,1],[0,3,2]]).listNonDegenerate()
          array([0, 2])

        """
        return arange(self.nelems())[~self.testDegenerate()]


    def removeDegenerate(self):
        """Remove the degenerate elements from a Connectivity table.

        Degenerate elements are rows with repeating values.
        Returns a Connectivity with the degenerate elements removed.

        Example:
        
        >>> Connectivity([[0,1,2],[0,1,1],[0,3,2]]).removeDegenerate()
        Connectivity([[0, 1, 2],
               [0, 3, 2]])

        """
        return self[~self.testDegenerate()]


    def reduceDegenerate(self,target=None):
        """Reduce degenerate elements to lower plexitude elements.

        This will try to reduce the degenerate elements of the Connectivity
        to a lower plexitude. This is only possible if an element type
        was set in the Connectivity. This function uses the data of
        the Element database in :mod:`elements`.

        If a target element type is given, only the reductions to that element
        type are performed. Else, all the target element types for which
        a reduction scheme is available, will be tried.

        Returns:
        
        A list of Connectivities of which the first one contains
        the originally non-degenerate elements and the last one contains
        the elements that could not be reduced and may be empty.
        If the original Connectivity does not have an element type set,
        or the element type does not have any reduction schemes defined,
        a list with only the original is returned.

        Remark: If the Connectivity is part of a Mesh, you should use the
        Mesh.reduceDegenerate method instead, as that one will preserve
        the property numbers into the resulting Meshes.

        Example:
        
        >>> C = Connectivity([[0,1,2],[0,1,1],[0,3,2]],eltype='line3')
        >>> print C.reduceDegenerate()
        [Connectivity([[0, 1]]), Connectivity([[0, 1, 2],
               [0, 3, 2]])]
        
        """
        from elements import elementType
        if not hasattr(self,'eltype'):
           return [ self ]

        eltype = elementType(self.eltype)
        if not hasattr(eltype,'degenerate'):
            return [ self ]

        # get all reductions for this eltype
        strategies = eltype.degenerate

        # if target, keep only those leading to target
        if target is not None:
            s = strategies.get(target,[])
            if s:
                strategies = {target:s}
            else:
                strategies = {}

        if not strategies:
            return [self]


        e = self
        ML = []

        for totype in strategies:

            elems = []
            for conditions,selector in strategies[totype]:
                cond = array(conditions)
                w = (e[:,cond[:,0]] == e[:,cond[:,1]]).all(axis=1)
                sel = where(w)[0]
                if len(sel) > 0:
                    elems.append(e[sel][:,selector])
                    # remove the reduced elems from m
                    e = e[~w]

                    if e.nelems() == 0:
                        break

            if elems:
                elems = concatenate(elems)
                ML.append(Connectivity(elems,eltype=totype))

            if e.nelems() == 0:
                break

        ML.append(e)

        return ML

    
    def testDoubles(self,permutations=True):
    #    This algorithm is faster than encode,
    #    but for nplex=2 enmagic2 would probably still be faster.
        """Test the Connectivity list for doubles.

        By default, doubles are elements that consist of the same set of
        nodes, in any particular order. Setting permutations to False
        will only find the double rows that have matching values at
        every position.

        This function returns a tuple with two arrays:
        
        - an index used to sort the elements
        - a flags array with the value True for indices of the unique elements
          and False for those of the doubles.

        Example:
        
          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).testDoubles()
          (array([0, 1, 2]), Connectivity([ True, False,  True], dtype=bool))
          
        """
        if permutations:
            C = self.copy()
            C.sort(axis=1)
        else:
            C = self
        ind = sortByColumns(C)
        C = C.take(ind,axis=0)
        ok = (C != roll(C,1,axis=0)).any(axis=1)
        if not ok[0]: # all doubles -> should result in one unique element
            ok[0] = True
        return ind,ok
    

    def listUnique(self,permutations=True):
        """Return a list with the numbers of the unique elements.

        Example:
        
          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).listUnique()
          array([0, 2])
          
        """
        ind,ok = self.testDoubles(permutations)
        return ind[ok]


    def listDoubles(self,permutations=True):
        """Return a list with the numbers of the double elements.

        Example:
        
          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).listDoubles()
          array([1])
          
        """
        ind,ok = self.testDoubles(permutations)
        return ind[~ok]

   
    def removeDoubles(self,permutations=True):
        """Remove doubles from a Connectivity list.

        By default, doubles are elements that consist of the same set of
        nodes, in any particular order. Setting permutations to False
        will only remove the double rows that have matching values at
        matching positions.

        Returns a new Connectivity with the double elements removed.

        Example:
        
          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).removeDoubles()
          Connectivity([[0, 1, 2],
                 [0, 3, 2]])
        """
        ind,ok = self.testDoubles(permutations)
        return self[ind[ok]]


    def reorder(self,order='nodes'):
        """Reorder the elements of a Connectivity in a specified order.

        This does not actually reorder the elements itself, but returns
        an index with the order of the rows (elements) in the connectivity
        table that meets the specified requirements.

        Parameters:

        - `order`: specifies how to reorder the elements. It is either one
          of the special string values defined below, or else it is an index
          with length equal to the number of elements. The index should be
          a permutation of the numbers in `range(self.nelems()`. Each value
          gives of the number of the old element that should be placed at
          this position. Thus, the order values are the old element numbers
          on the position of the new element number.

          `order` can also take one of the following predefined values,
          resulting in the corresponding renumbering scheme being generated:

          - 'nodes': the elements are renumbered in order of their appearance
            in the inverse index, i.e. first are the elements connected to
            node 0, then the as yet unlisted elements connected to node 1, etc.
          - 'random': the elements are randomly renumbered.
          - 'reverse': the elements are renumbered in reverse order.

        Returns:
          A 1-D integer array which is a permutation of
          `arange(self.nelems()`, such that taking the elements in this order
          will produce a Connectivity reordered as requested. In case an
          explicit order was specified as input, this order is returned after
          checking that it is indeed a permutation of `range(self.nelems()`.

        Example:

          >>> A = Connectivity([[1,2],[2,3],[3,0],[0,1]])
          >>> A[A.reorder('reverse')]
          Connectivity([[0, 1],
                 [3, 0],
                 [2, 3],
                 [1, 2]])
          >>> A.reorder('nodes')
          array([3, 2, 0, 1])
          >>> A[A.reorder([2,3,0,1])]
          Connectivity([[3, 0],
                 [0, 1],
                 [1, 2],
                 [2, 3]])
         
        """
        if order == 'nodes':
            a = sort(self,axis=-1)  # first sort rows
            order = sortByColumns(a)
        elif order == 'reverse':
            order = arange(self.nelems()-1,-1,-1)
        elif order == 'random':
            order = random.permutation(self.nelems())
        else:
            order = asarray(order)
            if not (order.dtype.kind == 'i' and \
                    (sort(order) == arange(order.size)).all()):
                raise ValueError,"order should be a permutation of range(%s)" % self.nelems()
        return order


    def inverse(self):
        """Return the inverse index of a Connectivity table.

        This returns the inverse index of the Connectivity, as computed
        by :func:`arraytools.inverseIndex`. See 
           
        Example:
        
          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2]]).inverse()
          array([[ 0,  1,  2],
                 [-1,  0,  1],
                 [-1,  0,  2],
                 [-1,  1,  2]])
        """
        if self.inv is None:
            if self.size > 0:
                self.inv = inverseIndex(self)
            else:
                self.inv = Connectivity()
        return self.inv


    def adjacency(self,kind='e'):
        """Return a table of adjacent items.

        Returns an element adjacency table (kind='e') or node adjacency
        table (kind='n').

        An element `i` is said to be ajacent to element `j`, if the two
        elements have at least one common node.

        A node `i` is said to be adjacent to node `j`, if there is at least
        one element containing both nodes.

        Parameters:

        - `kind`: 'e' or 'n', requesting resp. element or node adjacency.
        
        Returns: an integer array with shape (nr,nc),
        where row `i` holds a sorted list of all the items that are
        adjacent to item `i`, padded with -1 values to create an equal
        list length for all items.

           
        Example:
        
          >>> Connectivity([[0,1],[0,2],[1,3],[0,5]]).adjacency('e')
          array([[ 1,  2,  3],
                 [-1,  0,  3],
                 [-1, -1,  0],
                 [-1,  0,  1]])
          >>> Connectivity([[0,1],[0,2],[1,3],[0,5]]).adjacency('n')
          array([[ 1,  2,  5],
                 [-1,  0,  3],
                 [-1, -1,  0],
                 [-1, -1,  1],
                 [-1, -1, -1],
                 [-1, -1,  0]])
          >>> Connectivity([[0,1,2],[0,1,3],[2,4,5]]).adjacency('n')
          array([[-1,  1,  2,  3],
                 [-1,  0,  2,  3],
                 [ 0,  1,  4,  5],
                 [-1, -1,  0,  1],
                 [-1, -1,  2,  5],
                 [-1, -1,  2,  4]])
                 
        """
        inv = self.inverse()
        if kind == 'e':
            adj = inv[self].reshape((self.nelems(),-1))
        elif kind == 'n':
            adj = concatenate([where(inv>=0,self[:,i][inv],inv) for i in range(self.nplex())],axis=1)
        else:
            raise ValueError,"kind should be 'e' or 'n', got %s" % str(kind) 
        return reduceAdjacency(adj)


######### Creating intermediate levels ###################

    def selectNodes(self,selector):
        """Return a :class:`Connectivity` containing subsets of the nodes.

        Parameters:

        - `selector`: an object that can be converted to a 1-dim or 2-dim
          int array. Examples are a tuple of local node numbers, or a list
          of such tuples all having the same length.
          Each row of `selector` holds a list of the local node numbers that
          should be retained in the new Connectivity table.

        Returns a :class:`Connectivity` object with shape
        `(self.nelems*selector.nelems,selector.nplex)`. This function
        does not collapse the double elements. The eltype of the result
        is equal to that of the selector, possibly None.
           
        Example:
        
          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).selectNodes([[0,1],[0,2]])
          Connectivity([[0, 1],
                 [0, 2],
                 [0, 2],
                 [0, 1],
                 [0, 3],
                 [0, 2]])

        """
        sel = Connectivity(selector)
        try:
            eltype = sel.eltype
        except:
            eltype = None
        if sel.size > 0:
            return Connectivity(self[:,sel].reshape(-1,sel.nplex()),eltype=eltype)
        else:
            return Connectivity()
        

    def insertLevel(self,selector,lower_only=False):
        """Insert an extra hierarchical level in a Connectivity table.

        A Connectivity table identifies higher hierchical entities in function
        of lower ones. This method inserts an extra hierarchical level.
        For example, if you have volumes defined in function of points,
        you can insert an intermediate level of edges, or faces.
        Multiple intermediate level entities may be created from each
        element.

        Parameters:

        - `selector`: an object that can be converted to a 1-dim or 2-dim
          integer array. Examples are a tuple of local node numbers, or a list
          of such tuples all having the same length.
          Each row of `selector` holds a list of the local node numbers that
          should be retained in the new Connectivity table.
        - `lower_only`: if True, only the definition of the new (lower)
          entities is returned, complete without removing doubles.
          This is equivalent to using :meth:`selectNodes`, which
          is prefered when you do not need the higher level info. 
        
        Return value: a tuple of two Connectivities `hi`,`lo`, where:
    
        - `hi`: defines the original elements in function of the intermediate
          level ones,
        - `lo`: defines the intermediate level items in function of the lowest
          level ones (the original nodes). If the `selector` has an `eltype`
          attribute, then `lo` will inherit the same `eltype` value.
          
        Intermediate level items that consist of the same items in any
        permutation order are collapsed to single items.
        The low level items respect the numbering order inside the
        original elements, but it is undefined which of the collapsed
        sequences is returned.

        Because the precise order of the data in the collapsed rows is lost,
        it is in general not possible to restore the exact original table
        from the two result tables.
        See however :meth:`Mesh.getBorder` for an application where an
        inverse operation is possible, because the border only contains
        unique rows.
        See also :meth:`Mesh.combine`, which is an almost inverse operation
        for the general case, if the selector is complete.
        The resulting rows may however be permutations of the original.

        Example:
        
          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).insertLevel([[0,1],[1,2],[2,0]])
          (Connectivity([[0, 3, 1],
                 [1, 3, 0],
                 [2, 4, 1]]), Connectivity([[0, 1],
                 [2, 0],
                 [0, 3],
                 [1, 2],
                 [3, 2]]))
           
        """
        sel = Connectivity(selector)
        lo = self.selectNodes(sel)
        if lo.size > 0:
            uniq,uniqid = uniqueRows(lo,permutations=True)
            hi = Connectivity(uniqid.reshape(-1,sel.nelems()))
            lo = lo[uniq]
        else:
            hi = lo = Connectivity()
        #
        # PUT THIS BEHIND THE SELECTION, BECAUSE IT LOOSES THE 'eltype'
        #
        if hasattr(sel,'eltype'):
            lo.eltype = sel.eltype
        return hi,lo
    

    # BV: This is currently far from general!!!
    # should probably be move to Mesh/TriSurface if needed there
    def combine(self,lo):
        """Combine two hierarchical Connectivity levels to a single one.

        self and lo are two hierarchical Connectivity tables, representing
        higher and lower level respectively. This means that the elements
        of self hold numbers which point into lo to obtain the lowest level
        items.

        *In the current implementation, the plexitude of lo should be 2!*

        As an example, in a structure of triangles, hi could represent
        triangles defined by 3 edges and lo could represent edges defined
        by 2 vertices. This method will then result in a table
        with plexitude 3 defining the triangles in function of the vertices.

        This is the inverse operation of :meth:`insertLevel` with a selector
        which is complete.
        The algorithm only works if all vertex numbers of an element are
        unique.

        Example:

          >>> hi,lo = Connectivity([[0,1,2],[0,2,1],[0,3,2]]).insertLevel([[0,1],[1,2],[2,0]])
          >>> hi.combine(lo)
          Connectivity([[0, 1, 2],
                 [0, 2, 1],
                 [0, 3, 2]])

        """
        lo = Connectivity(lo) 
        if self.shape[1] < 2 or lo.shape[1] != 2:
            raise ValueError,"Can only combine plex>=2 with plex==2"
        elems = lo[self]
        elems1 = roll(elems,-1,axis=1)
        for i in range(elems.shape[1]):
            flags = (elems[:,i,1] != elems1[:,i,0]) * (elems[:,i,1] != elems1[:,i,1])
            elems[flags,i] = roll(elems[flags,i],1,axis=1)
        return Connectivity(elems[:,:,0])

    
    def resolve(self):
        """Resolve the connectivity into plex-2 connections.

        Creates a Connectivity table with a plex-2 (edge) connection
        between any two nodes that are connected to a common element.

        There is no point in resolving a plexitude 2 structure.
        Plexitudes lower than 2 can not be resolved.

        Returns a plex-2 Connectivity with all connections between node
        pairs. In each element the nodes are sorted.

        Example:
        
          >>> print([ i for i in combinations(range(3),2) ])
          [(0, 1), (0, 2), (1, 2)]
          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).resolve()
          Connectivity([[0, 1],
                 [0, 2],
                 [0, 3],
                 [1, 2],
                 [2, 3]])

        """
        ind = [ i for i in combinations(range(self.nplex()),2) ]
        hi,lo = self.insertLevel(ind)
        lo.sort(axis=1)
        ind = sortByColumns(lo)
        return lo[ind]


#######################################################################
    # class and static methods #

    @staticmethod
    def connect(clist,nodid=None,bias=None,loop=False):
        """Connect nodes from multiple Connectivity objects.

        clist is a list of Connectivities, nodid is an optional list of node
        indices and
        bias is an optional list of element bias values. All lists should have
        the same length.

        The returned Connectivity has a plexitude equal to the number of
        Connectivities in clist. Each element of the new Connectivity consist
        of a node from the corresponding element of each of the Connectivities
        in clist. By default this will be the first node of that element,
        but a nodid list may be given to specify the node id to be used for each
        of the Connectivities.
        Finally, a list of bias values may be given to specify an offset in
        element number for the subsequent Connectivities.
        If loop==False, the length of the Connectivity will be the minimum
        length of the Connectivities in clist, each minus its respective bias.
        By setting loop=True however, each Connectivity will loop around if
        its end is encountered, and the length of the result is the maximum
        length in clist.

        Example:

          >>> a = Connectivity([[0,1],[2,3],[4,5]])
          >>> b = Connectivity([[10,11,12],[13,14,15]])
          >>> c = Connectivity([[20,21],[22,23]])
          >>> print(Connectivity.connect([a,b,c]))
          [[ 0 10 20]
           [ 2 13 22]]
          >>> print(Connectivity.connect([a,b,c],nodid=[1,0,1]))
          [[ 1 10 21]
           [ 3 13 23]]
          >>> print(Connectivity.connect([a,b,c],bias=[1,0,1]))
          [[ 2 10 22]]
          >>> print(Connectivity.connect([a,b,c],bias=[1,0,1],loop=True))
          [[ 2 10 22]
           [ 4 13 20]
           [ 0 10 22]]
          
        """
        try:
            m = len(clist)
            for i in range(m):
                if isinstance(clist[i],Connectivity):
                    pass
                elif isinstance(clist[i],ndarray):
                    clist[i] = Connectivity(clist[i])
                else:
                    raise TypeError
        except TypeError:
            raise TypeError,'Connectivity.connect(): first argument should be a list of Connectivities'

        if not nodid:
            nodid = [ 0 for i in range(m) ]
        if not bias:
            bias = [ 0 for i in range(m) ]
        if loop:
            n = max([ clist[i].nelems() for i in range(m) ])
        else:
            n = min([ clist[i].nelems() - bias[i] for i in range(m) ])
        f = zeros((n,m),dtype=Int)
        for i,j,k in zip(range(m),nodid,bias):
            v = clist[i][k:k+n,j]
            if loop and k > 0:
                v = concatenate([v,clist[i][:k,j]])
            f[:,i] = resize(v,(n))
        return Connectivity(f)



######################################################################
    # BV: the methods below should probably be removed,
    # after a check that they are not essential

    @deprecation("tangle has been renamed to combine") 
    def tangle(self,*args,**kargs):
        return self.combine(*args,**kargs)
    
    
    @deprecation("untangle has been deprecated. Use insertLevel instead.") 
    def untangle(self,ind):
        return self.insertLevel(ind)


    @deprecation(_future_deprecation)
    def encode(self,permutations=True,return_magic=False):
        """_Encode the element connectivities into single integer numbers.

        Each row of numbers is encoded into a single integer value, such that
        equal rows result in the same number and different rows yield
        different numbers. Furthermore, enough information can be kept to
        restore the original rows from these single integer numbers.
        This is seldom needed however, because the original data are
        available from the Connectivity table itself.

        - permutations: if True(default), two rows are considered equal if
          they contain the same numbers regardless of their order.
          If False, two rows are only equal if they contain the same
          numbers at the same position.
        - return_magic: if True, return a codes,magic tuple. The default is
          to return only the codes.
          
        Return value(s):
    
        - codes: an (nelems,) shaped array with the element code numbers,
        - magic: the information needed to restore the original rows from
          the codes. See Connectivity.decode()

        Example:
        
          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2]]).encode(return_magic=True)
          (array([0, 1, 3]), [(2, array([0, 1]), array([2, 3])), (2, array([0]), array([1, 2]))])
          
        *The use of this function is deprecated.*
        """
        def compact_encode2(data):
            """Encode two columns of integers into a single column.

            This is like enmagic2 but results in smaller encoded values, because
            the original values are first replaced by indices into the sets of unique
            values.
            This encoding scheme is therefore usable for repeated application
            on multiple columns.

            The return value is the list of codes, the magic value used in encoding,
            and the two sets of uniq values for the columns, needed to restore the
            original data. Decoding can be done with compact_decode2.
            """
            # We could use a single compaction vector?
            uniqa, posa = unique(data[:,0], return_inverse=True)
            uniqb, posb = unique(data[:,1], return_inverse=True)
            # We could insert the encoding directly here,
            # or use an encoding function with 2 arguments
            # to avoid the column_stack operation
            rt = column_stack([posa, posb])
            codes, magic = enmagic2(rt)
            return codes,magic,uniqa,uniqb
        
        
        if permutations:
            data = self.copy()
            data.sort(axis=1)
        else:
            data = self
            
        magic = []
        codes = data[:,0]
        for i in range(1,data.shape[1]):
            cols = column_stack([codes,data[:,i]])
            codes,mag,uniqa,uniqb = compact_encode2(cols)
            # insert at the front so we can process in order
            magic.insert(0,(mag,uniqa,uniqb))

        if return_magic:
            return codes,magic
        else:
            return codes


    @deprecation(_future_deprecation)
    @staticmethod
    def decode(codes,magic):
        """Decode element codes into a Connectivity table.

        This is the inverse operation of the Connectivity.encode() method.
        It recreates a Connectivity table from the (codes,magic) information.

        This is a static method, and should be invoked as
        ```Connectivity.decode(codes,magic)```.
        
        - codes: code numbers as returned by Connectivity.encode, or a subset
          thereof.
        - magic: the magic information as returned by Connectivity.encode,
          with argument return_magic=True.

        Returns a Connectivity table.

        Example:
        
          >>> Connectivity.decode(array([0,1,3]), [(2, array([0, 1]), array([2, 3])), (2, array([0]), array([1, 2]))])
          Connectivity([[0, 1, 2],
                 [0, 1, 3],
                 [0, 2, 3]])

        *The use of this function is deprecated.*
        """

        def compact_decode2(codes,magic,uniqa,uniqb):
            """Decodes a single integer value into the original 2 values.

            This is the inverse operation of compact_encode2.
            Thus compact_decode2(*compact_encode(data)) will return data.

            codes can be a subset of the encoded values, but the other 3 arguments
            should be exactly those from the compact_encode2 result.
            """
            # decoding returns the indices into the uniq numberings
            pos = demagic2(codes,magic)
            return column_stack([uniqa[pos[:,0]],uniqb[pos[:,1]]])
        
        data = []
        for mag in magic:
            cols = compact_decode2(codes,mag[0],mag[1],mag[2])
            data.insert(0,cols[:,1])
            codes = cols[:,0]
        data.insert(0,codes)
        return Connectivity(column_stack(data))


############################################################################

def sortAdjacency(adj):
    """Sort an adjacency table.

    An adjacency table is an integer array where each row lists the numbers
    of the items that are connected to the item with number equal to the row
    index. Rows are padded with -1 value to create rows of equal length.

    This function sorts the rows of the adjacency table in ascending order
    and removes all columns containing only -1 values.

    Paramaters:

    - `adj`: an 2-D integer array with values >=0 or -1
   
    Returns: an integer array with shape (adj.shape[0],maxc), with
    maxc <= adj.shape[1], where the rows are sorted in ascending order
    and where columns with only -1 values are removed.

    Example:

      >>> a = array([[ 0,  2,  1, -1],
      ...            [-1,  3,  1, -1],
      ...            [ 3, -1,  0,  1],
      ...            [-1, -1, -1, -1]])
      >>> sortAdjacency(a)
      array([[ 0,  1,  2],
             [-1,  1,  3],
             [ 0,  1,  3],
             [-1, -1, -1]])

    """
    adj.sort(axis=-1)      # sort rows
    maxc = adj.max(axis=0) # find maximum per column
    adj = adj[:,maxc>=0]   # retain columns with non-negative maximum
    return adj


def reduceAdjacency(adj):
    """Reduce an adjacency table.

    An adjacency table is an integer array where each row lists the numbers
    of the items that are connected to the item with number equal to the row
    index. Rows are padded with -1 value to create rows of equal length.

    A reduced adjacency table is one where each row:

    - does not contain the row index itself,
    - does not contain doubles,
    - is sorted in ascending order,

    and that has at least one row without -1 value.

    Paramaters:

    - `adj`: an 2-D integer array with value >=0 or -1
   
    Returns: an integer array with shape (adj.shape[0],maxc), with
    maxc <= adj.shape[1], where row `i` retains the unique non-negative
    numbers of the original array except the valu `i`, and is possibly
    padded with -1 values.

    Example:

      >>> a = array([[ 0,  0,  0,  1,  2,  5],
      ...            [-1,  0,  1, -1,  1,  3],
      ...            [-1, -1,  0, -1, -1,  2],
      ...            [-1, -1,  1, -1, -1,  3],
      ...            [-1, -1, -1, -1, -1, -1],
      ...            [-1, -1,  0, -1, -1,  5]])
      >>> reduceAdjacency(a)
      array([[ 1,  2,  5],
             [-1,  0,  3],
             [-1, -1,  0],
             [-1, -1,  1],
             [-1, -1, -1],
             [-1, -1,  0]])

    """
    adj = checkArrayDim(adj,2)
    n = adj.shape[0]
    adj[adj == arange(n).reshape(n,-1)] = -1 # remove the item i
    adj = sortAdjacency(adj)
    adj[adj[:,:-1] == adj[:,1:]] = -1 #remove duplicate items
    adj = sortAdjacency(adj)
    return adj

# BV: This could become one of the schemes of Connectivity.reorder 

def findConnectedLineElems(elems):
    """Find a single path of connected line elems.

    This function is intended as a helper function for
    :func:`connectedLineElems`. It should probably not be used directly,
    because, as a side-effect, it changes the data in the `elems` argument.
    :func:`connectedLineElems` does not have this inconvenience.

    The function searches a Connectivity table for a chain of elements
    in which the first node of all but the first element is equal to the
    last node of the previous element. To create such a chain, elements may
    be reordered and the node sequence of an element may be reversed.

    Parameters:

    - `elems`: Connectivity-like. Any plexitude is allowed, but only the
      first and the last columna are relevant. 

    Returns: a Connectivity with a single chain extracted from the input
      Connectivity. The result will not necessarily be the longest path.
      It will however contain the first element of the input table.

      As a side-effect, all elements contained in the output chain will
      have their entries in the input table `elems` changed to -1.

    Example:

      >>> findConnectedLineElems([[0,1],[1,2],[0,4],[4,2]])
      Connectivity([[0, 1],
             [1, 2],
             [2, 4],
             [4, 0]])
             
      >>> findConnectedLineElems([[0,1],[1,2],[0,4]])
      Connectivity([[2, 1],
             [1, 0],
             [0, 4]])
             
      >>> C = Connectivity([[0,1],[0,2],[0,3],[4,5]])
      >>> findConnectedLineElems(C)
      Connectivity([[ 1,  0],
             [ 0,  2],
             [-1, -1],
             [-1, -1]])
      >>> print C
      [[-1 -1]
       [-1 -1]
       [ 0  3]
       [ 4  5]]
    """
    if not isinstance(elems,Connectivity):
        elems = Connectivity(elems)
    srt = zeros_like(elems) - 1
    ie = 0
    je = 0
    rev = False
    k = elems[0][0] # remember startpoint
    while True:
        # Store an element that has been found ok
        if rev:
            srt[ie] = elems[je,::-1]
        else:
            srt[ie] = elems[je]
        elems[je] = -1 # Done with this one
        j = srt[ie][-1] # remember endpoint
        if j == k:
            break
        ie += 1

        # Look for the next connected element (only thru fist or last node!)
        w = where(elems[:,[0,-1]] == j)
        #print w
        if w[0].size == 0:
            # Try reversing
            w = where(elems[:,[0,-1]] == k)
            #print w
            if w[0].size == 0:
                break
            else:
                j,k = k,j
                # reverse the table (colums and rows)
                srt[:ie] = srt[ie-1::-1,::-1].copy()  # copy needed!!
        je = w[0][0]
        rev = w[-1][0] > 0 #check if the target node is the first or last

    return srt


# BV: this could become a Connectivity function splitByConnection

def connectedLineElems(elems):
    """Partition a segmented curve into connected segments.
    
    The input argument is a (nelems,2) shaped array of integers.
    Each row holds the two vertex numbers of a single line segment.

    The return value is a list of (nsegi,2) shaped array of integers.

    Example:
    
      >>> connectedLineElems([[0,1],[1,2],[0,4],[4,2]])
      [Connectivity([[0, 1],
             [1, 2],
             [2, 4],
             [4, 0]])]
             
      >>> connectedLineElems([[0,1],[1,2],[0,4]])
      [Connectivity([[2, 1],
             [1, 0],
             [0, 4]])]
             
      >>> connectedLineElems([[0,1],[0,2],[0,3],[4,5]])
      [Connectivity([[1, 0],
             [0, 2]]), Connectivity([[0, 3]]), Connectivity([[4, 5]])]
             
      >>> connectedLineElems([[0,1,2],[2,0,3],[0,3,1],[4,5,2]])
      [Connectivity([[3, 0, 2],
             [2, 1, 0],
             [0, 3, 1]]), Connectivity([[4, 5, 2]])]
    """
    elems = Connectivity(elems).copy() # make copy to avoid side effects
    parts = []
    while elems.size != 0:
        loop = findConnectedLineElems(elems)
        parts.append(loop[(loop!=-1).any(axis=1)])
        elems = elems[(elems!=-1).any(axis=1)]
    return parts


## # This is an experimental replacement for connectedLineElems
## def splitConnectedLines(elems):
##     """Split and order a set of line segments into connected componenents.

##     Parameters:

##     - `elems`: Connectivity-like with plexitude 2 (for eltype line2 ) or 3 (for eltype line3)

##     Returns: a tuple (return_code,table):
    
##     - `return_code`: an integer with one of the following values:
    
##       - 0: the segments form a closed loop
##       - 1: the segments form a single non-closed path
##       - 2: the segments form multiple not connected paths
      
##     - `table`:

##       - if return_code is 0 or 1: a Connectivity table equivalent
##         to the input, but with the elements and their nodes sorted in order.
##       - if return_code is 2: a table with a singly connected part in the
##         top rows, followed by -1 values for the unconnected elements.

##     Example:

##       >>> splitConnectedLines([[0,1],[1,2],[0,4],[4,2]])
##       (0, Connectivity([[0, 1],
##              [1, 2],
##              [2, 4],
##              [4, 0]]))
             
##       >>> splitConnectedLines([[0,1],[1,2],[0,4]])
##       (1, Connectivity([[2, 1],
##              [1, 0],
##              [0, 4]]))
             
##       >>> splitConnectedLines([[0,1],[0,2],[0,3],[4,5]])
##       (2, Connectivity([[ 1,  0],
##              [ 0,  2],
##              [-1, -1],
##              [-1, -1]]))
##     """
##     elems = Connectivity(elems)

##     def findOneComponent(elems):
##         # Sorted list of elements: -1 is unused, >= 0 is element number
##         srt = zeros(elems.shape[0],dtype=Int) -1
##         # Status of elements:
##         # -1 is unused,
##         #  0 is used in forward direction
##         #  1 is used in backward direction
##         sta = zeros(elems.shape[0],dtype=Int) - 1
##         ie = 0
##         je = 0
##         rev = False
##         k = elems[0][0] # startpoint
##         while True:
##             # Store an element that has been found ok
##             # store new endpoint in j
##             # disable the element for further searches
##             srt[ie] = je
##             ie += 1
##             if rev:
##                 sta[je] = 1
##                 j = elems[je][0]
##             else:
##                 sta[je] = 0
##                 j = elems[je][-1]
##             elems[je] = [ -1, -1 ]

##             # check for a loop
##             if j == k:
##                 break

##             # Look for the next connected element
##             w = where(elems == j)
##             #print w
##             if w[0].size == 0:
##                 # Try other end of chain
##                 w = where(elems == k)
##                 #print w
##                 if w[0].size == 0:
##                     break
##                 else:
##                     j,k = k,j
##                     srt[:ie] = srt[ie-1::-1].copy() # copy needed !!!
##                     sta[:ie] = 1-sta[:ie]
##             je = w[0][0]
##             rev = w[1][0] == 1
##         if any(srt == -1):
##             ret = 2
##         else:
##             ret = 0
##         return ret,srt,sta


############################################################################
#
# Deprecated
#


@deprecation(_future_deprecation)
def connected(index,i):
    """Return the list of elements connected to element i.

    index is a (nr,nc) shaped integer array.
    An element j of index is said to be connected to element i, if element j
    has at least one (non-negative) value in common with element i.

    The result is a sorted list of unique element numbers, not containing
    the element number i itself.
    """
    adj = concatenate([ where(ind==j)[0] for j in ind[i] if j >= 0 ])
    return unique(adj[adj != i])

   
@deprecation(_future_deprecation)
def enmagic2(cols,magic=0):
    """Encode two integer values into a single integer.

    cols is a (n,2) array of non-negative integers smaller than 2**31.
    The result is an (n) array of type int64, where each value is
    unique for each row of values in the input.
    The original input can be restored with demagic2.

    If a magic value larger than the maximum integer in the table is
    given, it will be used. If not, it will be taken as the maximum+1.
    A negative magic value triggers a fastencode scheme.

    The return value is a tuple with the codes and the magic used.

    *The use of this function is deprecated.*
    """
    cmax = cols.max()
    if cmax >= 2**31 or cols.min() < 0:
        raise ValueError,"Integer value too high (>= 2**31) in enmagic2"
        
    if cols.ndim != 2 or cols.shape[1] != 2:
        raise ValueError,"Invalid array (type %s, shape %s) in enmagic2" % (cols.dtype,cols.shape)
    
    if magic < 0:
        magic = -1
        cols = array(cols,copy=True,dtype=int32,order='C')
        codes = cols.view(int64)
    else:
        if magic <= cmax:
            magic = cmax + 1
        codes = cols[:,0].astype(int64) * magic + cols[:,1]
    return codes,magic

        
@deprecation(_future_deprecation)
def demagic2(codes,magic):
    """Decode an integer number into two integers.

    The arguments `codes` and `magic` are the result of an enmagic2() operation.
    This will restore the original two values for the codes.

    A negative magic value flags the fastencode option.
    
    *The use of this function is deprecated.*
    """
    if magic < 0:
        cols = codes.view(int32).reshape(-1,2)
    else:
        cols = column_stack([codes/magic,codes%magic]).astype(int32)
    return cols


@deprecation("partitionSegmentedCurve is deprecated. Use connectedLineElems instead.")
def partitionSegmentedCurve(*args,**kargs):
    return connectedLineElems(*args,**karg)


#
# BV: the following functions have to be checked for their need
# and opportunity, and replaced by more general infrastrucuture
#

@deprecation(_future_deprecation)
def adjacencyList(elems):
    """Create adjacency lists for 2-node elements."""
    if len(elems.shape) != 2 or elems.shape[1] != 2:
        raise ValueError,"""Expected a set of 2-node elements."""
    elems = elems.astype(int)
    ok = [ where(elems==i) for i in range(elems.max()+1) ]
    return [ list(elems[w[0],1-w[1]]) for w in ok ]


@deprecation("adjacencyArray is deprecated. Use Connectivity().adjaccency('n')")
def adjacencyArray(index,maxcon=5):
    return Connectivity(index).adjacency('n')

    
# BV: Can this be replaced with a nodefront walker?
def adjacencyArrays(elems,nsteps=1):
    """Create adjacency arrays for 2-node elements.

    elems is a (nr,2) shaped integer array.
    The result is a list of adjacency arrays, where row i of adjacency array j
    holds a sorted list of the nodes that are connected to node i via a shortest
    path of j elements, padded with -1 values to create an equal list length
    for all nodes.
    This is: [adj0, adj1, ..., adjj, ... , adjn] with n=nsteps.

    Example:

    >>> adjacencyArrays([[0,1],[1,2],[2,3],[3,4],[4,0]],3)
    [array([[0],
           [1],
           [2],
           [3],
           [4]]), array([[1, 4],
           [0, 2],
           [1, 3],
           [2, 4],
           [0, 3]]), array([[2, 3],
           [3, 4],
           [0, 4],
           [0, 1],
           [1, 2]]), array([], shape=(5, 0), dtype=int64)]

    """
    elems = Connectivity(elems)
    if len(elems.shape) != 2 or elems.shape[1] != 2:
        raise ValueError,"""Expected a set of 2-node elements."""
    if nsteps < 1:
        raise ValueError, """The shortest path should be at least 1."""
    # Construct table of nodes connected to each node
    adj1 = elems.adjacency('n')
    m = adj1.shape[0]
    adj = [ arange(m).reshape(-1,1), adj1 ]
    nodes = adj1
    step = 2
    while step <= nsteps and nodes.size > 0:
        # Determine adjacent nodes
        t = nodes < 0
        nodes = adj1[nodes]
        nodes[t] = -1
        nodes = nodes.reshape((m,-1))
        nodes = reduceAdjacency(nodes)
        # Remove nodes of lower adjacency
        ladj = concatenate(adj[-2:],-1)
        t = [ in1d(n,l,assume_unique=True) for n,l in zip (nodes,ladj) ]
        t = asarray(t)
        nodes[t] = -1
        nodes = sortAdjacency(nodes)
        # Store current nodes
        adj.append(nodes)
        step += 1
    return adj


if __name__ == "__main__":

    C = Connectivity([[0,1],[2,3]],eltype='line2')
    print(C)
    print(C.eltype)
    print(C.report())
    print(C[0].report())
    print(C.selectNodes([1]))
    print(C.selectNodes([]))

    print(Connectivity().report())
    
    print connectedLineElems([[0,1],[0,2],[0,3],[4,5]])
# End
