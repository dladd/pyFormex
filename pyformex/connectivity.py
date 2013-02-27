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

"""A class and functions for handling nodal connectivity.

This module defines a specialized array class for representing nodal
connectivity. This is e.g. used in mesh models, where geometry is
represented by a set of numbered points (nodes) and the geometric elements
are described by refering to the node numbers.
In a mesh model, points common to adjacent elements are unique, and
adjacency of elements can easily be detected from common node numbers.
"""
from __future__ import print_function

from arraytools import *
from adjacency import *
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
    are called duplicates.

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
      specified plexitude. If an `eltype` is specified, the plexitude of
      the element type will override this value.
    - `eltype`: an Element type (a subclass of :class:`Element`) or the name
      of an Element type, or None (default). If the Connectivity will be used
      to create a Mesh, the proper element type or name should be set: either
      here or at Mesh creation time. If the Connectivity will be used for other
      purposes, the element type may be not important.

    Example:

    >>> print(Connectivity([[0,1,2],[0,1,3],[0,3,2],[0,5,3]]))
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
    def __new__(self,data=[],dtyp=None,copy=False,nplex=0,eltype=None,allow_negative=False):
        """Create a new Connectivity object."""
        if isinstance(data,Connectivity):
            if nplex == 0:
                nplex = data.nplex()
            if eltype is None:
                eltype = data.eltype

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
        ar.eltype = eltype  # ! this may be a string!!!!!!!!!!!
        ar.inv = None   # inverse index
        return ar


    def __array_finalize__(self,obj):
        # reset the attributes from passed original object
        # all extra attributes added in __new__ should be reset here
        self.eltype = getattr(obj, 'eltype', None)
        self.inv = getattr(obj, 'inv', None)


    def __reduce__(self):
        """Reduce the object to a pickled state"""
        # Get the pickled ndarray state (as a list, so we can change it)
        object_state = list(ndarray.__reduce__(self))
        # Define our own state with the extra attributes we added
        subclass_state = (self.eltype,None)
        # Store both in place of the original ndarray state
        object_state[2] = (object_state[2],subclass_state)
        return tuple(object_state)


    def __setstate__(self,state):
        """Restore from pickled state"""
        # In __reduce__, we replaced ndarray's state with a tuple
        # of itself and our own state
        try:
            nd_state, own_state = state
            ndarray.__setstate__(self,nd_state)
            self.eltype,self.inv = own_state
        except:
            try:
                # try to read legacy pickle format, which did not save
                # the element type (rev < 2360)
                ndarray.__setstate__(self,state)
                print("WARNING: Connectivity was restored without element type!")
            except:
                print("I could not unpickle the Connectivity, neither in old not new format")
                raise


    def nelems(self):
        """Return the number of elements in the Connectivity table.

        Example:

          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2],[0,5,3]]).nelems()
          4
        """
        return self.shape[0]


    def maxnodes(self):
        """Return an upper limit for number of nodes in the connectivity.

        This returns the highest node number plus one.
        """
        return self.max() + 1


    def nnodes(self):
        """Return the actual number of nodes in the connectivity.

        This returns the count of the unique node numbers.
        """
        return unique(self).shape[0]


    def nplex(self):
        """Return the plexitude of the elements in the Connectivity table.

        Example:

          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2],[0,5,3]]).nplex()
          3
        """
        return self.shape[1]


    def report(self):
        """Format a Connectivity table"""
        s = "Conn %s, eltype=%s" % (self.shape,self.eltype)
        return s + '\n' + ndarray.__str__(self)

############### Detecting degenerates and duplicates ##############


    def testDegenerate(self):
        """Flag the degenerate elements (rows).

        A degenerate element is a row which contains at least two
        equal values.

        Returns a boolean array with shape (self.nelems(),).
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

        .. note:: If the Connectivity is part of a Mesh, you should use the
           Mesh.reduceDegenerate method instead, as that one will preserve
           the property numbers into the resulting Meshes.

        Example:

        >>> C = Connectivity([[0,1,2],[0,1,1],[0,3,2]],eltype='line3')
        >>> print(C.reduceDegenerate())
        [Connectivity([[0, 1]]), Connectivity([[0, 1, 2],
               [0, 3, 2]])]

        """
        from elements import elementType
        if self.eltype is None:
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


    def testDuplicate(self,permutations=True, return_multiplicity=False):
        """Test the Connectivity list for duplicates.

        By default, duplicates are elements that consist of the same set of
        nodes, in any particular order. Setting permutations to False
        will only find the duplicate rows that have matching values at
        every position.

        This function returns a tuple with two arrays and optionally a dictionary:

        - an index used to sort the elements
        - a flags array with the value True for indices of the unique elements
          and False for those of the duplicates.
        - if return_multiplicity is True it returns also an extra dict with
          multiplicities as keys and a list of elements as value.

        Example:

          >>> conn = Connectivity([[0,1,3],[2,3,0],[0,2,3],[0,1,2],[0,2,1],[0,3,2]])
          >>> print(conn)
          [[0 1 3]
           [2 3 0]
           [0 2 3]
           [0 1 2]
           [0 2 1]
           [0 3 2]]
          >>> ind,ok,D = conn.testDuplicate(return_multiplicity=True)
          >>> print(ind,ok,D)
          [3 4 0 1 2 5] [ True False  True  True False False] {'1': array([0]), '3': array([1, 2, 5]), '2': array([3, 4])}
          >>> print(ok.cumsum())
          [1 1 2 3 3 3]

        """
        if permutations:
            C = self.copy()
            C.sort(axis=1)
        else:
            C = self
        ind = sortByColumns(C)
        C = C.take(ind,axis=0)
        ok = (C != roll(C,1,axis=0)).any(axis=1)
        if not ok[0]: # all duplicates -> should result in one unique element
            ok[0] = True
        if return_multiplicity:
            cs = ok.cumsum()-1
            mult, uniq = multiplicity(cs)
            D={}
            for m in unique(mult):
                sel = where(mult==m)[0]
                w=ind[in1d(cs,sel)]
                D.update({'%d'%m:w})
            return ind,ok, D
        return ind,ok


    def listUnique(self,permutations=True):
        """Return a list with the numbers of the unique elements.

        Example:

          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).listUnique()
          array([0, 2])

        """
        ind,ok = self.testDuplicate(permutations)
        return ind[ok]


    def listDuplicate(self,permutations=True):
        """Return a list with the numbers of the duplicate elements.

        Example:

          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).listDuplicate()
          array([1])

        """
        ind,ok = self.testDuplicate(permutations)
        return ind[~ok]


    def removeDuplicate(self,permutations=True):
        """Remove duplicate elements from a Connectivity list.

        By default, duplicates are elements that consist of the same set of
        nodes, in any particular order. Setting permutations to False
        will only remove the duplicate rows that have matching values at
        matching positions.

        Returns a new Connectivity with the duplicate elements removed.

        Example:

          >>> Connectivity([[0,1,2],[0,2,1],[0,3,2]]).removeDuplicate()
          Connectivity([[0, 1, 2],
                 [0, 3, 2]])
        """
        ind,ok = self.testDuplicate(permutations)
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
          a permutation of the numbers in ``range(self.nelems()``. Each value
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
          ``arange(self.nelems()``, such that taking the elements in this order
          will produce a Connectivity reordered as requested. In case an
          explicit order was specified as input, this order is returned after
          checking that it is indeed a permutation of ``range(self.nelems()``.

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


    def renumber(self,start=0):
        """Renumber the nodes to a consecutive integer range.

        The node numbers in the table are changed thus that they
        form a consecutive integer range starting from the specified
        value.

        Returns a tuple:

        - `elems`: the renumbered connectivity
        - `oldnrs`: The sorted list of unique (old) node numbers. The new
          node numbers are assigned in order of increasing old node numbers,
          thus the old node number for new node number `i` can be found
          at position `i - start`.

        Example:

          >>> e,n = Connectivity([[0,2],[1,4],[4,2]]).renumber(7)
          >>> print(e,n)
          [[ 7  9]
           [ 8 10]
           [10  9]] [0 1 2 4]
        """
        nodes = asarray(unique(self))
        if nodes.size == 0:
            elems = self

        else:
            old = arange(nodes.max()+1)
            if nodes.shape[0] == old.shape[0]:
                # we have a consecutive range
                if nodes[0] == start:
                    # numbering is ok, keep
                    elems = self
                else:
                    # add the correct offset
                    elems = self + (start-nodes[0])
            else:
                # need to renumber
                elems = inverseUniqueIndex(nodes)[self] + start
                elems = Connectivity(elems,eltype=self.eltype)

        return elems,nodes


    def inverse(self):
        """Return the inverse index of a Connectivity table.

        Returns the inverse index of the Connectivity, as computed
        by :func:`arraytools.inverseIndex`.

        Example:

          >>> Connectivity([[0,1,2],[0,1,4],[0,4,2]]).inverse()
          array([[ 0,  1,  2],
                 [-1,  0,  1],
                 [-1,  0,  2],
                 [-1, -1, -1],
                 [-1,  1,  2]])
        """
        if self.inv is None:
            if self.size > 0:
                self.inv = inverseIndex(self)
            else:
                self.inv = Connectivity()
        return self.inv


    def nParents(self):
        """Return the number of elements connected to each node.

        Returns a 1-D int array with the number of elements connected
        to each node. The length of the array is equal to the highest
        node number + 1. Unused node numbers will have a count of zero.

        Example:

          >>> Connectivity([[0,1,2],[0,1,4],[0,4,2]]).nParents()
          array([3, 2, 2, 0, 2])
        """
        r = self.inverse()
        return (r>=0).sum(axis=1)


    def connectedTo(self,nodes):
        """Return a list of elements connected to the specified nodes.

        `nodes`: a single node number or a list/array thereof

        Returns an int array with the numbers of the elements that
        contain at least one of the specified nodes.

        Example:

          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2]]).connectedTo(2)
          array([0, 2])
        """
        ad = self.inverse()[nodes]
        return unique(ad[ad >= 0])


    def notConnectedTo(self,nodes):
        """Return a list of elements not connected to the specified nodes.

        `nodes`: a single node number or a list/array thereof

        Returns an int array with the numbers of the elements that
        do not contain any of the specified nodes.

        Example:

          >>> Connectivity([[0,1,2],[0,1,3],[0,3,2]]).notConnectedTo(2)
          array([1])
        """
        connected = self.connectedTo(nodes)
        return complement(connected,self.nelems())


    def adjacency(self,kind='e',mask=None):
        """Return a table of adjacent items.

        Create an element adjacency table (kind='e') or node adjacency
        table (kind='n').

        An element `i` is said to be adjacent to element `j`, if the two
        elements have at least one common node.

        A node `i` is said to be adjacent to node `j`, if there is at least
        one element containing both nodes.

        Parameters:

        - `kind`: 'e' or 'n', requesting resp. element or node adjacency.
        - `mask`: Either None or a boolean array or index flagging the nodes
          which are to be considered connectors between elements. If None,
          all nodes are considered connections.
          This option is only useful in the case `kind` == 'e'. If you want to
          use an element mask for the 'n' case, just apply the (element) mask
          beforehand::

            self[mask].adjacency('n')

        Returns:

        An Adjacency array with shape (nr,nc),
        where row `i` holds a sorted list of all the items that are
        adjacent to item `i`, padded with -1 values to create an equal
        list length for all items.

        Example:

          >>> Connectivity([[0,1],[0,2],[1,3],[0,5]]).adjacency('e')
          Adjacency([[ 1,  2,  3],
                 [-1,  0,  3],
                 [-1, -1,  0],
                 [-1,  0,  1]])
          >>> Connectivity([[0,1],[0,2],[1,3],[0,5]]).adjacency('e',mask=[1,2,3,5])
          Adjacency([[ 2],
                 [-1],
                 [ 0],
                 [-1]])
          >>> Connectivity([[0,1],[0,2],[1,3],[0,5]]).adjacency('n')
          Adjacency([[ 1,  2,  5],
                 [-1,  0,  3],
                 [-1, -1,  0],
                 [-1, -1,  1],
                 [-1, -1, -1],
                 [-1, -1,  0]])
          >>> Connectivity([[0,1,2],[0,1,3],[2,4,5]]).adjacency('n')
          Adjacency([[-1,  1,  2,  3],
                 [-1,  0,  2,  3],
                 [ 0,  1,  4,  5],
                 [-1, -1,  0,  1],
                 [-1, -1,  2,  5],
                 [-1, -1,  2,  4]])
          >>> Connectivity([[0,1,2],[0,1,3],[2,4,5]])[[0,2]].adjacency('n')
          Adjacency([[-1, -1,  1,  2],
                 [-1, -1,  0,  2],
                 [ 0,  1,  4,  5],
                 [-1, -1, -1, -1],
                 [-1, -1,  2,  5],
                 [-1, -1,  2,  4]])
        """
        inv = self.inverse()
        if kind == 'e':
            if mask is not None:
                mask = complement(mask,inv.shape[0])
                inv[mask] = -1
            adj = inv[self].reshape((self.nelems(),-1))
        elif kind == 'n':
            adj = concatenate([where(inv>=0,self[:,i][inv],inv) for i in range(self.nplex())],axis=1)
        else:
            raise ValueError,"kind should be 'e' or 'n', got %s" % str(kind)
        return Adjacency(adj)


######### Creating intermediate levels ###################

    def selectNodes(self,selector):
        """Return a :class:`Connectivity` containing subsets of the nodes.

        Parameters:

        - `selector`: an object that can be converted to a 1-dim or 2-dim
          int array. Examples are a tuple of local node numbers, or a list
          of such tuples all having the same length.
          Each row of `selector` holds a list of the local node numbers that
          should be retained in the new Connectivity table.

        Returns:

          A :class:`Connectivity` object with shape
          ``(self.nelems*selector.nelems,selector.nplex)``.
          This function does not collapse the duplicate elements. The eltype
          of the result is equal to that of the selector, possibly None.

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


    # BV: should we add a 'unique=False' option to create tables of
    # all intermediate entities without uniqifying?
    #
    def insertLevel(self,selector):
        """Insert an extra hierarchical level in a Connectivity table.

        A Connectivity table identifies higher hierarchical entities in
        function of lower ones. This method inserts an extra level in the
        hierarchy.
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

          If the Connectivity has an element type, selector can also be a
          single integer specifying one of the hierarchical levels of element
          entities (See the Element class). In that case the selector is
          constructed automatically from self.eltype.getEntities(selector).

        Returns:

        - `hi`: a :class:`Connectivity` defining the original elements
          in function of the intermediate level ones,
        - `lo`: a :class:`Connectivity` defining the intermediate level
          items in function of the lowest level ones (the original nodes).
          If the `selector` has an `eltype` attribute, then `lo` will inherit
          the same `eltype` value.

        All intermediate level items that consist of the same set of nodes
        in any permutation order and with any multiplicity, are considered
        identical and are collapsed into single items.
        The resulting node numbering of the created intermediate entities
        (the `lo` return value) respects the numbering order of the original
        elements and applied the selector, but it is undefined which of the
        collapsed sequences is returned.

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
           >>> Connectivity([[0,1,2,3]]).insertLevel([[0,1,2],[1,2,3],[0,1,1],[0,0,1],[1,0,0]])
           (Connectivity([[1, 2, 0, 0, 0]]), Connectivity([[0, 1, 1],
                  [0, 1, 2],
                  [1, 2, 3]]))
        """
        from elements import elementType

        if isInt(selector):
            if hasattr(self,'eltype'):
                sel = self.eltype.getEntities(selector)
            else:
                raise ValueError,"Specified an int as selector, but no eltype was defined"
        else:
            sel = Connectivity(selector)
        lo = self.selectNodes(sel)
        if lo.size > 0:
            if sel.testDegenerate().any():
                LO = lo.copy()
                # change the double entries to -1
                LO[LO[:,:-1] == LO[:,1:]] = -1
            else:
                LO = lo
            uniq,uniqid = uniqueRows(LO,permutations=True)
            hi = Connectivity(uniqid.reshape(-1,sel.nelems()))
            lo = lo[uniq]
        else:
            hi = lo = Connectivity()
        if hasattr(sel,'eltype'):
            lo.eltype = elementType(sel.eltype)
        return hi,lo


    # TODO: This is currently far from general!!!
    # should probably be moved to Mesh/TriSurface if needed there
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

          >>> hi,lo = Connectivity([[0,1,2],[0,2,1],[0,3,2]]).\
          insertLevel([[0,1],[1,2],[2,0]])
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


    # BV: UNTESTED !
    def reorderNodes(schemes,reverse=False):
        """_Convert Connectivity to/from foreign node numbering schemes.

        The order in which the element's nodes are numbered internally in
        pyFormex may be different than the numbering scheme used in external
        software packages. To allow correct export/import to/from other
        software, the nodes have to be renumbered.
        This function provides such a facility.

        Parameters:

        - `schemes`: a dictionary having pyFormex element names as keys and
          the matching nodal permutation arrays as values. The length of
          the array should match the plexitude of the Connectivity.
        - `reverse`: if True, the conversion is from external to internal.
          In this case, the Connectivity's eltype is interpreted as the
          pyFormex target element type (and should be set beforehand).

        Returns:

        - If the Connectivity has an element type and `scheme` has a key
          matching the element's name, a Connectivity with the renumbered
          elements is returned.

          - If `reverse` is False (default), the renumbering is done according
            to the permutation given by the `scheme` value matching the
            element name and the returned Connectivity will have no element
            type.

          - If `reverse` is True, the permutation scheme is reversed prior
            to using it. The target element type is retained in the returned
            Connectivity.

        - If the Connectivity has no element type or `scheme` has no matching
          key, the input Connectivity is returned unchanged.

        """
        if self.eltype is not None:
            eltype = ElementType(self.eltype)
            key = eltype.name()
            if scheme.haskey(key):
                print('key = %s' % key)
                trl = scheme[key]
                print('trl = %s' % trl)
                elems = self[trl]
                if not reverse:
                    delattr(self,'eltype')
                return elems

        return self


    def sharedNodes(self,elist):
        """Return the list of nodes shared by all elements in elist

        Parameters:

        - `elist`: an integer list-like with element numbers.

        Returns a 1-D integer array with the list of nodes that are
        common to all elements in the specified list. This array may be
        empty.
        """
        m,u = multiplicity(self[elist].ravel())
        return u[m==len(elist)]


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

    ## @deprecation("tangle has been renamed to combine")
    ## def tangle(self,*args,**kargs):
    ##     return self.combine(*args,**kargs)


    ## @deprecation("untangle has been deprecated. Use insertLevel instead.")
    ## def untangle(self,ind):
    ##     return self.insertLevel(ind)


    ## @deprecation(_future_deprecation)
    ## def encode(self,permutations=True,return_magic=False):
    ##     """_Encode the element connectivities into single integer numbers.

    ##     Each row of numbers is encoded into a single integer value, such that
    ##     equal rows result in the same number and different rows yield
    ##     different numbers. Furthermore, enough information can be kept to
    ##     restore the original rows from these single integer numbers.
    ##     This is seldom needed however, because the original data are
    ##     available from the Connectivity table itself.

    ##     - permutations: if True(default), two rows are considered equal if
    ##       they contain the same numbers regardless of their order.
    ##       If False, two rows are only equal if they contain the same
    ##       numbers at the same position.
    ##     - return_magic: if True, return a codes,magic tuple. The default is
    ##       to return only the codes.

    ##     Returns:

    ##     - codes: an (nelems,) shaped array with the element code numbers,
    ##     - magic: the information needed to restore the original rows from
    ##       the codes. See Connectivity.decode()

    ##     Example:

    ##       >>> Connectivity([[0,1,2],[0,1,3],[0,3,2]]).encode(return_magic=True)
    ##       (array([0, 1, 3]), [(2, array([0, 1]), array([2, 3])), (2, array([0]), array([1, 2]))])

    ##     *The use of this function is deprecated.*
    ##     """
    ##     def compact_encode2(data):
    ##         """Encode two columns of integers into a single column.

    ##         This is like enmagic2 but results in smaller encoded values, because
    ##         the original values are first replaced by indices into the sets of unique
    ##         values.
    ##         This encoding scheme is therefore usable for repeated application
    ##         on multiple columns.

    ##         The return value is the list of codes, the magic value used in encoding,
    ##         and the two sets of uniq values for the columns, needed to restore the
    ##         original data. Decoding can be done with compact_decode2.
    ##         """
    ##         # We could use a single compaction vector?
    ##         uniqa, posa = unique(data[:,0], return_inverse=True)
    ##         uniqb, posb = unique(data[:,1], return_inverse=True)
    ##         # We could insert the encoding directly here,
    ##         # or use an encoding function with 2 arguments
    ##         # to avoid the column_stack operation
    ##         rt = column_stack([posa, posb])
    ##         codes, magic = enmagic2(rt)
    ##         return codes,magic,uniqa,uniqb


    ##     if permutations:
    ##         data = self.copy()
    ##         data.sort(axis=1)
    ##     else:
    ##         data = self

    ##     magic = []
    ##     codes = data[:,0]
    ##     for i in range(1,data.shape[1]):
    ##         cols = column_stack([codes,data[:,i]])
    ##         codes,mag,uniqa,uniqb = compact_encode2(cols)
    ##         # insert at the front so we can process in order
    ##         magic.insert(0,(mag,uniqa,uniqb))

    ##     if return_magic:
    ##         return codes,magic
    ##     else:
    ##         return codes


    ## @deprecation(_future_deprecation)
    ## @staticmethod
    ## def decode(codes,magic):
    ##     """Decode element codes into a Connectivity table.

    ##     This is the inverse operation of the Connectivity.encode() method.
    ##     It recreates a Connectivity table from the (codes,magic) information.

    ##     This is a static method, and should be invoked as
    ##     ``Connectivity.decode(codes,magic)``.

    ##     - codes: code numbers as returned by Connectivity.encode, or a subset
    ##       thereof.
    ##     - magic: the magic information as returned by Connectivity.encode,
    ##       with argument return_magic=True.

    ##     Returns:

    ##     A Connectivity table.

    ##     Example:

    ##       >>> Connectivity.decode(array([0,1,3]), [(2, array([0, 1]), array([2, 3])), (2, array([0]), array([1, 2]))])
    ##       Connectivity([[0, 1, 2],
    ##              [0, 1, 3],
    ##              [0, 2, 3]])

    ##     *The use of this function is deprecated.*
    ##     """

    ##     def compact_decode2(codes,magic,uniqa,uniqb):
    ##         """Decodes a single integer value into the original 2 values.

    ##         This is the inverse operation of compact_encode2.
    ##         Thus compact_decode2(*compact_encode(data)) will return data.

    ##         codes can be a subset of the encoded values, but the other 3 arguments
    ##         should be exactly those from the compact_encode2 result.
    ##         """
    ##         # decoding returns the indices into the uniq numberings
    ##         pos = demagic2(codes,magic)
    ##         return column_stack([uniqa[pos[:,0]],uniqb[pos[:,1]]])

    ##     data = []
    ##     for mag in magic:
    ##         cols = compact_decode2(codes,mag[0],mag[1],mag[2])
    ##         data.insert(0,cols[:,1])
    ##         codes = cols[:,0]
    ##     data.insert(0,codes)
    ##     return Connectivity(column_stack(data))

############################################################################


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
      first and the last column are relevant.

    Returns:

    - `con`: a Connectivity with the same shape as the input Connectivity
      `elems`, holding a single chain extracted from the input and filled
      with -1 for the remainder (if any). The chain will not necessarily
      be the longest path. It will however at least contain the first element
      of the input table.

    - `inv`: an int array with two columns and number of rows equal to that of
      `con`. The first column holds the row number in `elems` of the entries
      in `con`. The second column holds a value +1 or -1, flagging whether
      the element is traversed in original direction (+1) in the chain or in
      the reverse direction (-1).

    .. warning:

       As a side-effect, all elements contained in the output Connectivity
       will have their entries in the input table `elems` changed to -1.

    Example:

      >>> con,inv = findConnectedLineElems([[0,1],[1,2],[0,4],[4,2]])
      >>> print(con)
      [[0 1]
       [1 2]
       [2 4]
       [4 0]]
      >>> print(inv)
      [[ 0  1]
       [ 1  1]
       [ 3 -1]
       [ 2 -1]]

      >>> con,inv = findConnectedLineElems([[0,1],[1,2],[0,4]])
      >>> print(con)
      [[2 1]
       [1 0]
       [0 4]]
      >>> print(inv)
      [[ 1 -1]
       [ 0 -1]
       [ 2  1]]

      >>> C = Connectivity([[0,1],[0,2],[0,3],[4,5]])
      >>> con,inv = findConnectedLineElems(C)
      >>> print(con)
      [[ 1  0]
       [ 0  2]
       [-1 -1]
       [-1 -1]]
      >>> print(inv)
      [[ 0 -1]
       [ 1  1]
       [-1  0]
       [-1  0]]
      >>> print(C)
      [[-1 -1]
       [-1 -1]
       [ 0  3]
       [ 4  5]]

    """

    #
    # BV: The side effect in this function could be removed,
    #     because we now have the information on the used elements
    #     in the ind array
    #     That would avoid the need to make a copy in connectedLineElems
    #

    if not isinstance(elems,Connectivity):
        elems = Connectivity(elems)
    #
    # srt will store the sorted connectivity
    #    initialize to -1 to make all invalid
    # ind will store the implicated element numbers and the direction
    #    first column element number, second 1 or -1 (reverse)
    #    first column initialized to -1, second to 0
    # ind is only needed for the return_indices argument in connectedLineElems
    #    but we compute it always for simplicity of the code
    #
    srt = zeros_like(elems) - 1
    ind = zeros((elems.shape[0],2),dtype=Int)
    ind[:,0] = -1
    ie = 0
    je = 0
    rev = False
    k = elems[0][0] # remember startpoint
    while True:
        # Store an element that has been found ok
        if rev:
            srt[ie] = elems[je,::-1]
            ind[ie] = ( je, -1 )
        else:
            srt[ie] = elems[je]
            ind[ie] = ( je, +1 )
        elems[je] = -1 # Done with this one
        j = srt[ie][-1] # remember endpoint
        if j == k:
            break
        ie += 1

        # Look for the next connected element (only thru first or last node!)
        w = where(elems[:,[0,-1]] == j)
        #print(w)
        if w[0].size == 0:
            # Try reversing
            w = where(elems[:,[0,-1]] == k)
            #print(w)
            if w[0].size == 0:
                break
            else:
                j,k = k,j
                # reverse the table (colums and rows)
                srt[:ie] = srt[ie-1::-1,::-1].copy()  # copy needed!!
                ind[:ie] = ind[ie-1::-1].copy() # rows only
                ind[:ie,1] *= -1 # change sign of 2nd column
        je = w[0][0]
        rev = w[-1][0] > 0 #check if the target node is the first or last

    return srt,ind


# BV: this could become a Connectivity function splitByConnection

def connectedLineElems(elems,return_indices=False):
    """Partition a collection of line segments into connected polylines.

    The input argument is a (nelems,2) shaped array of integers.
    Each row holds the two vertex numbers of a single line segment.

    The return value is a list of Connectivity tables of plexitude 2.
    The line elements of each Connectivity are ordered to form a continuous
    connected segment, i.e. the first vertex of each line element in a
    table is equal to the last vertex of the previous element.
    The connectivity tables are sorted in order of decreasing length.

    If return_indices = True, a second list of tables is returned, with
    the same shape as those in the first list. The tables of the second
    list contain in the first column the original element number of the
    entries, and in the second column a value +1 or -1 depending on
    whether the element traversal in the connected segment is in the
    original direction (+1) or the reverse (-1).

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
             [0, 2]]), Connectivity([[4, 5]]), Connectivity([[0, 3]])]

      >>> connectedLineElems([[0,1],[0,2],[0,3],[4,5]])
      [Connectivity([[1, 0],
             [0, 2]]), Connectivity([[4, 5]]), Connectivity([[0, 3]])]

      >>> connectedLineElems([[0,1],[0,2],[0,3],[4,5]],True)
      ([Connectivity([[1, 0],
             [0, 2]]), Connectivity([[4, 5]]), Connectivity([[0, 3]])], [array([[ 0, -1],
             [ 1,  1]], dtype=int32), array([[3, 1]], dtype=int32), array([[2, 1]], dtype=int32)])
      >>> connectedLineElems([[0,1,2],[2,0,3],[0,3,1],[4,5,2]])
      [Connectivity([[3, 0, 2],
             [2, 1, 0],
             [0, 3, 1]]), Connectivity([[4, 5, 2]])]

    Obviously, from the input elems table and the second return value,
    the first return value could be reconstructed::

      first = [
          where(i[:,-1:] > 0, elems[i[:,0]], elems[i[:,0],::-1])
          for i in second
      ]

    But since the construction of the first list is required by the algorithm,
    it is returned anyway.
    """
    elems = Connectivity(elems).copy() # make copy to avoid side effects
    elnrs =  arange(elems.shape[0]) # needed to return indices
    parts = []
    chains = []
    while elems.size != 0:
        loop,ind = findConnectedLineElems(elems)
        ind[:,0] = elnrs[ind[:,0]]
        parts.append(loop[(loop!=-1).any(axis=1)])
        chains.append(ind[(loop!=-1).any(axis=1)])
        todo = (elems!=-1).any(axis=1)
        elems = elems[todo]
        elnrs = elnrs[todo]
    # sort according to decreasing number of elements
    nel = [ p.nelems() for p in parts ]
    srt = argsort(nel)[::-1]
    parts = [ parts[i] for i in srt ]
    chains = [ chains[i] for i in srt ]
    if return_indices:
        return parts,chains
    else:
        return parts



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



#
# BV: the following functions have to be checked for their need
# and opportunity, and replaced by more general infrastrucuture
#

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
           [4]]), Adjacency([[1, 4],
           [0, 2],
           [1, 3],
           [2, 4],
           [0, 3]]), array([[2, 3],
           [3, 4],
           [0, 4],
           [0, 1],
           [1, 2]]), array([], shape=(5, 0), dtype=int64)]

    """
    #utils.warn("depr_adjacencyArrays")
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

    print(connectedLineElems([[0,1],[0,2],[0,3],[4,5]]))
# End
