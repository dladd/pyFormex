# $Id$
##
##  This file is part of pyFormex
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
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##

"""A class for storing and handling adjacency tables.

This module defines a specialized array class for representing adjacency
of items of a single type. This is e.g. used in mesh models, to store
the adjacent elements.
"""

from arraytools import *

############### Utility functions ##############

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
    if adj.shape[1] > 0:
        adj.sort(axis=-1)      # sort rows
        maxc = adj.max(axis=0) # find maximum per column
        adj = adj[:,maxc>=0]   # retain columns with non-negative maximum
    return adj


def reduceAdjacency(adj):
    """Reduce an adjacency table.

    An adjacency table is an integer array where each row lists the numbers
    of the items that are connected to the item with number equal to the row
    index. Rows are padded with -1 values to create rows of equal length.

    A reduced adjacency table is one where each row:

    - does not contain the row index itself,
    - does not contain duplicates,
    - is sorted in ascending order,

    and that has at least one row without -1 value.

    Paramaters:

    - `adj`: an 2-D integer array with value >=0 or -1

    Returns: an integer array with shape (adj.shape[0],maxc), with
    maxc <= adj.shape[1], where row `i` retains the unique non-negative
    numbers of the original array except the value `i`, and is possibly
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


############################################################################
##
##   class Adjacency
##
####################
#

class Adjacency(ndarray):
    """A class for storing and handling adjacency tables.

    An adjacency table defines a neighbouring relation between elements of
    a single collection. The nature of the relation is not important, but
    should be a binary relation: two elements are either related or they are
    not.

    Typical applications in pyFormex are the adjacency tables for storing
    elements connected by a node, or by an edge, or by a node but not by an
    edge, etcetera.

    Conceptually the adjacency table corresponds with a graph. In graph
    theory however the data are usually stored as a set of tuples `(a,b)`
    indicating a connection between the elements `a` and `b`.
    In pyFormex elements are numbered consecutively from 0 to nelems-1, where
    nelems is the number of elements. If the user wants another numbering,
    he can always keep an array with the actual numbers himself.
    Connections between elements are stored in an efficient two-dimensional
    array, holding a row for each element. This row contains the numbers
    of the connected elements.
    Because the number of connections can be different for each
    element, the rows are padded with an invalid elements number (-1).

    A normalized Adjacency is one where all rows do not contain duplicate
    nonnegative entries and are sorted in ascending order and where no column
    contains only -1 values.
    Also, since the adjacency is defined within a single collection, no row
    should contain a value higher than the maximum row index.

    A new Adjacency table is created with the following syntax ::
    
      Adjacency(data=[],dtyp=None,copy=False,ncon=0,normalize=True)

    Parameters:
    
    - `data`: should be compatible with an integer array with shape
      `(nelems,ncon)`, where `nelems` is the number of elements and
      `ncon` is the maximum number of connections per element. 
    - `dtyp`: can be specified to force an integer type but is set by
      default from the passed `data`. 
    - `copy`: can be set True to force copying the data. By default, the
      specified data will be used without copying, if possible.
    - `ncon`: can be specified to force a check on the plexitude of the
      data, or to set the plexitude for an empty Connectivity.
      An error will be raised if the specified data do not match the
      specified plexitude.
    - `normalize`: boolean: if True (default) the Adjacency will be normalized
      at creation time.
    - `allow_self`: boolean: if True, connections of elements with itself are
      allowed. The default (False) will remove self-connections when the table
      is normalized.

    .. warning: The `allow_self` parameter is currently inactive.

    Example:

    >>> print Adjacency([[1,2,-1],
    ...                  [3,2,0],
    ...                  [1,-1,3],
    ...                  [1,2,-1],
    ...                  [-1,-1,-1]])
    [[-1  1  2]
     [ 0  2  3]
     [-1  1  3]
     [-1  1  2]
     [-1 -1 -1]]
    """
    #
    #  BV: WE SHOULD ADD A CONSISTENCY CHECK THAT WE HAVE BIDIRECTIONAL
    #      CONNECTIONS: if row a has a value b, row b should have a value a
    #
    
    #
    # :DEV
    # Because we have a __new__ constructor here and no __init__,
    # we have to list the arguments explicitely in the docstring above.
    #
    def __new__(clas,data=[],dtyp=None,copy=False,ncon=0,normalize=True,allow_self=False,bidirectional=False):
        """Create a new Adjacency table."""
        
        # Turn the data into an array, and copy if requested
        ar = array(data, dtype=dtyp, copy=copy)
        if ar.ndim < 2:
            if ncon > 0:
                ar = ar.reshape(-1,ncon)
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
            if maxval > ar.shape[0]-1:
                raise ValueError,"Too large element number (%s) for number of rows(%s)" % (maxval,ar.shape[0])
            if ncon > 0 and ar.shape[1] != ncon:
                raise ValueError,"Expected data with %s columns" % ncon
        else:
            maxval = -1
            ar = ar.reshape(0,ncon)
            
        # Transform 'subarr' from an ndarray to our new subclass.
        ar = ar.view(clas)

        if normalize:
            ar = reduceAdjacency(ar).view(clas)

        return ar


    def nelems(self):
        """Return the number of elements in the Adjacency table.

        """
        return self.shape[0]


    def maxcon(self):
        """Return the maximum number of connections for any element.

        This returns the row count of the Adjacency.
        """
        return self.shape[1]


    ### normalize ###


    def normalize(self):
        """Normalize an adjacency table.

        A normalized adjacency table is one where each row:

        - does not contain the row index itself,
        - does not contain duplicates,
        - is sorted in ascending order,

        and that has at least one row without -1 value.

        By default, an Adjacency is normalized when it is constructed.
        Performing operations on an Adjacency may however leave it in
        a non-normalized state. Calling this method will normalize it again.
        This can obviously also be obtained by creating a new Adjacency
        with self as data.

        Returns: an integer array with shape (adj.shape[0],maxc), with
        ``maxc <= adj.shape[1]``, where row `i` retains the unique non-negative
        numbers of the original array except the value `i`, and is possibly
        padded with -1 values.

        Example:

          >>> a = Adjacency([[ 0,  0,  0,  1,  2,  5],
          ...                [-1,  0,  1, -1,  1,  3],
          ...                [-1, -1,  0, -1, -1,  2],
          ...                [-1, -1,  1, -1, -1,  3],
          ...                [-1, -1, -1, -1, -1, -1],
          ...                [-1, -1,  0, -1, -1,  5]])
          >>> a.normalize()
          Adjacency([[ 1,  2,  5],
                 [-1,  0,  3],
                 [-1, -1,  0],
                 [-1, -1,  1],
                 [-1, -1, -1],
                 [-1, -1,  0]])
        """
        return Adjacency(self)
    

    ### operations ###

    def pairs(self):
        """Return all pairs of adjacent element.

        Returns an integer array with two columns, where each row contains
        a pair of adjacent elements. The element number in the first columne
        is always the smaller of the two element numbers.
        """
        p = [ [[i,j] for j in k if j >= 0] for i,k in enumerate(self[:-1]) if max(k) >= 0]
        p = row_stack(p)
        return p[p[:,1] > p[:,0]]
  
  
    def symdiff(self,adj):
        """Return the symmetric difference of two adjacency tables.

        Parameters:

        - `adj`: Adjacency with the same number of rows as `self`.

        Returns an adjacency table of the same length, where each
        row contains all the (nonnegative) numbers of the corresponding
        rows of self and adj, except those that occur in both.
        """
        if adj.nelems() != self.nelems():
            raise ValueError,"`adj` should have same number of rows as `self`"
        adj = concatenate([self,adj],axis=-1)
        adj = sortAdjacency(adj)
        dup = adj[:,:-1] == adj[:,1:] # duplicate items
        adj[dup] = -1
        adj = roll(adj,-1,axis=-1)
        adj[dup] = -1
        adj = roll(adj,1,axis=-1)
        return Adjacency(adj)


    ### frontal methods ###

    def frontFactory(self,startat=0,frontinc=1,partinc=1):
        """Generator function returning the frontal elements.

        This is a generator function and is normally not used directly,
        but via the :meth:`frontWalk` method.

        It returns an int array with a value for each element.
        On the initial call, all values are -1, except for the elements
        in the initial front, which get a value 0. At each call a new front
        is created with all the elements that are connected to any of the
        current front and which have not yet been visited. The new front 
        elements get a value equal to the last front's value plus the
        `frontinc`. If the front becomes empty and a new starting front is
        created, the front value is extra incremented with `partinc`.

        Parameters: see :meth:`frontWalk`.

        Example:

        >>> A = Adjacency([[1,2,-1],
        ...                  [3,2,0],
        ...                  [1,-1,3],
        ...                  [1,2,-1],
        ...                  [-1,-1,-1]])
        >>> for p in A.frontFactory(): print p
        [ 0 -1 -1 -1 -1]
        [ 0  1  1 -1 -1]
        [ 0  1  1  2 -1]
        [0 1 1 2 4]
        """
        p = -ones((self.nelems()),dtype=int)
        if self.nelems() <= 0:
            return

        # Remember current elements front
        elems = clip(asarray(startat),0,self.nelems())
        prop = 0
        while elems.size > 0:
            # Store prop value for current elems
            p[elems] = prop
            yield p

            prop += frontinc

            # Determine adjacent elements
            elems = unique(asarray(self[elems]))
            elems = elems[elems >= 0]
            elems = elems[p[elems] < 0 ]
            if elems.size > 0:
                continue

            # No more elements in this part: start a new one
            elems = where(p<0)[0]
            if elems.size > 0:
                # Start a new part
                elems = elems[[0]]
                prop += partinc


    def frontWalk(self,startat=0,frontinc=1,partinc=1,maxval=-1):
        """Walks through the elements by their node front.

        A frontal walk is executed starting from the given element(s).
        A number of steps is executed, each step advancing the front
        over a given number of single pass increments. The step number at
        which an element is reached is recorded and returned.

        Parameters:

        - `startat`: initial element numbers in the front. It can be a single
          element number or a list of numbers.
        - `frontinc`: increment for the front number on each frontal step.
        - `partinc`: increment for the front number when the front
        - `maxval`: maximum frontal value. If negative (default) the walk will
          continue until all elements have been reached. If non-negative,
          walking will stop as soon as the frontal value reaches this
          maximum.

        Returns: an array of integers specifying for each element in which step
        the element was reached by the walker.

        Example:

          >>> A = Adjacency([
          ...       [-1,  1,  2,  3],
          ...       [-1,  0,  2,  3],
          ...       [ 0,  1,  4,  5],
          ...       [-1, -1,  0,  1],
          ...       [-1, -1,  2,  5],
          ...       [-1, -1,  2,  4]])
          >>> print A.frontWalk()
          [0 1 1 1 2 2]
        """
        for p in self.frontFactory(startat=startat,frontinc=frontinc,partinc=partinc):
            if maxval >= 0:
                if p.max() > maxval:
                    break
        return p


# End
