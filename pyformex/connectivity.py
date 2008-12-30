# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
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

"""connectivity.py

A pyFormex plugin for handling connectivity of nodes and elements.
"""

import pyformex as GD
from numpy import *

############################################################################
##
##   class Connectivity
##
#########################
#

class Connectivity(ndarray):
    """A class for handling element/node connectivity.

    A connectivity object is an 2-dimensional integer array with all
    non-negative values.
    In this implementation, al values should be lower than 2**31.
    """

    def __new__(self,data,dtyp=None,copy=False):
        """Create a new Connectivity object.

        data should be integer type and evaluate to an 2-dim array.
        If copy==True, the data are copied.
        If no dtype is given, that of data are used, or int32 by default.
        """
        # Turn the data into an array, and copy if requested
        ar = array(data, dtype=dtyp, copy=copy, ndmin=2)
        if len(ar.shape) != 2:
            raise ValueError,"Expected 2-dim data"

        # Make sure dtype is an int type
        if ar.dtype.kind != 'i':
            ar = ar.astype(Int)
 
        # Check values
        if ar.size > 0:
            self.magic = ar.max() + 1
            if self.magic > 2**31 or ar.min() < 0:
                raise ValueError,"Negative or too large positive value in data"
        else:
            self.magic = 0
            
        # Transform 'subarr' from an ndarray to our new subclass.
        ar = ar.view(self)

        # Other data
        self.rev = None

        return ar


    def nelems(self):
        return self.shape[0]
    
    def nplex(self):
        return self.shape[-1]

    def Max(self):
        if self.magic is None:
            self.magic = self.max() + 1
        return self.magic - 1
            
    def revIndex(self):
        if self.rev is None:
            self.rev = reverseIndex(self)
        return self.rev


############################################################################


def expandElems(elems):
    """Transform elems to edges and faces.

    elems is an (nelems,nplex) integer array of element node numbers.
    The maximum node number should be less than 2**31 or approx. 2 * 10**9 !!

    Return a tuple edges,faces where
    - edges is an (nedges,2) int32 array of edges connecting two node numbers.
    - faces is an (nelems,nplex) int32 array with the edge numbers connecting
      each pair os subsequent nodes in the elements of elems.

    The order of the edges respects the node order, and starts with nodes 0-1.
    The node numbering in the edges is always lowest node number first.

    The inverse operation is compactElems.
    """
    elems = asarray(elems)
    nelems,nplex = elems.shape
    magic = elems.max() + 1
    if magic > 2**31:
        raise RuntimeError,"Cannot compact edges for more than 2**31 nodes"
    n = arange(nplex)
    edg = column_stack([n,roll(n,-1)])
    alledges = elems[:,edg]
    # sort edge nodes with lowest number first
    alledges.sort()
    if GD.options.fastencode:
        edg = alledges.reshape((-1,2))
        codes = edg.view(int64)
    else:
        edg = alledges.astype(int64).reshape((-1,2))
        codes = edg[:,0] * magic + edg[:,1]
    # keep the unique edge numbers
    uniqid,uniq = unique1d(codes,True)
    # uniq is sorted 
    uedges = uniq.searchsorted(codes)
    edges = column_stack([uniq/magic,uniq%magic])
    faces = uedges.reshape((nelems,nplex))
    return edges,faces


def reverseUniqueIndex(index):
    """Reverse an index.

    index is a one-dimensional integer array with unique non-negative values.

    The return value is the reverse index: each value shows the position
    of its index in the index array. The length of the reverse index is
    equal to maximum value in index plus one. Values not occurring in index
    get a value -1 in the reverse index.

    Remark that reverseUniqueIndex(index)[index] == arange(1+index.max()).
    The reverse index thus translates the unique index numbers in a
    sequential index.
    """
    index = asarray(index)
    rev = zeros(1+index.max(),dtype=index.dtype) - 1
    rev[index] = arange(index.size,dtype=rev.dtype)
    return rev

    
def reverseIndex(index,maxcon=3):
    """Reverse an index.

    index is a (nr,nc) shaped integer array.

    The result is a (mr,mc) shaped integer array, where row i contains
    all the row numbers of index containing i.

    Negative numbers in index are disregarded.
    mr will be equal to the highest positive value in index, +1.
    mc will be equal to the highest multiplicity of any number in index.
    On entry, maxcon is an estimate for this value. The procedure will
    automatically change it if needed.

    Each row of the reverse index for a number that occurs less than mc
    times in index, will be filled up with -1 values.

    mult is the highest possible multiplicity of any number in a single
    column of index.
    """
    nr,nc = index.shape
    mr = index.max() + 1
    mc = maxcon*nc
    # start with all -1 flags, maxcon*nc columns (because in each column
    # of index, some number might appear with multiplicity maxcon)
    reverse = zeros((mr,mc),dtype=index.dtype) - 1
    i = 0 # column in reverse where we will store next result
    c = 0 # column in index from which to process data
    for c in range(nc):
        col = index[:,c].copy()  # make a copy, because we will change it
        while(col.max() >= 0):
            # we still have values to process in this column
            pos,uniq = unique1d(col,True)
            #put the unique values at a unique position in reverse index
            ok = uniq >= 0
            if i >= reverse.shape[1]:
                # no more columns available, expand it
                reverse = concatenate([reverse,zeros_like(reverse)-1],axis=-1)
            reverse[uniq[ok],i] = pos[ok]
            i += 1
            # remove the stored values from index
            col[pos[ok]] = -1

    reverse.sort(axis=-1)
    maxc = reverse.max(axis=0)
    reverse = reverse[:,maxc>=0]
    return reverse


def adjacencyList(elems):
    """Create adjacency lists for 2-node elements."""
    if len(elems.shape) != 2 or elems.shape[1] != 2:
        raise ValueError,"""Expected a set of 2-node elements."""
    elems = elems.astype(int)
    ok = [ where(elems==i) for i in range(elems.max()+1) ]
    return [ list(elems[w[0],1-w[1]]) for w in ok ]


def connected(index,i):
    """Return the list of elements connected to element i.

    index is a (nr,nc) shaped integer array.
    An element j of index is said to be connected to element i, iff element j
    has at least one (non-negative) value in common with element i.

    The result is a sorted list of unique element numbers, not containing
    the element number i itself.
    """
    adj = concatenate([ where(ind==j)[0] for j in ind[i] if j >= 0 ])
    return unique(adj[adj != i])


def adjacent(index,rev=None):
    """Return an index of connected elements.

    index is a (nr,nc) shaped integer array.
    An element j of index is said to be connected to element i, iff element j
    has at least one (non-negative) value in common with element i.

    The result is an integer array with shape (nr,mc), where row i holds
    a sorted list of the elements that are connected to element i, padded with
    -1 values to created an equal list length for all elements.

    The result of this method provides the same information as repeated calls
    of connected(index,i), but may be more efficient if nr becomes large.

    The reverse index may be specified, if it was already computed.
    """
    n = index.shape[0]
    if rev is None:
        rev = reverseIndex(index)
    adj = rev[index].reshape((n,-1))
    #print adj
    k =arange(n)
    # remove the element itself
    adj[adj == k.reshape(n,-1)] = -1
    adj.sort(axis=-1)
    #print adj
    ncols = adj.shape[1]
    pos = (ncols-1) * ones((n,),dtype=int32)
    #pos = column_stack([arange(n),pos])
    j = ncols-1
    while j > 0:
        j -= 1
        #print pos
        #print adj[k,pos]
        #print adj[:,j]
        t = adj[:,j] < adj[k,pos]
        w = where(t)
        x = where(t == 0)
        pos[w] -= 1
        adj[w,pos[w]] = adj[w,j]
        #print adj

    pmin = pos.min()
    p = pos.max()
    while p > pmin:
        #print pos==p
        adj[pos==p,p] = -1
        pos[pos==p] -= 1
        p = pos.max()
        #print adj
    adj = adj[:,pmin+1:]
    #print adj
    return adj


def closedLoop(elems):
    """Check if a set of line elements form a closed curve.

    elems is a connection table of line elements, such as obtained
    from the feModel() method on a plex-2 Formex.

    The return value is a tuple of:
    - return code:
      - 0: the segments form a closed loop
      - 1: the segments form a single non-closed path
      - 2: the segments form multiple not connected paths
    - a new connection table which is equivalent to the input if it forms
    a closed loop. The new table has the elements in order of the loop.
    """
    srt = zeros_like(elems) - 1
    ie = 0
    je = 0
    rev = False
    k = elems[je][0]
    while True:
        if rev:
            srt[ie] = elems[je][[1,0]]
        else:
            srt[ie] = elems[je]
        elems[je] = [ -1,-1 ] # Done with this one
        j = srt[ie][1]
        if j == k:
            break
        w = where(elems == j)
        if w[0].size == 0:
            print "No match found"
            break
        je = w[0][0]
        ie += 1
        rev = w[1][0] == 1
    if any(srt == -1):
        ret = 2
    elif srt[-1][1] != srt[0][0]:
        ret = 1
    else:
        ret = 0
    return ret,srt


def partitionSegmentedCurve(elems):
    """Partition a segmented curve into connected segments.
    
    The input argument is a (nelems,2) shaped array of integers.
    Each row holds the two vertex numbers of a single line segment.

    The return value ia a list of (nsegi,2) shaped array of integers. 
    
    is returned.
    Each border is a (nelems,2) shaped array of integers in
    which the element numbers are ordered.
    """
    borders = []
    while elems.size != 0:
        closed,loop = closedLoop(elems)
        borders.append(loop[loop!=-1].reshape(-1,2))
        elems = elems[elems!=-1].reshape(-1,2)
    return borders


############################################################################
#
# Testing
#

if __name__ == "__main__":

    c = Connectivity([[0,2,3],[2,4,5]])
    print c
    print c.magic
    print c.nelems()
    print c.nplex()
    print c.revIndex()
 
# End
