# $Id$

"""connectivity.py

A pyFormex plugin for handling connectivity of nodes and elements.

"""

from numpy import *

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
 
# End
