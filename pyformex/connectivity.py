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

"""A class and functions for handling nodal connectivity.

This module defines a specialized array class for representing nodal
connectivity. This is e.g. used in mesh models, where geometry is
represented by a set of numbered points (nodes) and the geometric elements
are described by refering to the node numbers.
In a mesh model, points common to adjacent elements are unique, and
adjacency of elements can easily be detected from common node numbers. 
"""

import pyformex as GD
from arraytools import *
from utils import deprecation

def encode2(cols,magic=0):
    """Encode two integer values into a single integer.

    cols is a (n,2) array of non-negative integers smaller than 2**31.
    The result is an (n) array of type int64, where each value is
    unique for each row of values in the input.
    The original input can be restored with decode2.

    If a magic value larger than the maximum integer in the table is
    given, it will be used. If not, it will be taken as the maximum+1.
    A negative magic value triggers a fastencode scheme.

    The return value is a tuple with the codes and the magic used.
    """
    cmax = cols.max()
    if cmax >= 2**31 or cols.min() < 0:
        raise ValueError,"Integer value too high (>= 2**31) in encode2"
        
    if cols.ndim != 2 or cols.shape[1] != 2:
        raise ValueError,"Invalid array (type %s, shape %s) in encode2" % (cols.dtype,cols.shape)
    
    if magic < 0:
        magic = -1
        cols = array(cols,copy=True,dtype=int32,order='C')
        codes = cols.view(int64)
    else:
        if magic <= cmax:
            magic = cmax + 1
        codes = cols[:,0].astype(int64) * magic + cols[:,1]
    return codes,magic

        
def decode2(codes,magic):
    """Decode an integer number into two integers.

    codes and magic are the result of an encode2() operation.
    This will restore the original two values for the codes.

    A negative magic value flags the fastencode option.
    """
    if magic < 0:
        cols = codes.view(int32).reshape(-1,2)
    else:
        cols = column_stack([codes/magic,codes%magic]).astype(int32)
    return cols


@deprecation("\n Use 'enmagic3' instead")
def magic_numbers(*args,**kargs):
    return enmagic3(*args,**kargs)

@deprecation("\n Use 'demagic3' instead")
def demagic(*args,**kargs):
    return enmagic3(*args,**kargs)


def enmagic3(elems,magic):
    elems = elems.astype(int64)
    elems.sort(axis=1)
    mag = ( elems[:,0] * magic + elems[:,1] ) * magic + elems[:,2]
    return mag


def demagic3(mag,magic):
    first2,third = mag / magic, mag % magic
    first,second = first2 / magic, first2 % magic
    return column_stack([first,second,third]).astype(int32)


############################################################################
##
##   class Connectivity
##
#########################
#

class Connectivity(ndarray):
    """A class for handling element/node connectivity.

    A connectivity object is a 2-dimensional integer array with all
    non-negative values.
    In this implementation, all values should be smaller than 2**31.
    
    Furthermore, all values in a row should be unique. This is not enforced
    at creation time, but a method is provided to check the uniqueness.

    Create a new Connectivity object
    --------------------------------
    Connectivity(data=[],dtyp=None,copy=False,nplex=0)
    
    - data: should be integer type and evaluate to an 2-dim array.
    - dtype: can be specified to force an integer type.
      By default set from data.
    - copy: can be set True to force copying the data. By default, the
      specified data will be used without copying, if possible.
    - nplex: can be specified to force a check on the plexitude of the
      data, or to set the plexitude for an empty Connectivity.
      An error will be raised if the specified data do not match the
      specified plexitude.

    A Connectivity object stores its maximum value found at creation time
    in an attribute _max.
    """

    def __new__(self,data=[],dtyp=None,copy=False,nplex=0):
        """Create a new Connectivity object."""
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
            self._max = ar.max()
            if self._max > 2**31-1 or ar.min() < 0:
                raise ValueError,"Negative or too large positive value in data"
            if nplex > 0 and ar.shape[1] != nplex:
                raise ValueError,"Expected data of plexitude %s" % nplex
        else:
            self._max = -1
            ar = ar.reshape(0,nplex)
            
        # Transform 'subarr' from an ndarray to our new subclass.
        ar = ar.view(self)

        # Other data
        self.rev = None

        return ar


    def nelems(self):
        """Return the number of elements in the Connectivity table."""
        return self.shape[0]
    
    def nplex(self):
        """Return the plexitude of the elements in the Connectivity table."""
        return self.shape[1]


    def encode(self,permutations=True,compact=True):
        """Encode the element connectivities into single integer numbers.

        Each row of numbers is encoded into a single integer value, so that
        equal rows result in the same number and different rows yield
        different numbers. Furthermore, enough information is kept to
        restore the original rows from these single integer numbers.

        - permutations: if True, two rows are considered equal if they contain
          contain the same numbers regardless of their order. If False, two
          rows are only equal if they contain the same number at the same
          position.
        - comapct: if True, the resulting numbering scheme will be the
          lowest available numbers: 0..nelems-1. Else, a non-compact
          number set may be returned.

        Returns a tuple codes,magic:
        - codes is an (nelems,) shaped array with the element code numbers,
        - magic is the information needed to restore the original rows from
          the codes. See Connectivity.decode() 
        """
        if permutations or compact:
            raise ValueError,"Permutations and compact are not yet implemented"
        return encode(self)


    @staticmethod
    def decode(codes,magic):
        """Decode element codes into a Connectivity table.

        This is the inverse operation of the Connectivity.encode() method.
        It recreates a Connectivity table from the (codes,magic) information.

        This is a static method, and should be invoked as
        ```Connectivity.decode(codes,magic)```.
        - codes: code numbers as returned by Connectivity.encode, or a subset
          thereof.
        - magic: the magic information as returned by Connectivity.encode.

        Returns a Connectivity table.
        """
        return Connectivity(decode(codes,magic))


    def testDegenerate(self):
        """Flag the degenerate elements (rows).

        A degenerate element is a row which contains at least two
        equal values. 

        This function returns an array with the value True or False
        for each row. The True values flag the degenerate rows.
        """
        srt = self.copy()
        srt.sort(axis=1)
        return (srt[:,:-1] == srt[:,1:]).any(axis=1)
        

    def listDegenerate(self):
        """Return a list with the numbers of the degenerate elements."""
        return arange(self.nelems())[self.testDegenerate()]


    def listNonDegenerate(self):
        """Return a list with the numbers of the non-degenerate elements."""
        return arange(self.nelems())[~self.testDegenerate()]


    def removeDegenerate(self):
        """Remove the degenerate elements from a Connectivity table.

        Degenerate elements are rows with repeating values.
        Returns a Connectivity with the degenerate elements removed.
        """
        return self[~self.testDegenerate()]

    
    def removeDoubles(self):
        """Remove doubles from a Connectivity list.

        Doubles are elements that consist of the same set of nodes,
        in any particular order.

        Currently, this is only implemented for plexitude up to 3.
        """
        if self.nplex() == 0:
            return self
        
#        elif self.nplex() == 1:
#            return Connectivity(unique1d(self),nplex=1)

        else:
            codes,magic = self.encode(False,False)
            ucodes,pos = unique1d(codes,True)
            return self[pos]
            

            
    def reverseIndex(self):
        """Return a reverse index for the connectivity table.

        This is equivalent to the function reverseIndex()
        """
        if self.rev is None:
            self.rev = reverseIndex(self)
        return self.rev


    def selectNodes(self,nodsel):
        """Return a connectivity table with a subset of the nodes.

        `nodsel` is an object that can be converted to a 1-dim or 2-dim
        array. Examples are a tuple of local node numbers, or a list
        of such tuples all having the same length.
        Each row of `nodsel` holds a list of local node numbers that
        should be retained in the new Connectivity table.
        """
        nodsel = asarray(nodsel)
        nplex = nodsel.shape[-1]

        #print "NPLEX = %s" % nplex
        nodsel = nodsel.reshape(-1,nplex)
        return Connectivity(self[:,nodsel].reshape(-1,nplex))


    # THIS SHOULD BE GENERALIZED FOR intermediate plexitude > 2
    def expand(self,edg=None):
        """Transform elems to edges and faces.

        Return a tuple edges,faces where
        
        - edges is an (nedges,2) int array of edges connecting two node numbers.
        - faces is an (nelems,nplex) int array with the edge numbers connecting
          each pair os subsequent nodes in the elements of elems.

        The order of the edges respects the node order, and starts with
        nodes 0-1.
        The node numbering in the edges is always lowest node number first.

        The inverse operation is available as the static method
        Connectivity.compress().
        """
        nelems,nplex = self.shape
        if edg is None:
            n = arange(nplex)
            edg = column_stack([n,roll(n,-1)])
        else:
            edg = asarray(edg)
            if edg.ndim != 2 or edg.shape[-1] != 2:
                raise ValueError,"edg should be a (n,2) shaped array!"
            
        alledges = self[:,edg].astype(int32).reshape(-1,2)
        # sort edge nodes with lowest number first
        alledges.sort()
        codes,magic = encode2(alledges,self._max+1)
        # keep the unique edge numbers
        uniq,uniqid = unique1d(codes,True)
        # uniq is sorted 
        uedges = uniq.searchsorted(codes)
        edges = decode2(uniq,magic)
        faces = uedges.reshape((nelems,nplex))
        return edges,faces


    # THIS SHOULD BE GENERALIZED
    @staticmethod
    def compress(hi,lo):
        """Compress two hierarchical Connectiovity levels to a single one.

        hi and lo are two hierarchical Connectivity tables, representing
        higher and lower level respecively. This means that the elements
        of hi hold numbers which point into lo to obtain the lowest level
        items.

        As an example, in a structure of triangles, hi could represent
        triangles defined by 3 edges and lo could represent edges defined
        by 2 vertices. The compress method will then result in a table
        with plexitude 3 defining the triangles in function of the vertices.

        This is the inverse operation of expandElems.
        The algorithm only works if all vertex numbers of an element are
        unique.
        """
        elems = lo[hi]
        elems1 = roll(elems,-1,axis=1)
        for i in range(elems.shape[1]):
            flags = (elems[:,i,1] != elems1[:,i,0]) * (elems[:,i,1] != elems1[:,i,1])
            elems[flags,i] = roll(elems[flags,i],1,axis=1)
        return Connectivity(elems[:,:,0])


############################################################################

@deprecation("\n Use 'Connectivity.expand()' instead")
def expandElems(elems):
    return Connectivity(elems).expand()
    
@deprecation("\n Use 'Connectivity.compress()' instead")
def compactElems(edges,faces):
    return Connectivity.compress(faces,edges)



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
    if len(index.shape) != 2:
        raise ValueError,"Index should be an integer array with dimension 2"
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
            uniq,pos = unique1d(col,True)
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


def adjacencyArray(elems,maxcon=3,neighbours=1):
    """Create adjacency array for 2-node elements.
    
    The n-ring neighbourhood of the nodes is calculated (n=neighbours).
    These are the nodes connected through maximum n elements.
    """
    if len(elems.shape) != 2 or elems.shape[1] != 2:
        raise ValueError,"""Expected a set of 2-node elements."""
    # STEP 1: calculate the one-ring neighbourhood (nodes connected
    # through one element)
    nr,nc = elems.shape
    mr = elems.max() + 1
    mc = maxcon*nc
    # start with all -1 flags, maxcon*nc columns (because in each column
    # of elems, some number might appear with multiplicity maxcon)
    adj = zeros((mr,mc),dtype=elems.dtype) - 1
    i = 0 # column in adj where we will store next result
    for c in range(nc):
        col = elems[:,c].copy()  # make a copy, because we will change it
        while(col.max() >= 0):
            # we still have values to process in this column
            uniq,pos = unique1d(col,True)
            #put the unique values at a unique position in reverse index
            ok = uniq >= 0
            if i >= adj.shape[1]:
                # no more columns available, expand it
                adj = concatenate([adj,zeros_like(adj)-1],axis=-1)
            adj[uniq[ok],i] = elems[:,1-c][pos[ok]]
            i += 1
            # remove the stored values from elems
            col[pos[ok]] = -1
    adj.sort(axis=-1)
    maxc = adj.max(axis=0)
    adj = adj[:,maxc>=0]
    # STEP 2: extend with nodes connected through 2, ... , 'neighbours' elements
    if neighbours > 1:
        adj0 = adj # one-ring neighbourhood
        adj1 = adj # last added neighbours
        n = len(adj)
        for i in range(neighbours-1):
            t = adj1<0
            adj1 = adj0[adj1]
            adj1[t] = -1
            adj1 = adj1.reshape(n,-1)
            adj = column_stack([adj,adj1])
        # remove the element itself
        k =arange(n)
        adj[adj == k.reshape(n,-1)] = -1
        # remove duplicate elements
        adj.sort(axis=-1)
        ncols = adj.shape[1]
        pos = (ncols-1) * ones((n,),dtype=int32)
        j = ncols-1
        while j > 0:
            j -= 1
            t = adj[:,j] < adj[k,pos]
            w = where(t)
            x = where(t == 0)
            pos[w] -= 1
            adj[w,pos[w]] = adj[w,j]
        pmin = pos.min()
        p = pos.max()
        while p > pmin:
            adj[pos==p,p] = -1
            pos[pos==p] -= 1
            p = pos.max()
        adj = adj[:,pmin+1:]
    return adj


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
    #print(adj)
    k =arange(n)
    # remove the element itself
    adj[adj == k.reshape(n,-1)] = -1
    adj.sort(axis=-1)
    #print(adj)
    ncols = adj.shape[1]
    pos = (ncols-1) * ones((n,),dtype=int32)
    #pos = column_stack([arange(n),pos])    
    j = ncols-1
    while j > 0:
        j -= 1
        #print(pos)
        #print(adj[k,pos])
        #print(adj[:,j])
        t = adj[:,j] < adj[k,pos]
        w = where(t)
        x = where(t == 0)
        pos[w] -= 1
        adj[w,pos[w]] = adj[w,j]
        #print(adj)

    pmin = pos.min()
    p = pos.max()
    while p > pmin:
        #print(pos==p)
        adj[pos==p,p] = -1
        pos[pos==p] -= 1
        p = pos.max()
        #print(adj)
    adj = adj[:,pmin+1:]
    #print(adj)
    return adj


def closedLoop(elems):
    """Check if a set of line elements form a closed curve.

    elems is a connection table of line elements, such as obtained
    from the fuse() method on a plex-2 Formex.

    The return value is a tuple of:
    
    - return code:
    
      - 0: the segments form a closed loop
      - 1: the segments form a single non-closed path
      - 2: the segments form multiple not connected paths
      
    - a new connection table which is equivalent to the input if it forms
      a closed loop. The new table has the elements in order of the loop.
    """
    def reverse_table(tbl,nrows):
        """Reverse the table of a connected line

        The first nrows rows of table are reversed in row and column order.
        """
        tbl[:nrows] = reverseAxis(reverseAxis(tbl[:nrows],0),1)


    srt = zeros_like(elems) - 1
    ie = 0
    je = 0
    rev = False
    k = elems[0][0] # remember startpoint
    while True:
        # Store an element that has been found ok
        if rev:
            srt[ie] = elems[je][[1,0]]
        else:
            srt[ie] = elems[je]
        elems[je] = [ -1,-1 ] # Done with this one
        j = srt[ie][1] # remember endpoint
        if j == k:
            break
        ie += 1

        # Look for the next connected element
        w = where(elems == j)
        if w[0].size == 0:
            # Try reversing
            w = where(elems == k)
            if w[0].size == 0:
                break
            else:
                j,k = k,j
                reverse_table(srt,ie)
        je = w[0][0]
        rev = w[1][0] == 1
    if any(srt == -1):
        ret = 2
    elif srt[-1][1] != srt[0][0]:
        ret = 1
    else:
        ret = 0
    return ret,srt


def connectedLineElems(elems):
    """Partition a segmented curve into connected segments.
    
    The input argument is a (nelems,2) shaped array of integers.
    Each row holds the two vertex numbers of a single line segment.

    The return value is a list of (nsegi,2) shaped array of integers. 
    """
    parts = []
    while elems.size != 0:
        closed,loop = closedLoop(elems)
        parts.append(loop[loop!=-1].reshape(-1,2))
        elems = elems[elems!=-1].reshape(-1,2)
    return parts

partitionSegmentedCurve = connectedLineElems


#######################################
####new encoding and decoding scheme#######


# THINGS TO DO:
#
# - RETURN SINGLE MAGIC INFORMATION ON ENCODING ? (codes,magic)
# - COMPACT THE WHOLE ARRAY AT ONCE ?
# - SORT VALUES IN THE axis=1 DIRECTION
# - REPLACE magic3, ...
# - ADD A FINAL RENUMBERING
    

def compact_encode2(data):
    """Encode two columns of integers into a single column.

    This is like encode2 but results in smaller encoded values, because
    the original values are first replaced by indices into the sets of unique
    values.
    This encoding scheme is therefore usable for repeated application
    on multiple columns.

    The return value is the list of codes, the magic value used in encoding,
    and the two sets of uniq values for the columns, needed to restore the
    original data. Decoding can be done with compact_decode2.
    """
    # We could use a single compaction vector?
    uniqa, posa = unique1d(data[:,0], return_inverse=True)
    uniqb, posb = unique1d(data[:,1], return_inverse=True)
    # We could insert the encoding directly here,
    # or use an encoding function with 2 arguments
    # to avoid the column_stack operation
    rt = column_stack([posa, posb])
    codes, magic = encode2(rt)
    return codes,magic,uniqa,uniqb


def compact_decode2(codes,magic,uniqa,uniqb):
    """Decodes a single integer value into the original 2 values.

    This is the inverse operation of compact_encode2.
    Thus compact_decode2(*compact_encode(data)) will return data.

    codes can be a subset of the encoded values, but the other 3 arguments
    should be exactly those from the compact_encode2 result.
    """
    # decoding returns the indices into the uniq numberings
    pos = decode2(codes,magic)
    return column_stack([uniqa[pos[:,0]],uniqb[pos[:,1]]])


def encode(data,compact=True):
    """Encode multiple columns of integer data into a single column.

    The preferable way to use this function is via
    Connectivity.encode()
    """
    magic = []
    codes = data[:,0]
    for i in range(1,data.shape[1]):
        cols = column_stack([codes,data[:,i]])
        codes,mag,uniqa,uniqb = compact_encode2(cols)
        # insert at the front so we can process in order
        magic.insert(0,(mag,uniqa,uniqb))

    return codes,magic


def decode(codes,magic):
    """Decode multiple columns from a single column of codes and the magic.

    The preferable way to use this function is via
    Connectivity.decode()
    """
    data = []
    # process in reverse direction
    for mag in magic:
        cols = compact_decode2(codes,mag[0],mag[1],mag[2])
        data.insert(0,cols[:,1])
        codes = cols[:,0]
    data.insert(0,codes)
    return column_stack(data)
        

############################################################################
#
# Testing
#

if __name__ == "__main__":

    import sys
    
    c = Connectivity([[0,2,3],[2,4,5]])
    print(c)
    print(c._max)
    print(c.nelems())
    print(c.nplex())
    print(c.reverseIndex())

    a = array([2**31-1, 2**31])
    print a
    print a.astype(int32)

    n = 10
    m = 9
    a = array(arange(n))
    cols = column_stack([a,a**m])

    #cols = array([[0,0],[1,1]],dtype=int32,order='C')
    print cols
    #print cols.strides
    #print cols.dtype
    #print 2**31
    codes,magic = encode2(cols,-1)
    print codes
    print magic
    cols = decode2(codes,magic)
    print cols
    codes,magic = encode2(cols,0)
    print codes
    print magic
    cols = decode2(codes,magic)
    print cols
    
    #### example with 4 columns ####
    print "======== testing new encode/decode ============="
    data = array([[6, 20,101, 2000 ],[20, 6, 120, 2020], [6, 20, 101, 2000], [4, 36, 200, 3002], [50, 2, 100, 2001], [4, 36, 430, 6004], [6, 20, 101, 2000], ])
    print data
    print "ENCODING"
    codes = encode(data)
    print codes[0]
    print "DECODING"
    decoded = decode(*codes)
    print decoded
    print "%s ERRORS" % (data-decoded).sum()

    print "======== using rolled data ============="
  
    for i in range(data.shape[0]):
        data[i] = roll(data[i],i)
    print data
    print "ENCODING"
    codes = encode(data)
    print codes[0]
    print "DECODING"
    decoded = decode(*codes)
    print decoded
    print "%s ERRORS" % (data-decoded).sum()

    print "======== using sorted data ============="
    data.sort(axis=1)
    print data
    print "ENCODING"
    codes = encode(data)
    print codes[0]
    print "DECODING"
    decoded = decode(*codes)
    print decoded
    print "%s ERRORS" % (data-decoded).sum()

    print "========== test encoding and decoding =========="
    C = Connectivity(random.randint(10,size=(200,1)))
    D = C.copy()     
    print C
    C.sort(axis=1)
    codes,magic = C.encode(False,False)
    print "%s ERRORS" % (Connectivity.decode(codes,magic) - C).sum()

    D = C.removeDoubles()
    print "%s UNIQUE ELEMENTS" % D.shape[0]
    print D
    


# End
