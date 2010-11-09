# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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


# THINGS TO DO:
#
# - RETURN SINGLE MAGIC INFORMATION ON ENCODING ? (codes,magic)
# - COMPACT THE WHOLE ARRAY AT ONCE ?
# - SORT VALUES IN THE axis=1 DIRECTION
# - REPLACE magic3, ...
# - ADD A FINAL RENUMBERING

 
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

        
def demagic2(codes,magic):
    """Decode an integer number into two integers.

    codes and magic are the result of an enmagic2() operation.
    This will restore the original two values for the codes.

    A negative magic value flags the fastencode option.
    """
    if magic < 0:
        cols = codes.view(int32).reshape(-1,2)
    else:
        cols = column_stack([codes/magic,codes%magic]).astype(int32)
    return cols


# These are the equivalents for enmagic2 en demagic2 for any plexitude
# The -2 versions however have a fastencode option

def enmagic(elems):
    elems.sort(axis=1)
    magic = elems.max(axis=0) + 1
    prod = magic.prod()
    #print "magic = %s" % magic
    #print "product = %s" % prod
    #print "2**63   = %s" % 2**63
    #if prod >= 2**63 or prod < 0:
    #    raise RuntimeError,"There may be overflow in enmagic! Use encode instead" 
    codes = elems[:,0].astype(int64)
    i = 0
    for m in magic[1:]:
        codes = codes * m
        i += 1
        codes = codes + elems[:,i]
    return codes,magic

def demagic(codes,magic):
    nelems = len(codes)
    nplex = len(magic)
    elems = zeros((nelems,nplex),int32)
    i = nplex-1
    while i > 0:
        m = magic[i]
        elems[:,i] = codes % m
        codes /= m
        i -= 1
    elems[:,0] = codes
    return elems


def inverseIndex(index,maxcon=4):
    """Return an inverse index.

    Index is an (nr,nc) array of integers, where only non-negative
    integers are meaningful, and negative values are silently ignored.
    A Connectivity is a suitable argument.

    The inverse index is an integer array,
    where row i contains all the row numbers of index that contain
    the number i. Because the number of rows containing the number i
    is usually not a constant, the resulting array will have a number
    of columns mr corresponding to the highest row-occurrence of any
    single number. Shorter rows are padded with -1 values to flag
    non-existing entries.

    Negative numbers in index are disregarded.
    The return value is an (mr,mc) shaped integer array where:
    - mr will be equal to the highest positive value in index, +1.
    - mc will be equal to the highest multiplicity of any number in index.
    
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
            uniq,pos = unique(col,True)
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


@deprecation("\n Use 'Connectivity.inverse()' instead")
def reverseIndex(*args,**kargs):
   return inverseIndex(*args,**kargs)


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


    def encode(self,permutations=True,return_magic=False):
        """Encode the element connectivities into single integer numbers.

        Each row of numbers is encoded into a single integer value, so that
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


    #    This algorithm is faster than encode,
    #    but for nplex=2 a enmagic2 would probably still be faster.
    
    def testDoubles(self,permutations=True):
        """Test the Connectivity list for doubles.

        By default, doubles are elements that consist of the same set of
        nodes, in any particular order. Setting permutations to False
        will only find the double rows that have matching values at
        every position.

        This function returns a tuple with two arrays:
        - an index used to sort the elements
        - a flags array with the value True for indices of the unique elements
          and False for those of the doubles.
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
    

    def listUnique(self):
        """Return a list with the numbers of the unique elements."""
        ind,ok = self.testDoubles()
        return ind[ok]


    def listDoubles(self):
        """Return a list with the numbers of the double elements."""
        ind,ok = self.testDoubles()
        return ind[~ok]

   
    def removeDoubles(self,permutations=True):
        """Remove doubles from a Connectivity list.

        By default, doubles are elements that consist of the same set of
        nodes, in any particular order. Setting permutations to False
        will only remove the double rows that have matching values at
        matching positions.

        Returns a new Connectivity with the double elements removed.
        """
        ind,ok = self.testDoubles(permutations)
        return self[ind[ok]]


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
        nodsel = nodsel.reshape(-1,nplex)
        return Connectivity(self[:,nodsel].reshape(-1,nplex))


    def insertLevel(self,nodsel):
        """Insert an extra hierarchical level in a Connectivity table.

        A Connectivity table identifies higher hierchical entities in function
        of lower ones. This function will insert an extra hierarchical level.
        For example, if you have volumes defined in function of points,
        you can insert an intermediate level of edges, or faces.
        The return value is a tuple of two Connectivities (hi,lo), where:
        - hi: defines the original elements in function of the intermediate
          level ones,
        - lo: defines the intermediate level items in function of the lowest
          level ones.
        Intermediate level items that consist of the same items in any
        permutation order are collapsed to single items.
        The low level items respect the numbering order inside the
        original elements, but it is undefined which of the collapsed
        sequences is returned.

        There is currently no inverse operation, because the precise order
        of the items in the collapsed rows is lost.
        """
        nmult,nplex = nodsel.shape
        lo = self.selectNodes(nodsel)
        srt = lo.copy()
        srt.sort(axis=1)
        uniq,uniqid = uniqueRows(srt)
        hi = Connectivity(uniqid.reshape(-1,nmult))
        lo = lo[uniq]
        return hi,lo

    
    def untangle(self,ind=None):
        """Untangle a Connectivity into lower plexitude tables.

        There is no point in untangling a plexitude 2 structure.
        Plexitudes lower than 2 can not be untangled.
        Default is to untangle to plex-2 data (as in polygon to line segment).
        
        Return a tuple edges,faces where
        
        - edges is an (nedges,2) int array of edges connecting two node numbers.
        - faces is an (nelems,nplex) int array with the edge numbers connecting
          each pair os subsequent nodes in the elements of elems.

        The order of the edges respects the node order, and starts with
        nodes 0-1.
        The node numbering in the edges is always lowest node number first.

        For untangled Connectivities obtained with the default indices,
        an inverse operation is available as hi.tangle(lo).
        Degenerate rows may come back as a permutation!
        """
        nelems,nplex = self.shape
        
        if ind is None:
            # Default is to untangle to a 2-plex structure
            if nplex > 2:
                n = arange(nplex)
                ind = column_stack([n,roll(n,-1)])
            elif nplex == 2:
                print("There is no point in untangling a 2-plex Connectivity\nI'll go ahead anyway")
                ind = array([[0,1]])
            else:
                raise RuntimeError,"Can not untangle a Connectivity with plexitude < 2"
        else:
            ind = asarray(ind)
            if ind.ndim != 2 or ind.shape[-1] != 2:
                raise ValueError,"ind should be a (n,2) shaped array!"

        return self.insertLevel(ind)
    

    def tangle(self,lo):
        """Compress two hierarchical Connectivity levels to a single one.

        self and lo are two hierarchical Connectivity tables, representing
        higher and lower level respectively. This means that the elements
        of self hold numbers which point into lo to obtain the lowest level
        items.

        In the current implementation, the plexitude of lo should be 2!

        As an example, in a structure of triangles, hi could represent
        triangles defined by 3 edges and lo could represent edges defined
        by 2 vertices. The compress method will then result in a table
        with plexitude 3 defining the triangles in function of the vertices.

        This is the inverse operation of untangle (without specifying ind).
        The algorithm only works if all vertex numbers of an element are
        unique.
        """
        if self.shape[1] < 3:
            raise ValueError,"Can only tangle plexitudes >2 Connectivities"
        if lo.shape[1] != 2:
            raise ValueError,"Expected plexitudes ==2 for tangle argument"
        elems = lo[self]
        elems1 = roll(elems,-1,axis=1)
        for i in range(elems.shape[1]):
            flags = (elems[:,i,1] != elems1[:,i,0]) * (elems[:,i,1] != elems1[:,i,1])
            elems[flags,i] = roll(elems[flags,i],1,axis=1)
        return Connectivity(elems[:,:,0])


    def inverse(self):
        """Return the inverse index of a Connectivity table"""
        return inverseIndex(self)


    @classmethod
    def connect(clas,clist,nodid=None,bias=None,loop=False):
        """Return a Connectivity which connects the nodes of the Connectivity list.

        clist is a list of Connectivities, nodid is an optional list of nod ids and
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
        If loop==False, the lecgth of the Connectivity will be the minimum length
        of the Connectivities in clist, each minus its respective bias. By setting
        loop=True however, each Connectivity will loop around if its end is
        encountered, and the length of the result is the maximum length in clist.
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
    


############################################################################


@deprecation("\n Use 'arraytools.inverseUniqueIndex()' instead")
def reverseUniqueIndex(*args):
    return inverseUniqueIndex(*args)


def adjacencyList(elems):
    """Create adjacency lists for 2-node elements."""
    if len(elems.shape) != 2 or elems.shape[1] != 2:
        raise ValueError,"""Expected a set of 2-node elements."""
    elems = elems.astype(int)
    ok = [ where(elems==i) for i in range(elems.max()+1) ]
    return [ list(elems[w[0],1-w[1]]) for w in ok ]


def adjacencyArray(elems,maxcon=5):
    """Create adjacency array for 2-node elements.

    elems is a (nr,2) shaped integer array.
    The result is an integer array with shape (nr,mc), where row i holds
    a sorted list of the nodes that are connected to node i, padded with
    -1 values to create an equal list length for all nodes.
    """
    if len(elems.shape) != 2 or elems.shape[1] != 2:
        raise ValueError,"""Expected a set of 2-node elements."""
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
            uniq,pos = unique(col,True)
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
    return adj
    

def adjacencyArrays(elems,nsteps=1):
    """Create adjacency arrays for 2-node elements.

    elems is a (nr,2) shaped integer array.
    The result is a list of adjacency arrays, where row i of adjacency array j
    holds a sorted list of the nodes that are connected to node i via a shortest
    path of j elements, padded with -1 values to create an equal list length
    for all nodes.
    This is: [adj0, adj1, ..., adjj, ... , adjn] with n=nsteps.
    """
    if len(elems.shape) != 2 or elems.shape[1] != 2:
        raise ValueError,"""Expected a set of 2-node elements."""
    if nsteps < 1:
        raise ValueError, """The shortest path should be at least 1."""
    # Construct table of nodes connected to each node
    adj1 = adjacencyArray(elems)
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
        nodes.sort(axis=-1)
        maxc = nodes.max(axis=0)
        nodes = nodes[:,maxc>=0]
        # Remove duplicate nodes
        nodes[nodes[:,:-1] == nodes[:,1:]] = -1
        nodes.sort(axis=-1)
        maxc = nodes.max(axis=0)
        nodes = nodes[:,maxc>=0]
        # Remove nodes of lower adjacency
        ladj = concatenate(adj[-2:],-1)
        t = map(setmember1d,nodes,ladj)
        t = asarray(t)
        nodes[t] = -1
        nodes.sort(axis=-1)
        maxc = nodes.max(axis=0)
        nodes = nodes[:,maxc>=0]
        # Store current nodes
        adj.append(nodes)
        step += 1
    return adj


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


def adjacent(index,inv=None):
    """Return an index of connected elements.

    index is a (nr,nc) shaped integer array.
    An element j of index is said to be connected to element i, if element j
    has at least one (non-negative) value in common with element i.

    The result is an integer array with shape (nr,mc), where row i holds
    a sorted list of the elements that are connected to element i, padded with
    -1 values to create an equal list length for all elements.

    The result of this method provides the same information as repeated calls
    of connected(index,i), but may be more efficient if nr becomes large.

    The inverse index may be specified, if it was already computed.
    """
    n = index.shape[0]
    if inv is None:
        inv = inverseIndex(index)
    adj = inv[index].reshape((n,-1))
    #print(adj)
    k =arange(n)
    # remove the element itself
    adj[adj == k.reshape(n,-1)] = -1
    adj.sort(axis=-1)
    maxc = adj.max(axis=0)
    adj = adj[:,maxc>=0]
    # remove duplicate elements
    adj[adj[:,:-1] == adj[:,1:]] = -1
    adj.sort(axis=-1)
    maxc = adj.max(axis=0)
    adj = adj[:,maxc>=0]
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

   


############################################################################
#
# Testing
#

if __name__ == "__main__":

    import sys
    import utils

    def test_unique():
        C = Connectivity(random.randint(8,size=(20,3)))
        D = utils.timeEval("C.removeDoubles()",globals())
        print "%s UNIQUE ELEMENTS" % D.shape[0]
        print C
        print D
        print C.listDoubles()
        print C.listUnique()
    
    def test_encoding():
        print "========== test encoding and decoding =========="
        for nplex in range(1,5):
            print "PLEXITUDE %s" % nplex
            C = Connectivity(random.randint(10,size=(200,nplex)))
            #print C
            D = C.copy()     
            C.sort(axis=1)
            codes,magic = C.encode(return_magic=True)
            print "  %s ERRORS" % (Connectivity.decode(codes,magic) - C).sum()

            D = C.removeDoubles()
            print "  %s UNIQUE ELEMENTS" % D.shape[0]
            #print D

    def test_untangle():
        print "========== test untangling and tangling =========="
        for nplex in range(3,5):
            print "PLEXITUDE %s" % nplex
            C = Connectivity(random.randint(10,size=(10,nplex))).removeDegenerate()
            print C
            D,E = C.untangle()
            print D
            print E
            F = D.tangle(E)
            print F
            print "  %s ERRORS" % (F - C).sum()


    test_untangle()

# End
