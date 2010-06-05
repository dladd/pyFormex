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

"""A collection of numerical array utilities.

These are general utility functions that depend only on the :mod:`numpy`
array model. All pyformex modules needing :mod:`numpy` should import
everything from this module::

   from arraytools import * 
"""

from numpy import *
from utils import deprecation,checkVersion


if checkVersion('python','2.5') < 0:
    print("""
This version of pyFormex was developed for Python 2.5.
We advice you to upgrade your Python version.
Getting pyFormex to run on Python 2.4 should be possible with a
a few adjustements. Make it run on a lower version is problematic.
""")
    sys.exit()
    
if checkVersion('python','2.6') >= 0:
    print("""
This version of pyFormex was developed for Python 2.5.
There should not be any major problem with running on version 2.6,
but if you encounter some problems, please contact the developers at
pyformex.berlios.de.
""")
    from itertools import combinations
else:
    def combinations(iterable, r):
        # combinations('ABCD', 2) --> AB AC AD BC BD CD
        # combinations(range(4), 3) --> 012 013 023 123
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = range(r)
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)


###########################################################################
##
##   some math functions
##
#########################

# Define a wrapper function for old versions of numpy
#

if unique1d([1],True)[0][0] == 0:
    # We have the old numy version
    import warnings
    warnings.warn("BEWARE: OLD VERSION OF NUMPY!!!! We advise you to upgrade NumPy!")
    def unique1d(a,return_indices=False):
        """Replacement for numpy's unique1d"""
        import numpy
        if return_indices:
            indices,uniq = numpy.unique1d(a,True)
            return uniq,indices
        else:
            return numpy.unique1d(a)


# default float and int types
Float = float32
Int = int32


###########################################################################
##
##   some math functions
##
#########################
   

def niceLogSize(f):
    """Return the smallest integer e such that 10**e > abs(f)."""
    return int(ceil(log10(abs(f))))
   

def niceNumber(f,approx=floor):
    """Return a nice number close to but not smaller than f."""
    n = int(approx(log10(f)))
    m = int(str(f)[0])
    return m*10**n

# pi is defined in numpy
# Deg is a multiplier to transform degrees to radians
# Rad is a multiplier to transform radians to radians
Deg = pi/180.
Rad = 1.

# Convenience functions: trigonometric functions with argument in degrees
# Should we keep this in ???


def sind(arg,angle_spec=Deg):
    """Return the sin of an angle in degrees.

    For convenience, this can also be used with an angle in radians,
    by specifying `angle_spec=Rad`.
    """
    return sin(arg*angle_spec)


def cosd(arg,angle_spec=Deg):
    """Return the cos of an angle in degrees.

    For convenience, this can also be used with an angle in radians,
    by specifying ``angle_spec=Rad``.
    """
    return cos(arg*angle_spec)


def tand(arg,angle_spec=Deg):
    """Return the tan of an angle in degrees.

    For convenience, this can also be used with an angle in radians,
    by specifying ``angle_spec=Rad``.
    """
    return tan(arg*angle_spec)


def dotpr (A,B,axis=-1):
    """Return the dot product of vectors of A and B in the direction of axis.

    The default axis is the last.
    """
    A = asarray(A)
    B = asarray(B)
    return (A*B).sum(axis)


def length(A,axis=-1):
    """Returns the length of the vectors of A in the direction of axis.

    The default axis is the last.
    """
    A = asarray(A)
    return sqrt((A*A).sum(axis))


def normalize(A,axis=-1):
    """Normalize the vectors of A in the direction of axis.

    The default axis is the last.
    """
    A = asarray(A)
    shape = list(A.shape)
    shape[axis] = 1
    Al = length(A,axis).reshape(shape)
#    if (Al == 0.).any():
#        raise ValueError,"Normalization of zero vector."
    return A/Al


def projection(A,B,axis=-1):
    """Return the (signed) length of the projection of vector of A on B.

    The default axis is the last.
    """
    d = dotpr(A,B,axis)
    Bl = length(B,axis)
    if (Bl == 0.).any():
        raise ValueError,"Projection on zero vector."
    return dotpr(A,B,axis)/length(B,axis)


def norm(v,n=2):
    """Return thr `n`-norm of the vector `v`.

    Default is the quadratic norm (vector length).
    `n == 1` returns the sum.
    ``n <= 0`` returns the max absolute value.
    """
    a = asarray(v).flat
    if n == 2:
        return sqrt((a*a).sum())
    if n > 2:
        return (a**n).sum()**(1./n)
    if n == 1:
        return a.sum()
    if n <= 0:
        return abs(a).max()
    return


def solveMany(A,b):
    """Solve many systems of linear equations.
    
    A is a (M,M,...) shaped array.
    b is a (M,...) shaped array.
    
    This solves all equations A[:,:,i].x = b[:,i].
    The return value is a (M,...) shaped array.
    """
    shape = b.shape
    n = shape[0]
    if A.shape != (n,)+shape:
        raise ValueError,"A(%s) and b(%s) have incompatible shape" % (A.shape,b.shape)
    A = A.reshape(n,n,-1)
    b = b.reshape(n,-1)
    x = column_stack([ linalg.solve(A[:,:,i],b[:,i]) for i in range(b.shape[1])])
    return x.reshape(shape)


# Build-in function for Python 2.6
def permutations(iterable, r=None):
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    if r > n:
        return
    indices = range(n)
    cycles = range(n, n-r, -1)
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


def inside(p,mi,ma):
    """Return true if point p is inside bbox defined by points mi and ma"""
    return p[0] >= mi[0] and p[1] >= mi[1] and p[2] >= mi[2] and \
           p[0] <= ma[0] and p[1] <= ma[1] and p[2] <= ma[2]


def isClose(values,target,rtol=1.e-5,atol=1.e-8):
    """Returns an array flagging the elements close to target.

    `values` is a float array, `target` is a float value.
    `values` and `target` should be broadcastable to the same shape.
    
    The return value is a boolean array with shape of `values` flagging
    where the values are close to target.
    Two values `a` and `b` are considered close if
    :math:`| a - b | < atol + rtol * | b |`
    """
    values = asarray(values)
    target = asarray(target) 
    return abs(values - target) < atol + rtol * abs(target) 


def unitVector(v):
    """Return a unit vector in the direction of v.

    `v` is either an integer specifying one of the global axes (0,1,2),
    or a 3-element array or compatible.
    """
    if type(v) is int:
        u = zeros((3),dtype=Float)
        u[v] = 1.0
    else:
        u = asarray(v,dtype=Float)
        ul = length(u)
        if ul <= 0.0:
            raise ValueError,"Zero length vector %s" % v
        u /= ul
    return u


def rotationMatrix(angle,axis=None,angle_spec=Deg):
    """Return a rotation matrix over angle, optionally around axis.

    The angle is specified in degrees, unless angle_spec=Rad is specified.
    If axis==None (default), a 2x2 rotation matrix is returned.
    Else, axis should specifying the rotation axis in a 3D world. It is either
    one of 0,1,2, specifying a global axis, or a vector with 3 components
    specifying an axis through the origin.
    In either case a 3x3 rotation matrix is returned.
    Note that:

    - rotationMatrix(angle,[1,0,0]) == rotationMatrix(angle,0) 
    - rotationMatrix(angle,[0,1,0]) == rotationMatrix(angle,1) 
    - rotationMatrix(angle,[0,0,1]) == rotationMatrix(angle,2)
      
    but the latter functions calls are more efficient.
    The result is returned as an array.
    """
    a = angle*angle_spec
    c = cos(a)
    s = sin(a)
    if axis==None:
        f = [[c,s],[-s,c]]
    elif type(axis) == int:
        f = [[0.0 for i in range(3)] for j in range(3)]
        axes = range(3)
        i,j,k = axes[axis:]+axes[:axis]
        f[i][i] = 1.0
        f[j][j] = c
        f[j][k] = s
        f[k][j] = -s
        f[k][k] = c
    else:
        t = 1-c
        X,Y,Z = axis
        f = [ [ t*X*X + c  , t*X*Y + s*Z, t*X*Z - s*Y ],
              [ t*Y*X - s*Z, t*Y*Y + c  , t*Y*Z + s*X ],
              [ t*Z*X + s*Y, t*Z*Y - s*X, t*Z*Z + c   ] ]
        
    return array(f)


def rotMatrix(u,w=[0.,0.,1.],n=3):
    """Create a rotation matrix that rotates axis 0 to the given vector.

    u is a vector representing the 
    Return either a 3x3(default) or 4x4(if n==4) rotation matrix.
    """
    u = unitVector(u)

    try:
        v = unitVector(cross(w,u))
    except:
        if w == [0.,0.,1.]:
            w = [0.,1.,0.]
            v = unitVector(cross(w,u))
        else:
            raise
        
    w = unitVector(cross(u,v))

    m = row_stack([u,v,w])
    
    if n != 4:
        return m
    else:
        a = identity(4)
        a[0:3,0:3] = m
        return a


def growAxis(a,size,axis=-1,fill=0):
    """Grow a single array axis to the given size and fill with given value."""
    if axis >= len(a.shape):
        raise ValueError,"No such axis number!"
    if size <= a.shape[axis]:
        return a
    else:
        missing = list(a.shape)
        missing[axis] = size-missing[axis]
        return concatenate([a,fill * ones(missing,dtype=a.dtype)],axis=axis)


def reverseAxis(a,axis=-1):
    """Reverse the elements along axis."""
    a = asarray(a)
    try:
        n = a.shape[axis]
    except:
        raise ValueError,"Invalid axis %s for array shape %s" % (axis,a.shape)
    return a.take(arange(n-1,-1,-1),axis)


def checkArray(a,shape=None,kind=None,allow=None):
    """Check that an array a has the correct shape and type.

    The input `a` is anything that can be converted into a numpy array.
    Either `shape` and/or `kind` can be specified. and will then be checked.
    The dimensions where `shape` contains a -1 value are not checked. The
    number of dimensions should match.
    If `kind` does not match, but the value is included in `allow`,
    conversion to the requested type is attempted.

    Returns the array if valid; else, an error is raised.
    """
    try:
        a = asarray(a)
        shape = asarray(shape)
        w = where(shape >= 0)[0]
        if (asarray(a.shape)[w] != shape[w]).any():
            raise
        if kind is not None:
            if allow is None and a.dtype.kind != kind:
                raise
            if kind == 'f':
                a = a.astype(Float)
        return a
    except:
        raise ValueError,"Expected shape %s, kind %s, got: %s, %s" % (shape,kind,a.shape,a.dtype.kind)
    


def checkArray1D(a,size=None,kind=None,allow=None):
    """Check that an array a has the correct size and type.

    Either size and or kind can be specified.
    If kind does not match, but is included in allow, conversion to the
    requested type is attempted.
    Returns the array if valid.
    Else, an error is raised.
    """
    try:
        a = asarray(a)#.ravel() # seems sensible not to ravel!
        if (size is not None and a.size != size):
            raise
        if kind is not None:
            if allow is None and a.dtype.kind != kind:
                raise
            if kind == 'f':
                a = a.astype(Float)
        return a
    except:
        print("Expected size %s, kind %s, got: %s" % (size,kind,a))
    raise ValueError
              

def checkUniqueNumbers(nrs,nmin=0,nmax=None):
    """Check that an array contains a set of unique integers in a given range.

    This functions tests that all integer numbers in the array are within the
    range math:`nmin <= i < nmax`
    
    nrs: an integer array of any shape.
    nmin: minimum allowed value. If set to None, the test is skipped.
    nmax: maximum allowed value + 1! If set to None, the test is skipped.
    Default range is [0,unlimited].

    If the numbers are no unique or one of the limits is passed, an error
    is raised. Else, the sorted list of unique values is returned.
    """
    nrs = asarray(nrs)
    uniq = unique1d(nrs)
    if uniq.size != nrs.size or \
           (nmin is not None and uniq.min() < nmin) or \
           (nmax is not None and uniq.max() > nmax):
        raise ValueError,"Values not unique or not in range"
    return uniq


def readArray(file,dtype,shape,sep=' '):
    """Read an array from an open file.

    This uses :func:`numpy.fromfile` to read an array with known shape and
    data type from an open file.
    The sep parameter can be specified as in fromfile.
    """
    shape = asarray(shape)
    size = shape.prod()
    return fromfile(file=file,dtype=dtype,count=size,sep=sep).reshape(shape)


def writeArray(file,array,sep=' '):
    """Write an array to an open file.

    This uses :func:`numpy.tofile` to write an array to an open file.
    The sep parameter can be specified as in tofile.
    """
    array.tofile(file,sep=sep)


def cubicEquation(a,b,c,d):
    """Solve a cubiq equation using a direct method.

    a,b,c,d are the (floating point) coefficients of a third degree
    polynomial equation::
  
      a * x**3  +  b * x**2  +  c * x  +  d   =   0

    This function computes the three roots (real and complex) of this equation
    and returns full information about their kind, sorting order, occurrence
    of double roots. It uses scaling of the variables to enhance the accuracy.
    
    The return value is a tuple (r1,r2,r3,kind), where r1,r2 and r3 are three
    float values and kind is an integer specifying the kind of roots.

    Depending on the value of `kind`, the roots are defined as follows:
    
    ====      ==========================================================
    kind      roots
    ====      ==========================================================
    0         three real roots r1 < r2 < r3
    1         three real roots r1 < r2 = r3
    2         three real roots r1 = r2 < r3
    3         three real roots r1 = r2 = r3
    4         one real root r1 and two complex conjugate roots with real
              part r2 and imaginary part r3; the complex roots are thus:
              r2+i*r3 en r2-i*r3, where i = sqrt(-1).
    ====      ==========================================================

    If the coefficient a==0, a ValueError is raised.
    """
    #print "Coeffs: %s" % str((a,b,c,d))
    if a == 0.0:
        raise ValueError,"Coeeficient a of cubiq equation should not be 0"

    e3 = 1./3.
    pie = pi*2.*e3
    r = b/a
    s = c/a
    t = d/a
    #print "r,s,t = %s" % str((r,s,t))

    # scale variable
    sc = max(abs(r),sqrt(abs(s)),abs(t)**e3)
    sc = 10**(int(log10(sc)))
    r = r/sc
    s = s/sc/sc
    t = t/sc/sc/sc
    #print "scaled (%s) r,s,t = %s" % (sc,str((r,s,t)))
    
    rx = r*e3
    p3 = (s-r*rx)*e3
    q2 = rx**3-rx*s/2.+t/2.
    #print "rx,p3,q2 = %s" % str((rx,p3,q2))
    
    q2s = q2*q2
    p3c = p3**3
    som = q2s+p3c
    #print "q2s,p3c,som = %s" % str((q2s,p3c,som))

    if som <= 0.0:

        # 3 different real roots
        ic = 0
        roots = [ -rx ] * 3
        rt = sqrt(-p3c)
        if abs(rt) > 0.0:
            phi = cos(-q2/rt)*e3
            rt = 2.*sqrt(-p3)
            roots += rt * cos(phi + [0.,+pie, -pie])
        
        # sort the 3 roots

        roots.sort()
        if roots[1] == roots[2]:
            ic += 1
        if roots[1] == roots[0]:
            ic += 2

    else: # som < 0.0
        #  1 real and 2 complex conjugate roots
        ic = 4
        som = sqrt(som)
        u = -q2+som
        u = sign(abs(u)**e3) * u
        v = -q2-som
        v = sign(abs(v)**e3) * v
        r1 = u+v
        r2 = -r1/2-rx
        r3 = (u-v)*sqrt(3.)/2.
        r1 = r1-rx
        roots = array([r1,r2,r3])

    # scale and return values
    roots *= sc
    return roots,ic
 

# THIS MAY BE FASTER THAN olist.collectOnLength, BUT IT IS DEPENDENT ON NUMPY

## def collectOnLength(items):
##     """Collect items with same length.

##     a is a list of items of any type for which the function len()
##     returns an integer value.
##     The items are sorted in a number of bins, each containing the
##     items with the same length.
##     The return value is a tuple of:
##     - a list of bins with the sorted items,
##     - a list of indices of these items in the input list,
##     - a list of lengths of the bins,
##     - a list of the item length in each bin.
##     """
##     np = array([ len(e) for e in items ])
##     itemlen = unique1d(np)
##     itemnrs = [ where(np==p)[0] for p in itemlen ]
##     itemgrps = [ olist.select(items,i) for i in itemnrs ]
##     itemcnt = [ len(i) for i in itemnrs ]
##     return itemgrps,itemnrs,itemcnt,itemlen


def unique1dOrdered(ar1, return_index=False, return_inverse=False):
    """
    Find the unique elements of an array.
    
    This works like numpy's unique1d, but uses a stable sorting algorithm.
    The returned index may therefore hold other entries for multiply
    occurring values. In such case, unique1dOrdered returns the first
    occurrence in the flattened array.
    The unique elements and the inverse index are allways the same as those
    returned by numpy's unique1d.

    Parameters
    ----------
    ar1 : array_like
        This array will be flattened if it is not already 1-D.
    return_index : bool, optional
        If True, also return the indices against `ar1` that result in the
        unique array.
    return_inverse : bool, optional
        If True, also return the indices against the unique array that
        result in `ar1`.

    Returns
    -------
    unique : ndarray
        The unique values.
    unique_indices : ndarray, optional
        The indices of the unique values. Only provided if `return_index` is
        True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the original array. Only provided if
        `return_inverse` is True.

    Examples
    --------
    >>> a = array([2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,7,8])
    >>> numpy.unique1d(a,True)
    (array([1, 2, 3, 4, 5, 6, 7, 8]), array([ 7,  0,  1, 10,  3,  4,  5,  6]))
    >>> unique1dOrdered(a,True)
    (array([1, 2, 3, 4, 5, 6, 7, 8]), array([7, 0, 1, 2, 3, 4, 5, 6]))

    Notice the difference in the 4-th entry of the second array.

    """
    import numpy as np
    ar = np.asanyarray(ar1).flatten()
    if ar.size == 0:
        if return_inverse and return_index:
            return ar, np.empty(0, np.bool), np.empty(0, np.bool)
        elif return_inverse or return_index:
            return ar, np.empty(0, np.bool)
        else:
            return ar

    if return_inverse or return_index:
        perm = ar.argsort(kind='mergesort')
        aux = ar[perm]
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            iperm = perm.argsort()
            if return_index:
                return aux[flag], perm[flag], iflag[iperm]
            else:
                return aux[flag], iflag[iperm]
        else:
            return aux[flag], perm[flag]

    else:
        ar.sort()
        flag = np.concatenate(([True], ar[1:] != ar[:-1]))
        return ar[flag]


def renumberIndex(index):
    """Renumber an index sequentially.

    Given a one-dimensional integer array with only non-negative values,
    and `max` being the highest value in it, the elements are replaced
    with new values in the range 0..max, such that identical numbers are
    allways replaced with the same number and the new values at their
    first occurrence form an increasing sequence 0..max.
    
    The return value is a one-dimensional integer array with length equal to
    max+1, holding the original values corresponding to the new value 0..max.

    Parameters
    ----------
    index : array_like, 1d, integer
        An array with non-negative integer values

    Returns
    -------
    index : ndarray, length `max`
        The orginal values that have been replaced with 0..max.

    See also
    --------
    inverseUniqueIndex: find the inverse mapping.
    """
    un,pos = unique1dOrdered(index,True)
    srt = pos.argsort()
    old = un[srt]
    return old


def inverseUniqueIndex(index):
    """Inverse an index.

    index is a one-dimensional integer array with *unique* non-negative values.

    The return value is the inverse index: each value shows the position
    of its index in the index array. The length of the inverse index is
    equal to maximum value in index plus one. Values not occurring in index
    get a value -1 in the inverse index.

    Remark that inverseUniqueIndex(index)[index] == arange(1+index.max()).
    The inverse index thus translates the unique index numbers in a
    sequential index.
    """
    index = asarray(index)
    inv = -ones(index.max()+1,dtype=index.dtype)
    inv[index] = arange(index.size,dtype=inv.dtype)
    return inv
    


def sortByColumns(A):
    """Sort an array on all its columns, from left to right.

    The rows of a 2-dimensional array are sorted, first on the first
    column, then on the second to resolve ties, etc..

    The result value is an index returning the order in which the rows
    have to be taken to obtain the sorted array.
    """
    keys = [A[:,i] for i in range(A.shape[1]-1,-1,-1)]
    return lexsort(keys)


def uniqueRows(A):
    """Return (the indices of) the unique rows of an 2-d array.

    The input is an (nr,nc) shaped array.
    The return value is a tuple of two indices:
    - uniq: an (nuniq) shaped array with the numbers of the unique rows from A
    - uniqid: an (nr) shaped array with the numbers of uniq corresponding to
      all the rows of the input array A.

    The order of the rows in uniq is determined by the sorting procedure.
    Currently, this is sortByColumns.
    """
    srt = sortByColumns(A)
    inv = inverseUniqueIndex(srt)
    A = A.take(srt,axis=0)
    ok = (A != roll(A,1,axis=0)).any(axis=1)
    w = where(ok)[0]
    uniqid = w.searchsorted(inv,side='right')-1
    uniq = srt[ok]
    return uniq,uniqid


if __name__ == "__main__":

    A = array([
        [1,2,3],
        [2,3,4],
        [5,6,7],
        [3,4,5],
        [1,2,3],
        [2,3,4],
        [3,4,5],
        [5,6,7],
        [1,2,3],
        [2,3,4],
        ])

    uniq,uniqid = uniqueRows(A)
    B = A[uniq]
    print(B)

    print(uniqid)
    AB = B.take(uniqid,axis=0)
    print(A-AB)
    
    
# End
