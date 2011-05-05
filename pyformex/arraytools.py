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

"""A collection of numerical array utilities.

These are general utility functions that depend only on the :mod:`numpy`
array model. All pyformex modules needing :mod:`numpy` should import
everything from this module::

  from arraytools import * 
"""

from numpy import *
import utils


if utils.checkVersion('python','2.6') >= 0:
    from itertools import combinations,permutations
else:
    # Provide our own implementation of combinations,permutations
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

    def permutations(iterable, r=None):
        # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
        # permutations(range(3)) --> 012 021 102 120 201 210
        pool = tuple(iterable)
        n = len(pool)
        if r is None:
            r = n
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

# Define a wrapper function for old versions of numpy

try:
    unique([1],True)
except TypeError:
    from numpy import unique1d as unique
   
if unique([1],True)[0][0] == 0:
    # We have the old numy version
    import warnings
    warnings.warn("BEWARE: OLD VERSION OF NUMPY!!!! We advise you to upgrade NumPy!")
    def unique(a,return_indices=False):
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

# pi is defined in numpy
# Deg is a multiplier to transform degrees to radians
# Rad is a multiplier to transform radians to radians
Deg = pi/180.
Rad = 1.
golden_ratio = 0.5 * (1.0 + sqrt(5.))


# Convenience functions: trigonometric functions with argument in degrees

def sind(arg,angle_spec=Deg):
    """Return the sin of an angle in degrees.

    For convenience, this can also be used with an angle in radians,
    by specifying `angle_spec=Rad`.

    >>> print sind(30), sind(pi/6,Rad)
    0.5 0.5
    """
    return sin(arg*angle_spec)


def cosd(arg,angle_spec=Deg):
    """Return the cos of an angle in degrees.

    For convenience, this can also be used with an angle in radians,
    by specifying ``angle_spec=Rad``.

    >>> print cosd(60), cosd(pi/3,Rad)
    0.5 0.5
    """
    return cos(arg*angle_spec)


def tand(arg,angle_spec=Deg):
    """Return the tan of an angle in degrees.

    For convenience, this can also be used with an angle in radians,
    by specifying ``angle_spec=Rad``.
    """
    return tan(arg*angle_spec)
   

def niceLogSize(f):
    """Return the smallest integer e such that 10**e > abs(f).

    This returns the number of digits before the decimal point.

    >>> print [ niceLogSize(a) for a in [1.3, 35679.23, 0.4, 0.00045676] ]
    [1, 5, 0, -3]
  
    """
    return int(ceil(log10(abs(f))))
   

def niceNumber(f,below=False):
    """Return a nice number close to f.

    f is a float number, whose sign is disregarded.

    A number close to abs(f) but having only 1 significant digit is returned.
    By default, the value is above abs(f). Setting below=True returns a
    value above.

    Example:

    >>> [ str(niceNumber(f)) for f in [ 0.0837, 0.837, 8.37, 83.7, 93.7] ]
    ['0.09', '0.9', '9.0', '90.0', '100.0']
    >>> [ str(niceNumber(f,below=True)) for f in [ 0.0837, 0.837, 8.37, 83.7, 93.7] ]
    ['0.08', '0.8', '8.0', '80.0', '90.0']

    """
    fa = abs(f)
    s = "%.0e" % fa
    m,n = map(int,s.split('e'))
    if not below:
        m = m+1
    return m*10.**n


def dotpr (A,B,axis=-1):
    """Return the dot product of vectors of A and B in the direction of axis.

    This multiplies the elements of the arrays A and B, and the sums the
    result in the direction of the specified axis. Default is the last axis.
    Thus, if A and B are sets of vectors in their last array direction, the
    result is the dot product of vectors of A with vectors of B.
    A and B should be broadcast compatible.

    >>> A = array( [[1.0, 1.0], [1.0,-1.0], [0.0, 5.0]] )
    >>> B = array( [[5.0, 3.0], [2.0, 3.0], [1.33,2.0]] )
    >>> print dotpr(A,B)
    [  8.  -1.  10.]

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


def horner(a,u):
    """Compute the value of a polynom using Horner's rule.

    Params:

    - `a`: float(n+1,nd), `nd`-dimensional coefficients of the polynom of
      degree `n`, starting from lowest degree.
    - `u`: float(nu), parametric values where the polynom is evaluated

    Returns:
    float(nu,nd), nd-dimensional values of the polynom.

    >>> print horner([[1.,1.,1.],[1.,2.,3.]],[0.5,1.0])
    [[ 1.5  2.   2.5]
     [ 2.   3.   4. ]]

    """
    a = asarray(a)
    u = asarray(u).reshape(-1,1)
    c = a[-1]
    for i in range(-2,-1-len(a),-1):
        c = c * u + a[i]
    return c


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


def anyVector(v):
    """Create a 3D vector.

    v is some data compatible with a (3)-shaped float array.
    Returns v as such an array.
    """
    return asarray(v,dtype=Float).reshape((3))


def unitVector(v):
    """Return a unit vector in the direction of v.

    `v` is either an integer specifying one of the global axes (0,1,2),
    or a 3-element array or compatible.
    """
    if type(v) is int:
        u = zeros((3),dtype=Float)
        u[v] = 1.0
    else:
        u = asarray(v,dtype=Float).reshape((3))
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
        X,Y,Z = unitVector(axis)
        t = 1.-c
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


def rotationAnglesFromMatrix(mat,angle_spec=Deg):
    """Return rotation angles from rotation matrix mat.
    
    This returns the three angles around the global axes 0, 1 and 2.
    The angles are returned in degrees, unless angle_spec=Rad.
    """
    rx = arctan(mat[1,2]/mat[2,2])
    ry = -arcsin(mat[0,2])
    rz = arctan(mat[0,1]/mat[0,0])
    R = dot(dot(rotationMatrix(rx,0,Rad),rotationMatrix(ry,1,Rad)),rotationMatrix(rz,2,Rad))
    T = isClose(mat,R,rtol=1.e-3,atol=1.e-5)
    w = where(~T.ravel())[0]
    w = w.tolist()
    if w == [3,4,5,6,7,8]:
        rx = pi + rx
    elif w == [0,1,3,4,6,7]:
        rz = pi + rz
    elif w == [0,1,5,8]:
        ry = pi - ry
    return rx / angle_spec, ry / angle_spec, rz / angle_spec


def vectorRotation(vec1,vec2,upvec=[0.,0.,1.]):
    """Return a rotation matrix for rotating vector vec1 to vec2

    The rotation matrix will be such that the plane of vec2 and the
    rotated upvec will be parallel to the original upvec.

    This function is like :func:`arraytools.rotMatrix`, but allows the
    specification of vec1.
    The returned matrix should be used in postmultiplication to the Coords.
    """
    u = normalize(vec1)
    u1 = normalize(vec2)
    w = normalize(upvec)
    v = normalize(cross(w,u))
    w = normalize(cross(u,v))
    v1 = normalize(cross(w,u1))
    w1 = normalize(cross(u1,v1))
    mat1 = column_stack([u,v,w])
    mat2 = row_stack([u1,v1,w1])
    mat = dot(mat1,mat2)
    return mat


def growAxis(a,add,axis=-1,fill=0):
    """Increase the length of a single array axis.

    The specified axis of the array `a` is increased with a value `add` and
    the new elements all get the value `fill`.

    Parameters:

    - `a`: array

    - `add`: int
      The value to add to the axis length. If <= 0, the unchanged array
      is returned.

    - `axis`: int
      The axis to change, default -1 (last).

    - `fill`: int or float
      The value to set the new elements to.

    Returns:
      An array with same dimension and type as `a`, but with a length along
      `axis` equal to ``a.shape[axis] + add``. The new elements all have the
      value `fill`.

    Example:

      >>> growAxis([[1,2,3],[4,5,6]],2)
      array([[1, 2, 3, 0, 0],
             [4, 5, 6, 0, 0]])

    """
    a = asarray(a)
    if axis >= len(a.shape):
        raise ValueError,"No such axis number!"
    if add <= 0:
        return a
    else:
        missing = list(a.shape)
        missing[axis] = add
        return concatenate([a,fill * ones(missing,dtype=a.dtype)],axis=axis)


def reorderAxis(a,order,axis=-1):
    """Reorder the planes of an array along the specified axis.

    The elements of the array are reordered along the specified axis
    according to the specified order. 

    Parameters:

    - `a`: array_like
    - `order`: specifies how to reorder the elements. It is either one
      of the special string values defined below, or else it is an index
      holding a permutation of `arange(self.nelems()`. Each value specifies the
      index of the old element that should be placed at its position.
      Thus, the order values are the old index numbers at the position of the
      new index number.

      `order` can also take one of the following predefined values,
      resulting in the corresponding renumbering scheme being generated:

      - 'reverse': the elements along axis are placed in reverse order
      - 'random': the elements along axis are placed in random order

    Returns:
      An array with the same elements of self, where only the order
      along the specified axis has been changed.

    Example::
    
      >>> reorderAxis([[1,2,3],[4,5,6]],[2,0,1])
      array([[3, 1, 2],
             [6, 4, 5]])
      
    """
    a = asarray(a)
    n = a.shape[axis]
    if order == 'reverse':
        order = arange(n-1,-1,-1)
    elif order == 'random':
        order = random.permutation(n)
    else:
        order = asarray(order)
    return a.take(order,axis)


def reverseAxis(a,axis=-1):
    """Reverse the elements along a computed axis.

    Example::
    
      >>> reverseAxis([[1,2,3],[4,5,6]],0)
      array([[4, 5, 6],
             [1, 2, 3]])
      
    Remark: if the axis is known in advance, it may be more efficient to use
    an indexing operation, like ``a[:,:::-1,:]``
    """
    return reorderAxis(a,'reverse',axis)


def addAxis(a,axis=0):
    """Add an additional axis with length 1 to an array.

    The new axis is inserted before the specified one. Default is to
    add it at the front.
    """
    s = list(a.shape)
    s[axis:axis] = [1]
    return a.reshape(s)


def stack(al,axis=0):
    """Stack a list of arrays along a new axis.

    al is a list of arrays all of the same shape.
    The return value is a new array with one extra axis, along which the
    input arrays are stacked. The position of the new axis can be specified,
    and is the first axis by default.
    """
    return concatenate([addAxis(ai,axis) for ai in al],axis=axis)


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
        raise ValueError,"Expected size %s, kind %s, got: %s" % (size,kind,a)
    

def checkArrayDim(a,ndim=-1):
    """Check that an array has the correct dimensionality.

    Returns asarray(a) if ndim < 0 or a.ndim == ndim
    Else, an error is raised.
    """
    try:
        aa = asarray(a)
        if (ndim >= 0 and aa.ndim != ndim):
            raise
        return aa
    except:
        raise ValueError,"Expected an array with %s dimensions" % ndim
              

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
    uniq = unique(nrs)
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

    Example:

      >>> cubicEquation(1.,-3.,3.,-1.)
      ([1.0, 1.0, 1.0], 3)
      
    """
    #
    # BV: We should return the solution of a second degree equation if a==0
    #
    if a == 0.0:
        raise ValueError,"Coeeficient a of cubiq equation should not be 0"

    e3 = 1./3.
    pie = pi*2.*e3
    r = b/a
    s = c/a
    t = d/a

    # scale variable
    sc = max(abs(r),sqrt(abs(s)),abs(t)**e3)
    sc = 10**(int(log10(sc)))
    r = r/sc
    s = s/sc/sc
    t = t/sc/sc/sc
    
    rx = r*e3
    p3 = (s-r*rx)*e3
    q2 = rx**3-rx*s/2.+t/2.
    
    q2s = q2*q2
    p3c = p3**3
    som = q2s+p3c

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
##     itemlen = unique(np)
##     itemnrs = [ where(np==p)[0] for p in itemlen ]
##     itemgrps = [ olist.select(items,i) for i in itemnrs ]
##     itemcnt = [ len(i) for i in itemnrs ]
##     return itemgrps,itemnrs,itemcnt,itemlen


def uniqueOrdered(ar1, return_index=False, return_inverse=False):
    """
    Find the unique elements of an array.
    
    This works like numpy's unique, but uses a stable sorting algorithm.
    The returned index may therefore hold other entries for multiply
    occurring values. In such case, uniqueOrdered returns the first
    occurrence in the flattened array.
    The unique elements and the inverse index are always the same as those
    returned by numpy's unique.

    Parameters:
    
    - `ar1` : array_like
        This array will be flattened if it is not already 1-D.
    - `return_index` : bool, optional
        If True, also return the indices against `ar1` that result in the
        unique array.
    - `return_inverse` : bool, optional
        If True, also return the indices against the unique array that
        result in `ar1`.

    Returns:
    
    - `unique` : ndarray
        The unique values.
    - `unique_indices` : ndarray, optional
        The indices of the unique values. Only provided if `return_index` is
        True.
    - `unique_inverse` : ndarray, optional
        The indices to reconstruct the original array. Only provided if
        `return_inverse` is True.

    Example::
    
      >>> a = array([2,3,4,5,6,7,8,1,2,3,4,5,6,7,8,7,8])
      >>> unique(a,True)
      (array([1, 2, 3, 4, 5, 6, 7, 8]), array([ 7,  0,  1, 10,  3,  4,  5,  6]))
      >>> uniqueOrdered(a,True)
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
    and `nval` being the number of different values in it, and you want to
    replace its elements with values in the range `0..nval`, such that
    identical numbers are always replaced with the same number and the
    new values at their first occurrence form an increasing sequence `0..nval`.
    This function will give you the old numbers corresponding with each
    position `0..nval`.

    Parameters:
    
    - `index`: array_like, 1-D, integer
      An array with non-negative integer values

    Returns:
      A 1-D integer array with length equal to `nval`, where `nval`
      is the number of different values in `index`, and holding the original
      values corresponding to the new value `0..nval`.

    Remark:
      Use :func:`inverseUniqueIndex` to find the inverse mapping
      needed to replace the values in the index by the new ones.

    Example::
    
      >>> renumberIndex([0,5,2,2,6,0])
      array([0, 5, 2, 6])
      >>> inverseUniqueIndex(renumberIndex([0,5,2,2,6,0]))[[0,5,2,2,6,0]]
      array([0, 1, 2, 2, 3, 0])
      
    """
    un,pos = uniqueOrdered(index,True)
    srt = pos.argsort()
    old = un[srt]
    return old


def inverseUniqueIndex(index):
    """Inverse an index.

    Given a 1-D integer array with *unique* non-negative values, and
    `max` being the highest value in it, this function returns the position
    in the array of the values `0..max`. Values not occurring in input index
    get a value -1 in the inverse index.

    Parameters:
    
    - `index`: array_like, 1-D, integer
      An array with non-negative values, wihch all have to be unique.

    Returns:
      A 1-D integer array with length `max+1`, with the positions in
      `index` of the values `0..max`, or -1 if the value does not occur in
      `index`.
    
    Remark:
      The inverse index translates the unique index numbers in a
      sequential index, so that
      ``inverseUniqueIndex(index)[index] == arange(1+index.max())``.
      

    Example::
    
      >>> inverseUniqueIndex([0,5,2,6])
      array([ 0, -1,  2, -1, -1,  1,  3])
      >>> inverseUniqueIndex([0,5,2,6])[[0,5,2,6]]
      array([0, 1, 2, 3])

    """
    ind = asarray(index)
    inv = -ones(ind.max()+1,dtype=ind.dtype)
    inv[ind] = arange(ind.size,dtype=inv.dtype)
    return inv
    

def sortByColumns(a):
    """Sort an array on all its columns, from left to right.

    The rows of a 2-dimensional array are sorted, first on the first
    column, then on the second to resolve ties, etc..

    Parameters:

    - `a`: array_like, 2-D

    Returns:
      A 1-D integer array specifying the order in which the rows have to
      be taken to obtain an array sorted by columns.
    
    Example::
    
      >>> sortByColumns([[1,2],[2,3],[3,2],[1,3],[2,3]])
      array([0, 3, 1, 4, 2])
      
    """
    A = checkArrayDim(a,2)
    keys = [A[:,i] for i in range(A.shape[1]-1,-1,-1)]
    return lexsort(keys)


def uniqueRows(a,permutations=False):
    """Return (the indices of) the unique rows of a 2-D array.

    Parameters:

    - `a`: array_like, 2-D
    - `permutations`: bool
      If True, rows which are permutations of the same data are considered
      equal. The default is to consider permutations as different.

    Returns:

    - `uniq`: a 1-D integer array with the numbers of the unique rows from `a`.
      The order of the elements in `uniq` is determined by the sorting
      procedure, which in the current implementation is :func:`sortByColumns`.
      If `permutations==True`, `a` is sorted along its axis -1 before calling
      this sorting function. 
    - `uniqid`: a 1-D integer array with length equal to `a.shape[0]` with the
      numbers of `uniq` corresponding to each of the rows of `a`.

    Example::
    
      >>> uniqueRows([[1,2],[2,3],[3,2],[1,3],[2,3]])
      (array([0, 3, 1, 2]), array([0, 2, 3, 1, 2]))
      >>> uniqueRows([[1,2],[2,3],[3,2],[1,3],[2,3]],permutations=True)
      (array([0, 3, 1]), array([0, 2, 2, 1, 2]))
    
    """
    A = array(a,copy=permutations)
    if A.ndim != 2:
        raise ValueError
    if permutations:
        A.sort(axis=-1)
    srt = sortByColumns(A)
    A = A.take(srt,axis=0)
    ok = (A != roll(A,1,axis=0)).any(axis=1)
    if not ok[0]: # all doubles -> should result in one unique element
        ok[0] = True
    w = where(ok)[0]
    inv = inverseUniqueIndex(srt)
    uniqid = w.searchsorted(inv,side='right')-1
    uniq = srt[ok]
    return uniq,uniqid


def argNearestValue(values,target):
    """Return the index of the item nearest to target.

    Parameters:

    - `values`: a list of float values
    - `target`: a float value

    Returns: the position of the item in `values` that is
    nearest to `target`.

    Example:
    
    >>> argNearestValue([0.1,0.5,0.9],0.7)
    1
    """
    v = array(values).ravel()
    c = v - target
    return argmin(c*c)


def nearestValue(values,target):
    """Return the item nearest to target.

    ``values``: a list of float values
    
    ``target``: a single value

    Return value: the item in ``values`` values that is
    nearest to ``target``.
    """
    return values[argNearestValue(values,target)]

#
# BV: This is a candidate for the C-library
#

def inverseIndex(index,maxcon=4):
    """Return an inverse index.

    An index is an array pointing at other items by their position.
    The inverse index is a collection of the reverse pointers.
    Negative values in the input index are disregarded.

    Parameters:
    
    - `index`: an array of integers, where only non-negative values are
      meaningful, and negative values are silently ignored. A Connectivity
      is a suitable argument.
    - `maxcon`: int: an initial estimate for the maximum number of rows a
      single element of index occurs at. The default will usually do well,
      because the procedure will automatically enlarge it when needed.

    Returns:
      An (mr,mc) shaped integer array where:
    
      - `mr` will be equal to the highest positive value in index, +1.
      - `mc` will be equal to the highest row-multiplicity of any number
        in `index`.

      Row `i` of the inverse index contains all the row numbers of `index`
      that contain the number `i`. Because the number of rows containing
      the number `i` is usually not a constant, the resulting array will have
      a number of columns `mc` corresponding to the highest row-occurrence of
      any single number. Shorter rows are padded with -1 values to flag
      non-existing entries.

    Example::
    
      >>> inverseIndex([[0,1],[0,2],[1,2],[0,3]])
      array([[ 0,  1,  3],
             [-1,  0,  2],
             [-1,  1,  2],
             [-1, -1,  3]])
    
    """
    ind = asarray(index)
    if len(ind.shape) != 2 or ind.dtype.kind != 'i':
        raise ValueError,"nndex should be an integer array with dimension 2"
    nr,nc = ind.shape
    mr = ind.max() + 1
    mc = maxcon*nc
    # start with all -1 flags, maxcon*nc columns (because in each column
    # of index, some number might appear with multiplicity maxcon)
    inverse = zeros((mr,mc),dtype=ind.dtype) - 1
    i = 0 # column in inverse where we will store next result
    for c in range(nc):
        col = ind[:,c].copy()  # make a copy, because we will change it
        while(col.max() >= 0):
            # we still have values to process in this column
            uniq,pos = unique(col,True)
            #put the unique values at a unique position in inverse index
            ok = uniq >= 0
            if i >= inverse.shape[1]:
                # no more columns available, expand it
                inverse = concatenate([inverse,zeros_like(inverse)-1],axis=-1)
            inverse[uniq[ok],i] = pos[ok]
            i += 1
            # remove the stored values from index
            col[pos[ok]] = -1

    inverse.sort(axis=-1)
    maxc = inverse.max(axis=0)
    inverse = inverse[:,maxc>=0]
    return inverse


def matchIndex(target,values):
    """Find position of values in target.

    This function finds the position in the array `target` of the elements
    from the array `values`.

    Parameters:

    - `target`: an index array with all non-negative values. If not 1-D, it
      will be flattened.
    - `values`: an index array with all non-negative values. If not 1-D, it
      will be flattened.

    Returns: an index array with the same size as `values`.
    For each number in `values`, the index contains the position of that value
    in the flattened `target`, or -1 if that number does not occur in `target`.
    If an element from `values` occurs more than once in `target`, it is
    currently undefined which of those positions is returned.

    Remark that after ``m = matchIndex(target,values)`` the equality
    ``values[m] == target`` holds in all the non-negative positions of `m`.

    Example::
    
      >>> A = array([1,3,4,5,7,8,9])
      >>> B = array([0,6,7,1,2])
      >>> matchIndex(A,B)
      array([-1, -1,  4,  0, -1])

    """
    target = target.reshape(-1,1)
    values = values.reshape(-1)
    inv = inverseIndex(target)[:,0]
    diff = values.max()-len(inv)+1
    if diff > 0:
        inv = concatenate([inv,-ones((diff,),dtype=Int)])
    return inv[values]

# Working with sets of vectors

def vectorLength(vec):
    """Return the lengths of a set of vectors.

    vec is an (n,3) shaped array holding a collection of vectors.
    The result is an (n,) shaped array with the length of each vector.
    """
    return length(vec)


def vectorNormalize(vec):
    """Normalize a set of vectors.

    vec is a (n,3) shaped arrays holding a collection of vectors.
    The result is a tuple of two arrays:

    - length (n): the length of the vectors vec
    - normal (n,3): unit-length vectors along vec.
    """
    length = vectorLength(vec)
    normal = vec / length.reshape((-1,1))
    return length,normal


def vectorPairAreaNormals(vec1,vec2):
    """Compute area of and normals on parallellograms formed by vec1 and vec2.

    vec1 and vec2 are (n,3) shaped arrays holding collections of vectors.
    The result is a tuple of two arrays:
    
    - area (n) : the area of the parallellogram formed by vec1 and vec2.
    - normal (n,3) : (normalized) vectors normal to each couple (vec1,2).
    
    These are calculated from the cross product of vec1 and vec2, which indeed
    gives area * normal.

    Note that where two vectors are parallel, an area zero will results and
    an axis with components NaN.
    """
    normal = cross(vec1,vec2)
    area = vectorLength(normal)
    normal /= area.reshape((-1,1))
    return area,normal


def vectorPairArea(vec1,vec2):
    """Compute area of the parallellogram formed by a vector pair vec1,vec2.

    vec1 and vec2 are (n,3) shaped arrays holding collections of vectors.
    The result is an (n) shaped array with the area of the parallellograms
    formed by each pair of vectors (vec1,vec2).
    """
    return vectorPairAreaNormals(vec1,vec2)[0]


def vectorPairNormals(vec1,vec2):
    """Compute vectors normal to vec1 and vec2.

    vec1 and vec2 are (n,3) shaped arrays holding collections of vectors.
    The result is an (n,3) shaped array of unit length vectors normal to
    each couple (edg1,edg2).
    """
    return vectorPairAreaNormals(vec1,vec2)[1]
        

def vectorTripleProduct(vec1,vec2,vec3):
    """Compute triple product vec1 . (vec2 x vec3).

    vec1, vec2, vec3 are (n,3) shaped arrays holding collections of vectors.
    The result is a (n,) shaped array with the triple product of each set
    of corresponding vectors fromvec1,vec2,vec3.
    This is also the square of the volume of the parallellepid formex by
    the 3 vectors.
    """
    return dotpr(vec1,cross(vec2,vec3))


def vectorPairCosAngle(v1,v2):
    """Return the cosinus of the angle between the vectors v1 and v2."""
    v1 = asarray(v1)
    v2 = asarray(v2)
    return dotpr(v1,v2) / sqrt(dotpr(v1,v1)*dotpr(v2,v2))


def vectorPairAngle(v1,v2):
    """Return the angle (in radians) between the vectors v1 and v2."""
    return arccos(vectorPairCosAngle(v1,v2))


def histogram2(a,bins,range=None):
    """Compute the histogram of a set of data.

    This function is a like numpy's histogram function, but also returns
    the bin index for each individual entry in the data set.

    Parameters:
    
    - `a`: array_like.
      Input data. The histogram is computed over the flattened array.

    - `bins`: int or sequence of scalars.
      If bins is an int, it defines the number of equal-width bins
      in the given range. If bins is a sequence, it defines the bin edges,
      allowing for non-uniform bin widths. Both the leftmost and rightmost
      edges are included, thus the number of bins is len(bins)-1.

    - `range`: (float, float), optional. The lower and upper range of the bins.
      If not provided, range is simply (a.min(), a.max()). Values outside the
      range are ignored. This parameter is ignored if bins is a sequence.

    Returns:

    - `hist`: integer array with length nbins, holding the number of elements
      in each bin,
    - `ind`: a sequence of nbins integer arrays, each holding the indices of
      the elements fitting in the respective bins,
    - `xbins`: array of same type as data and with length nbins+1:
      returns the bin edges.

    Example:

    >>> histogram2([1,2,3,4,2,3,1],[1,2,3,4,5])
    (array([2, 2, 2, 1]), [array([0, 6]), array([1, 4]), array([2, 5]), array([3])], array([1, 2, 3, 4, 5]))

    """
    ar = asarray(a)
    if type(bins) == int:
        nbins = bins
        xbins = linspace(a.min(),a.max(),nbins+1)
    else:
        xbins = asarray(bins)
        nbins = len(xbins)-1
    d = digitize(ar,xbins)
    ind = [ where(d==i)[0] for i in arange(1,nbins+1) ]
    hist = asarray([ i.size for i in ind ])
    return hist,ind,xbins
   
# End
