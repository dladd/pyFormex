# $Id$

"""A collection of array utitlities.

These are general utility functions that depend on the numpy array model.
"""

from numpy import *


# default float and int types
Float = float32
Int = int32


def checkArray(a,shape=None,kind=None,allow=None):
    """Check that an array a has the correct shape and type.

    The input a is anything that can e converted into a numpy array.
    Either shape and or kind can be specified.
    The dimensions where shape contains a -1 value are not checked. The
    number of dimensions should match, though.
    If kind does not match, but is included in allow, conversion to the
    requested type is attempted.
    Returns the array if valid.
    Else, an error is raised.
    """
    try:
        a = asarray(a)
        shape = asarray(shape)
        w = where(shape >= 0)[0]
        if asarray(a.shape)[w] != shape[w]:
            raise
        if kind is not None:
            if allow is None and a.dtype.kind != kind:
                raise
            if kind == 'f':
                a = a.astype(Float)
        return a
    except:
        print "Expected shape %s, kind %s, got: %s" % (shape,kind,a)
    raise ValueError


def checkArray1D(a,size=None,kind=None,allow=None):
    """Check that an array a has the correct size and type.

    Either size and or kind can be specified.
    If kind does not match, but is included in allow, conversion to the
    requested type is attempted.
    Returns the array if valid.
    Else, an error is raised.
    """
    try:
        a = asarray(a).ravel()
        if (size is not None and a.size != size):
            raise
        if kind is not None:
            if allow is None and a.dtype.kind != kind:
                raise
            if kind == 'f':
                a = a.astype(Float)
        return a
    except:
        print "Expected size %s, kind %s, got: %s" % (size,kind,a)
    raise ValueError

   
def checkUniqueNumbers(nrs,nmin=0,nmax=None,error=None):
    """Check that an array contains a set of uniqe integers in range.

    nrs is an integer array with any shape.
    All integers should be unique and in the range(nmin,nmax).
    Beware: this means that    nmin <= i < nmax  !
    Default nmax is unlimited. Set nmin to None to
    error is the value to return if the tests are not passed.
    By default, a ValueError is raised.
    On success, None is returned
    """
    nrs = asarray(nrs)
    uniq = unique1d(nrs)
    if uniq.size != nrs.size or \
           (nmin is not None and uniq.min() < nmin) or \
           (nmax is not None and uniq.max() > nmax):
        if error is None:
            raise ValueError,"Values not unique or not in range"
        else:
            return error
    return uniq
    
    

# End
