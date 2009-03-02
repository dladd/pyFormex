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

"""A collection of array utitlities.

These are general utility functions that depend on the numpy array model.
"""

from numpy import *
from pyformex import odict


# default float and int types
Float = float32
Int = int32


def growAxis(a,size,fill=0,axis=0):
    """Grow a single array axis to the given size and fill with given value."""
    if axis >= len(a.shape):
        raise ValueError,"No such axis number!"
    if size <= a.shape[axis]:
        return a
    else:
        missing = list(a.shape)
        missing[axis] = size-missing[axis]
        return concatenate([a,fill * ones(missing,dtype=a.dtype)],axis=axis)


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
    

def collectOnLength(items):
    """Collect items with same length.

    a is a list of items of any type for which the function len()
    returns an integer value.
    The items are sorted in a number of bins, each containing the
    items with the same length.
    The return value is a tuple of:
    - a list of bins with the sorted items,
    - a list of indices of these items in the input list,
    - a list of lengths of the bins,
    - a list of the item length in each bin.
    """
    np = array([ len(e) for e in items ])
    itemlen = unique1d(np)
    itemnrs = [ where(np==p)[0] for p in itemlen ]
    itemgrps = [ odict.listSelect(items,i) for i in itemnrs ]
    itemcnt = [ len(i) for i in itemnrs ]
    return itemgrps,itemnrs,itemcnt,itemlen
    

# End
