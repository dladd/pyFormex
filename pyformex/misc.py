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
#
"""Python equivalents of the functions in lib.misc"""

import pyformex as GD
from formex import *
from numpy import *

# Default is to try using the compiled library
if GD.options.uselib is None:
    GD.options.uselib = True

# Try to load the library
success = False
if GD.options.uselib:
    try:
        from lib.misc import *
        GD.debug("Succesfully loaded the pyFormex compiled library")
        success = True
    except ImportError:
        GD.debug("Error while loading the pyFormex compiled library")
        GD.debug("Reverting to scripted versions")

if not success:
    GD.debug("Using the (slower) Python implementations")

    def nodalSum(val,elems,work,avg):
        """Compute the nodal sum of values defined on elements.

        val   : (nelems,nplex,nval) values at points of elements.
        elems : (nelems,nplex) nodal ids of points of elements.
        work  : (nnod,nval)  returns the summed values at the nodes 

        On return each value is replaced with the sum of values at that node.
        If avg=True, the values are replaced with the average instead.

        The summation is done inplace, so there is no return value!
        """
        nodes = unique1d(elems)
        for i in nodes:
            wi = where(elems==i)
            vi = val[wi]
            if avg:
                raise RuntimeError,"THIS DOES NOT WORK!!!!"
                vi = vi.sum(axis=0)/vi.shape[0]
            else:
                vi = vi.sum(axis=0)
            work[i] = vi
            val[wi] = vi


# Always define these until C implementation is done!
def average_close(a,tol=0.5):
    """Average values from an array according to some specification.

    The default is to have a direction that is nearly the same.
    a is a 2-dim array
    """
    if a.ndim != 2:
        raise ValueError,"array should be 2-dimensional!"
    n = normalize(a)
    nrow = a.shape[0]
    cnt = zeros(nrow,dtype=int32)
    while cnt.min() == 0:
        w = where(cnt==0)
        nw = n[w]
        wok = where(dotpr(nw[0],nw) >= tol)
        wi = w[0][wok[0]]
        cnt[wi] = len(wi)
        a[wi] = a[wi].sum(axis=0)
    return a,cnt

def nodalSum2(val,elems,tol):
    """Compute the nodal sum of values defined on elements.

    val   : (nelems,nplex,nval) values at points of elements.
    elems : (nelems,nplex) nodal ids of points of elements.
    work  : a work space (unused) 

    The return value is a tuple of two arrays:
    res:
    cnt
    On return each value is replaced with the sum of values at that node.
    """
    nodes = unique1d(elems)
    for i in nodes:
        wi = where(elems==i)
        vi = val[wi]
        ai,ni = average_close(vi,tol=tol)
        ai /= ni.reshape(ai.shape[0],-1)
        val[wi] = ai
