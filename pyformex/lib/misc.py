# $Id$
##
##  This file is part of pyFormex 0.8.4 Release Sat Jul  9 14:43:11 2011
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
#
"""Python equivalents of the functions in lib.misc

The functions in this module should be exact emulations of the
external functions in the compiled library.
"""

# There should be no other imports here than numpy
import numpy as np


def _fuse(x,val,flag,sel,tol):
    """Fusing nodes.

    This is a low level function performing the internal loop of
    the fuse operation. It is not intended to be called by the user.
    """
    nnod = val.shape[0]
    nexti = 1
    for i in range(1,nnod):
        j = i-1
        while j>=0 and val[i]==val[j]:
            if abs(x[i]-x[j]).max() < tol:
                # node i is same as previous node j
                flag[i] = 0
                sel[i] = sel[j]
                break
            j = j-1
        if flag[i]:
            # node i is a new node
            sel[i] = nexti
            nexti += 1


def nodalSum(val,elems,work,avg):
    """Compute the nodal sum of values defined on elements.

    val   : (nelems,nplex,nval) values at points of elements.
    elems : (nelems,nplex) nodal ids of points of elements.
    work  : (nnod,nval)  returns the summed values at the nodes 

    On return each value is replaced with the sum of values at that node.
    If avg=True, the values are replaced with the average instead.

    The summation is done inplace, so there is no return value!
    """
    nodes = np.unique(elems)
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



def tofile_int32(val,fil,fmt):
    fmt = fmt * val.shape[-1]
    for row in val:
        fil.write(fmt % tuple(row))
        fil.write('\n')

tofile_float32 = tofile_int32

# End
