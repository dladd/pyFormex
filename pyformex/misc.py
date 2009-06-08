# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
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
"""Python equivalents of the functions in lib.misc

The functions in this module should be exact emulations of the
external functions in the compiled library.
"""

# There should be no other imports here than array
from array import *

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


# End
