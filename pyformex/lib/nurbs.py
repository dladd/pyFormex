# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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
"""Python equivalents of the functions in lib.nurbs

The functions in this module should be exact emulations of the
external functions in the compiled library.
"""

# There should be no other imports here but numpy
from math import factorial

accelerated = False

def binomial(n,k):
    """Compute the binomial coefficient Cn,k.

    This computes the binomial coefficient Cn,k = fac(n) / fac(k) / fac(n-k).

    >>> print [ binomial(4,i) for i in range(5) ]
    [1, 4, 6, 4, 1]
    """
    f = factorial
    return f(n) / f(k) / f(n-k)


# End
