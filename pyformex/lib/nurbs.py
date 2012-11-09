# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
from __future__ import print_function

# There should be no other imports here but numpy
from math import factorial
from numpy import zeros

accelerated = False

def binomial(n,k):
    """Compute the binomial coefficient Cn,k.

    This computes the binomial coefficient Cn,k = fac(n) / fac(k) / fac(n-k).

    >>> print [ binomial(4,i) for i in range(5) ]
    [1, 4, 6, 4, 1]
    """
    f = factorial
    return f(n) / f(k) / f(n-k)


def allBernstein(n,u):
    """Compute the value of all n-th degree Bernstein polynomials.

    Parameters:

    - `n`: int, degree of the polynomials
    - `u`: float, parametric value where the polynomials are evaluated

    Returns: an (n+1,) shaped float array with the value of all n-th
    degree Bernstein polynomials B(i,n) at parameter value u.

    Algorithm A1.3 from 'The NURBS Book' p20.
    """
    # THIS IS NOT OPTIMIZED FOR PYTHON.
    B = zeros(n+1)
    B[0] = 1.0
    u1 = 1.0-u
    for j in range(1,n+1):
        saved = 0.0
        for k in range(j):
            temp = B[k]
            B[k] = saved + u1*temp
            saved = u * temp
        B[j] = saved
    return B

# End
