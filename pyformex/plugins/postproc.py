# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""Postprocessing functions

Postprocessing means collecting a geometrical model and computed values
from a numerical simulation, and render the values on the domain.
"""

from arraytools import *


# Some functions to calculate a scalar value from a vector

def norm2(A):
    return sqrt(square(asarray(A)).sum(axis=-1))

def norm(A,x):
    return power(power(asarray(A),x).sum(axis=-1),1./x)

def max(A):
    return asarray(A).max(axis=-1)

def min(A):
    return asarray(A).min(axis=-1)


def frameScale(nframes=10,cycle='up',shape='linear'):
    """Return a sequence of scale values between -1 and +1.

    ``nframes`` : the number of steps between 0 and -1/+1 values.

    ``cycle``: determines how subsequent cycles occur:
      
      ``'up'``: ramping up
      
      ``'updown'``: ramping up and down
      
      ``'revert'``: ramping up and down then reverse up and down

    ``shape``: determines the shape of the amplitude curve:
    
      ``'linear'``: linear scaling
      
      ``'sine'``: sinusoidal scaling
    """
    s = arange(nframes+1)
    if cycle in [ 'updown', 'revert' ]:
        s = concatenate([s, fliplr(s[:-1].reshape((1,-1)))[0]])
    if cycle in [ 'revert' ]: 
        s = concatenate([s, -fliplr(s[:-1].reshape((1,-1)))[0]])
    return s.astype(float)/nframes


# End
