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
"""Coordinate Systems.

"""
from __future__ import print_function

from coords import Coords
import numpy as np
import arraytools as at


###########################################################################
##
##   class CoordSys
##
#########################
#

#
# TODO: This should be redone:
#   - generalized: cartesian, cylindrical, spherical
#   - different initializations: e.g. as in arraytools.trfMatrix, matrix
#   - internal implementation is probably a 4x4 mat or (r,t) as in trfMatrix
#
class CoordinateSystem(Coords):
    """A CoordinateSystem defines a coordinate system in 3D space.

    The coordinate system is defined by and stored as a set of four points:
    three endpoints of the unit vectors along the axes at the origin, and
    the origin itself as fourth point.

    The constructor takes a (4,3) array as input. The default constructs
    the standard global Cartesian axes system::

      1.  0.  0.
      0.  1.  0.
      0.  0.  1.
      0.  0.  0.
    """
    def __new__(clas,coords=None,origin=None,axes=None):
        """Initialize the CoordinateSystem"""
        if coords is None:
            coords = np.eye(4,3)
            if axes is not None:
                coords[:3] = axes
            if origin is not None:
                coords += origin
        else:
            coords = at.checkArray(coords,(4,3),'f','i')
        coords = Coords.__new__(clas,coords)
        return coords


    def origin(self):
        """Return the origin of the CoordinateSystem."""
        return Coords(self[3])


    def axes(self):
        """Return the axes of the CoordinateSystem."""
        return Coords(self[:3]-self[3])


### End
