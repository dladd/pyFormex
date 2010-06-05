#!/usr/bin/env pyformex
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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
"""Some functions operating on 2D structures.

This is a plugin for pyFormex.
(C) 2002 Benedict Verhegghe

See the Section2D example for an example of its use.
"""

from plugins import sectionize
from numpy import *

## We should probably turn this into a class planeSection

class planeSection(object):
    """A class describing a general 2D section.

    The 2D section is the area inside a closed curve in the (x,y) plane.
    The curve is decribed by a finite number of points and by straight
    segments connecting them.
    """

    def __init__(self,F):
        """Initialize a plane section.

        Initialization can be done either by a list of points or a set of line
        segments.

        By Points:
          Each point is connected to the following one, and (unless they are
          very close) the last one back to the first. Traversing the resulting
          path should rotate positively around the z axis to yield a positive
          surface.

        By Segments:
          It is the responsibilty of the user to ensure that the segments
          form a closed curve. If not, the calculated section data will be
          rather meaningless.
        """
        if F.nplex() == 1:
            self.F = sectionize.connectPoints(F,close=True)
        elif F.nplex() == 2:
            self.F = F
        else:
            raise ValueError,"Expected a plex-1 or plex-2 Formex"

    def sectionChar(self):
        return sectionChar(self.F)


def sectionChar(F):
    """Compute characteristics of plane sections.

    The plane sections are described by their circumference, consisting of a
    sequence of straight segments.
    The segment end point data are gathered in a plex-2 Formex.
    The segments should form a closed curve.
    The z-value of the coordinates does not have to be specified,
    and will be ignored if it is.
    The resulting path through the points should rotate positively around the
    z axis to yield a positive surface.

    The return value is a dict with the following characteristics:
    
    - `L`   : circumference,
    - `A`   : enclosed surface,
    - `Sx`  : first area moment around global x-axis
    - `Sy`  : first area moment around global y-axis
    - `Ixx` : second area moment around global x-axis
    - `Iyy` : second area moment around global y-axis
    - `Ixy` : product moment of area around global x,y-axes
    """
    if F.nplex() != 2:
        raise ValueError, "Expected a plex-2 Formex!"
    #GD.debug("The circumference has %d segments" % F.nelems())
    x = F.x()
    y = F.y()
    x0 = x[:,0]
    y0 = y[:,0]
    x1 = x[:,1]
    y1 = y[:,1]
    a = (x0*y1 - x1*y0) / 2
    return {
        'L'   : sqrt((x1-x0)**2 + (y1-y0)**2).sum(),
        'A'   : a.sum(),
        'Sx'  : (a*(y0+y1)).sum()/3,
        'Sy'  : (a*(x0+x1)).sum()/3,
        'Ixx' : (a*(y0*y0+y0*y1+y1*y1)).sum()/6,
        'Iyy' : (a*(x0*x0+x0*x1+x1*x1)).sum()/6,
        'Ixy' :-(a*(x0*y0+x1*y1+(x0*y1+x1*y0)/2)).sum()/6,
        }


def extendedSectionChar(S):
    """Computes extended section characteristics for the given section.

    S is a dict with section basic section characteristics as returned by
    sectionChar().
    This function computes and returns a dict with the following:

    - `xG`, `yG` : coordinates of the center of gravity G of the plane section
    - `IGxx`, `IGyy`, `IGxy` : second area moments and product around axes
      through G and  parallel with the global x,y-axes
    - `alpha` : angle(in radians) between the global x,y axes and the principal
      axes (X,Y) of the section (X and Y always pass through G)
    - `IXX`, `IYY` : principal second area moments around X,Y respectively. (The
      second area product is always zero.)
    """
    xG =  S['Sy']/S['A']
    yG =  S['Sx']/S['A']
    IGxx = S['Ixx'] - S['A'] * yG**2
    IGyy = S['Iyy'] - S['A'] * xG**2
    IGxy = S['Ixy'] + S['A'] * xG*yG
    alpha,IXX,IYY = princTensor2D(IGxx,IGyy,IGxy)
    return {
        'xG'   : xG,
        'yG'   : yG,
        'IGxx' : IGxx,
        'IGyy' : IGyy,
        'IGxy' : IGxy,
        'alpha': alpha,
        'IXX'  : IXX,
        'IYY'  : IYY,
        }


def princTensor2D(Ixx,Iyy,Ixy):
    """Compute the principal values and directions of a 2D tensor.

    Returns a tuple with three values:
    
    - `alpha` : angle (in radians) from x-axis to principal X-axis
    - `IXX,IYY` : principal values of the tensor
    """
    from math import sqrt,atan2
    C = (Ixx+Iyy) * 0.5
    D = (Ixx-Iyy) * 0.5
    R = sqrt(D**2 + Ixy**2)
    IXX = C+R
    IYY = C-R
    alpha = atan2(Ixy,D) * 0.5
    return alpha,IXX,IYY
    

# End
