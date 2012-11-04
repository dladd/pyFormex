# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
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

"""Position

Position an object A thus that its three points X are aligned with the
three points X of object B.
"""
from __future__ import print_function
_status = 'checked'
_level = 'beginner'
_topics = ['geometry']
_techniques = ['position']

from gui.draw import *

def drawObjectWithName(obj,name):
    """Draw an object and show its name at the center"""
    drawText3D(obj.center(),name)
    draw(obj)

def drawPointsNumbered(pts,color,prefix):
    """Draw a set of points with their number"""
    draw(pts,color=color,ontop=True,nolight=True)
    drawNumbers(Coords(pts),leader=prefix)


def run():
    clear()
    smoothwire()

    # The object to reposition
    A = Formex('4:0123',1).replic2(6,3)
    # The object to define the position
    B = Formex('3:016',2).replic2(4,4,taper=-1).trl(0,7.)

    drawObjectWithName(A,'Object A')
    drawObjectWithName(B,'Object B')

    #define matching points

    X = A[0,[0,3,1]]
    drawPointsNumbered(X,red,'X')

    Y = B[3,[1,2,0]]
    Y[2] = Y[0].trl([0.,1.,1.])
    drawPointsNumbered(Y,green,'Y')
    zoomAll()

    pause()

    # Reposition A so that X are aligned with Y
    C = A.position(X,Y)
    draw(C,color=blue)
    zoomAll()

if __name__ == 'draw':
    run()
# End
