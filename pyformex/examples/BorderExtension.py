# $$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
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

"""BorderExtension

This example shows how to create extension tubes on the borders of a
surface techniques.
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['surface']
_techniques = ['extrude','borderfill','cut']

from gui.draw import *
import geomtools

def run():
    reset()
    clear()
    # read and draw the surface
    chdir(getcfg('datadir'))
    S = TriSurface.read('bifurcation.off')
    draw(S)
    # Get the center point and the border curves
    CS = S.center()
    border = S.border()

    BL = []
    for B in border:
        draw(B,color=red,linewidth=2)
        # find the smallest direction of the curve
        d,s = geomtools.smallestDirection(B.coords,return_size=True)
        # Find outbound direction of extension
        CB = B.center()
        p = sign(projection((CB-CS),d))
        # Flatten the border curve and translate it outwards
        B1 = B.projectOnPlane(d,CB).trl(d,s*4*p)
        draw(B1,color=green)
        # create a surface between border curve and translted flat curve
        M = B.connect(B1)
        draw(M,color=blue,bkcolor=yellow)
        BL.append(M)

    zoomAll()

    if ack("Convert extensions to 'tri3' and add to surface?"):
        # convert extensions to 'tri3'
        BL = [ B.setProp(i+1).convert('tri3') for i,B in enumerate(BL) ]
        T = TriSurface.concatenate([S]+BL).fixNormals()
        clear()
        draw(T)
        export({'T':T})


if __name__ == 'draw':
    run()
    
# End
