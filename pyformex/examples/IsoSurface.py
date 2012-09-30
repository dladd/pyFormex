# $Id$ *** pyformex ***
##
##  This file is part of pyFormex
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

"""IsoSurface

This example illustrates the use of the isosurface plugin to construct
isosurface through a set of data
"""
_status = 'checked'
_level = 'beginner'
_topics = ['surface']
_techniques = ['isosurface']

from gui.draw import *
from plugins import isosurface as sf
import elements

def run():

    clear()
    smooth()

    # data space: create a grid to visualize
    nx,ny,nz = 10,8,6
    F = elements.Hex8.toFormex().rep([nx,ny,nz],[0,1,2],[1.0]*3).setProp(1)
    draw(F,mode='wireframe')

    # function to generate data: the distance from the origin
    dist = lambda x,y,z: sqrt(x*x+y*y+z*z)
    data = fromfunction(dist,(nx+1,ny+1,nz+1))

    # level at which the isosurface is computed
    isolevel = 9
    pf.GUI.setBusy()
    tri = sf.isosurface(data,isolevel)
    pf.GUI.setBusy(False)

    if len(tri) > 0:
        S = TriSurface(tri)
        draw(S)
        export({'isosurf':S})

    else:
        print "No surface found"


# The following is to make it work as a script
if __name__ == 'draw':
    run()


# End
