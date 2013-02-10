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

"""WebGL

This example illustrates the use of the webgl plugin to create WebGL models
in pyFormex.

The example creates a sphere, a cylinder and a cone, draws them with color
and transparency, and exports an equivalent WebGL model in the current
working directory. Point your WebGL capable browser to the created
'scene1.html' file to view the WebGL model.
"""
from __future__ import print_function

_status = 'checked'
_level = 'normal'
_topics = ['export']
_techniques = ['webgl']

from gui.draw import *

from simple import sphere,sector,cylinder
from mydict import Dict
from plugins.webgl import WebGL

def run():
    reset()
    clear()
    smooth()
    transparent()
    view('right')

    # Create some geometry
    S = sphere()
    T = sector(1.0,360.,6,36,h=1.0,diag='u').toSurface().scale(1.5)
    C = cylinder(1.2,1.5,24,4,diag='u').toSurface().trl([0.5,0.5,0.5])

    # Draw the geometry with given colors/opacity
    draw(S,color=red,alpha=0.7)
    draw(T,color=blue,alpha=1.0)     #  1.0 means T is opaque !
    draw(C,color=yellow,alpha=0.7)
    zoomAll()
    rotRight(30.)

    # Write some of the geometry to STL file
    S.write('sphere.stl')
    T.write('cone.stl')

    if not checkWorkdir():
        return
    # Export everything to webgl
    # We can add a Geometry object or an STL file
    #

    W = WebGL()
    W.add(file='sphere.stl',caption='A sphere',color=red,alpha=0.7)
    W.add(file='cone.stl',name='cone',caption='A cone',color=blue)
    W.add(obj=C,name='cylinder',caption='A cylinder',color=yellow,alpha=0.7)
    # set the camera viewpoint
    W.view(position=[6.,0.,3.])
    W.export('scene1','Two spheres and a cone',createdby=True)

if __name__ == 'draw':
    run()

# End
