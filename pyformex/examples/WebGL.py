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
    T = sector(1.0,360.,6,36,h=1.0,diag='u').toSurface().scale(1.5).reverse()
    C = cylinder(1.2,1.5,24,4,diag='u').toSurface().trl([0.5,0.5,0.5]).reverse()

    # Draw the geometry with given colors/opacity
    # Settings colors and opacity in this way makes the model
    # directly ready to export as WebGL
    S.color = red
    S.alpha = 0.7
    S.caption = 'A sphere'
    S.control = ['visible','opacity','color']

    T.color = blue
    T.caption = 'A cone'
    T.alpha = 1.0
    T.control = ['visible','opacity','color']

    C.color = 'yellow'
    C.caption = 'A cylinder'
    C.alpha = 0.8
    C.control = ['visible','opacity','color']

    export({'sphere':S,'cone':T,'cylinder':C})

    draw([S,T,C])
    zoomAll()
    rotRight(30.)
    camera = pf.canvas.camera
    print("Camera focus: %s; position %s" % (camera.focus, camera.getPosition()))

    if checkWorkdir():
        # Export everything to webgl
        W = WebGL()
        W.addScene()
        W.export('Scene1','Two spheres and a cone',createdby=True)

if __name__ == 'draw':
    run()

# End
