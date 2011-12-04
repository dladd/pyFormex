#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
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
"""Colored

level = 'beginner'
topics = ['surface']
techniques = ['color']

"""

from gui.actors import *

smooth()
lights(False)

Rendermode = [ 'smooth','flat' ]
Lights = [ False, True ]
Shapes = [ '3:016', '4:0123', ]

color0 = None  # no color: current fgcolor
color1 = red   # single color
color2 = array([red,green,blue]) # 3 colors: will be repeated

for shape in Shapes:
    F = Formex(shape).replic2(8,4)
    color3 = resize(color2,F.shape()) # full color
    for mode in Rendermode:
        renderMode(mode)
        for c in [ color0,color1,color2,color3]:
            clear()
            FA = GeomActor(F,color=c)
            drawActor(FA)
	    print c
            zoomAll()
            for light in Lights:
                lights(light)
                sleep(1)


# End
