# $Id$ *** pyformex ***
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
"""Colored

"""
from __future__ import print_function
_status = 'checked'
_level = 'beginner'
_topics = ['surface']
_techniques = ['color']

from gui.draw import *
from gui.actors import *


def run():
    smooth()
    lights(False)

    Rendermode = [ 'wireframe','flat','smooth' ]
    Lights = [ False, True ]
    Shapes = [ '3:012', '4:0123', ]

    color0 = None  # no color: current fgcolor
    color1 = 'red'   # single color
    color2 = ['red','green','blue'] # 3 colors: will be repeated

    delay(0)
    i=0
    for shape in Shapes:
        F = Formex(shape).replic2(4,2)
        color3 = resize(color2,F.shape()) # full color
        #print F.shape(),color3
        #print [ GLcolor(c) for c in color3]
        #continue
        for c in [ color0,color1,color2,color3]:
            for mode in Rendermode:
                clear()
                renderMode(mode)
                FA = GeomActor(F,color=c)
                drawActor(FA)
                zoomAll()
                for light in Lights:
                    lights(light)
                    print("%s: color %s, mode %s, lights %s" % (i,str(c),mode,light))
                    i += 1
                    pause(1)


if __name__ == 'draw':
    run()
# End
