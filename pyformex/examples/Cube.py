#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 09:32:38 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Cube

level = 'normal'
topics = ['geometry','surface']
techniques = ['colors']

"""

from elements import Hex8
from formex import *

def cube_tri(color=None):
    """Create a cube with triangles."""
    back = Formex(mpattern('12-34'))
    left = back.rotate(-90,1) 
    bot = back.rotate(90,0)
    front = back.translate(2,1)
    right = left.translate(0,1).reverse()
    top = bot.translate(1,1).reverse()
    back = back.reverse()
    faces = front+top+right+back+bot+left
    if color == 'None':
        color = 'white'
    elif color == 'Single':
        color = 'blue'
    elif color == 'Face':
        color = arange(1,7).repeat(2)
    elif color == 'Full':
        color = array([[4,5,7],[7,6,4],[7,3,2],[2,6,7],[7,5,1],[1,3,7],
                       [3,1,0],[0,2,3],[0,1,5],[5,4,0],[0,4,6],[6,2,0]])
    return faces,color


def cube_quad(color=None):
    """Create a cube with quadrilaterals."""
    v = array(Hex8.vertices)
    f = array(Hex8.faces)
    faces = Formex(v[f])
    if color == 'Single':
        color = 'red'
    elif color == 'Face':
        color = [4,1,5,2,6,3]
    elif color == 'Full':
        color = array([7,6,4,5,3,2,0,1])[f]
    print color
    return faces,color


def showCube(base,color):
    #print base,color
    if base == 'Triangle':
        cube = cube_tri
    else:
        cube = cube_quad
    cube,color = cube(color)
    clear()
    draw(cube,color=color)
    export({'cube':cube})
#    zoomAll()
    
        
if __name__ == "draw":

    from gui import widgets

    clear()
    reset()
    smooth()
    view('iso')

    baseshape = ['Quad','Triangle']
    colormode = ['None','Single','Face','Full']

    all = False
    base = 'Quad'
    color = 'Full'
    while True:
        res = askItems([('All',all),
                        ('Base',base,'select',baseshape),
                        ('Color',color,'select',colormode),
                        ],caption="Make a selection or check 'All'")
        if not res:
            break;

        all = res['All']
        if all:
            bases = baseshape
            colors = colormode
        else:
            bases = [ res['Base'] ]
            colors = [ res['Color'] ]

        print bases,colors
        for base in bases:
            lights(False)

            for color in colors:
                showCube(base,color)
                if all:
                    sleep(1)

        # Break from endless loop if an input timeout is active !
        if widgets.input_timeout >= 0:
            break

    exit()
    

## The following was used to create the rendering icons
    
    draw(cube)
    view('iso')
    zoomAll()
    GD.canvas.zoom(1.5)

    export({'cube':cube})
    exit()

    import os
    os.chdir(os.path.dirname(GD.cfg['curfile']))
    from gui import draw as _draw

    for mode in [ 'wireframe', 'smooth', 'smoothwire', 'flat', 'flatwire' ]:
        getattr(_draw,mode)()
        image.saveIcon(mode)
        #sleep(1)


    clear()
    smoothwire()
    draw(cube.shrink(0.8),bbox=None)
    #GD.canvas.zoom(1.5)
    image.saveIcon('shrink')
    
# End
