#!/usr/bin/env pyformex --gui
# $Id$

from plugins.elements import Hex8
from formex import *

def cube_tri(color=None):
    """Create a cube with triangles."""
    back = Formex([[[0,0,0],[1,0,0],[1,1,0]],[[1,1,0],[0,1,0],[0,0,0]]]) # rev
    left = back.rotate(-90,1) 
    bot = back.rotate(90,0)
    front = back.translate(2,1)
    right = left.translate(0,1).reverse()
    top = bot.translate(1,1).reverse()
    back = back.reverse()
    cube = front+top+right+back+bot+left
    if color == 'Single':
        color = 'blue'
    elif color == 'Face':
        color = arange(1,7).repeat(2)
    elif color == 'Full':
        color = array([[4,5,7],[7,6,4],[7,3,2],[2,6,7],[7,5,1],[1,3,7],
                       [3,1,0],[0,2,3],[0,1,5],[5,4,0],[0,4,6],[6,2,0]])
    return cube,color


def cube_quad(color=None):
    """Create a cube with quadrilaterals."""
    vertices = Formex(Hex8.vertices)
    v=array(Hex8.vertices)
    e=array(Hex8.edges)
    f=array(Hex8.faces)
    c=array(Hex8.element)
    edges = Formex(v[e])
    faces = Formex(v[f])
    cube = Formex(v[c])
    if color == 'Single':
        color = 'red'
    elif color == 'Face':
        color = [4,1,5,2,6,3]
    elif color == 'Full':
        color = arange(8)[f]
    print color
    return faces,color

def showCube(base,color):
    print base,color
    if base == 'Triangle':
        cube = cube_tri
    else:
        cube = cube_quad
    cube,color = cube(color)
    clear()
    draw(cube,color=color)
    view('iso')
    zoomAll()
    zoom(1.5)
    
        
if __name__ == "draw":

    clear()
    reset()

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
            exit()

        all = res['All']
        if all:
            bases = baseshape
            colors = colormode
        else:
            bases = [ res['Base'] ]
            colors = [ res['Color'] ]

        print bases,colors
        for base in bases:
            if base == 'Quad':
                smooth()
            else:
                smoothwire()
            lights(False)

            for color in colors:
                showCube(base,color)
                if all:
                    sleep(2)

    exit()
    
##     draw(cube,color='yellow')

##     cutcube = cube.cutAtPlane([0.0,0.0,0.5],[0.0,0.0,1.0])

##     draw(cutcube,color='blue')
    
##     clear()
##     draw(cutcube,color='blue')
    
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
    
