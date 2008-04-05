#!/usr/bin/env pyformex --gui
# $Id$

from plugins import elements
from formex import *

def showCube(base,color,adjust):
    print base,color
    if base == 'Triangle':
        cube = cube_tri
    else:
        cube = cube_quad
    cube,color = cube(color)
    clear()
    draw(cube,color=color,coloradjust=adjust)
    view('iso')
    zoomAll()
    zoom(1.5)

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
    if color == 'None':
        color = 'white'
    elif color == 'Single':
        color = 'blue'
    elif color == 'Face':
        color = arange(1,7).repeat(2)
    elif color == 'Full':
        color = array([[4,5,7],[7,6,4],[7,3,2],[2,6,7],[7,5,1],[1,3,7],
                       [3,1,0],[0,2,3],[0,1,5],[5,4,0],[0,4,6],[6,2,0]])
    return cube,color


def element(eltype):
    """Create a Formex with given eltype."""
    if not hasattr(elements,eltype):
        return None
    el = getattr(elements,eltype)()
    v=array(el.vertices)
    e=array(el.edges)
    if eltype == 'Wedge6':
        f = [ array([fi for fi in el.faces if len(fi) == 3]),
              array([fi for fi in el.faces if len(fi) == 4])]
    else:
        f=array(el.faces)
    s=array(el.element)
    return v,e,f,s
    
        
if __name__ == "draw":

    smooth()
    view('iso')
    res = askItems([('Element Type',None,'select',['Tet4','Wedge6','Hex8']),])
    if not res:
        exit()
        
    eltype = res['Element Type']
    v,e,f,s = element(eltype)
    if eltype == 'Wedge6':
        F = [ Formex(v), Formex(v[e]), [Formex(v[fi]) for fi in f], Formex(v[s]) ]
    else:
        F = [ Formex(v), Formex(v[e]), Formex(v[f]), Formex(v[s]) ]
    for Fi in F:
        #print Fi.shape()
        draw(Fi)
        sleep(2)

    
    
# End
