#!/usr/bin/env pyformex --gui
# $Id$

from plugins import elements
from formex import *
import utils

def showElement(eltype):
    if not hasattr(elements,eltype):
        exit()
        
    el = getattr(elements,eltype)()
    v = array(el.vertices)
    e = array(el.edges)
    s = array([el.element])
    
    F = [ Formex(v), Formex(v[e]), Formex(v[s],eltype=eltype) ]
    smooth()
    for Fi in F:
        clear()
        draw(Fi)
        sleep(1)
    wireframe()
    drawVertexNumbers(Fi)
    sleep(1)
        
if __name__ == "draw":

    view('iso')
    ElemList = ['Tet4','Wedge6','Hex8','Icosa']
    res = askItems([('Element Type',None,'select',['All',]+ElemList),])
    if not res:
        exit()
        
    eltype = res['Element Type']
    if eltype == 'All':
        ellist = ElemList
    else:
        ellist = [eltype]
    for el in ellist:
        showElement(el)
    
    
# End
