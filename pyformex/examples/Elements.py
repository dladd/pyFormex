#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Elements

level = 'beginner'
topics = ['geometry','mesh']
techniques = ['dialog','elements']

"""

import elements
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
    ElemList = []
    for ndim in [2,3]:
        ElemList += elements.elementTypes(ndim)
        
    res = askItems([('Element Type',None,'select',['All',]+ElemList),],legacy=True)
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
