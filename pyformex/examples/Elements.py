#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

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
