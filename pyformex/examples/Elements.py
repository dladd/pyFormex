#!/usr/bin/env pyformex --gui
# -*- coding: utf-8 -*-
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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

level = 'normal'
topics = ['geometry','mesh']
techniques = ['dialog','elements']

"""

from elements import *
from mesh import Mesh
from gui.widgets import simpleInputItem as I
import utils
import olist

colors = [black,blue,yellow,red]

def showElement(eltype,deformed,reduced,drawas):
    clear()
    flat()
    drawText("Element type: %s" %eltype,100,200,font='times',size=18,color=black)
    el = elementType(eltype)
    #print el.report()
    M = el.toMesh()
    
    ndim = 3
    if reduced:
        ndim = el.ndim

    if ndim == 3:
        view('iso')
    else:
        view('front')

    if M.eltype.ndim == 2:
        drawNumbers(M.coords)
        print M.elems.report()
        draw(M,color=red,bkcolor=blue)

        #draw(M.reverse(),color=red,bkcolor=blue)
    return
        
    v = M.coords
    if deformed:
        dv = ( random.rand(v.size).reshape(v.shape) - 0.5 ) * 0.1
        v += dv
        if ndim < 3:
            v[...,2] = 0.0
        if ndim < 2:
            v[...,1] = 0.0

    for i in range(el.ndim+1):
        e = M.getLowerEntities(i)
        F = Mesh(v,e)
     
        if drawas == 'Formex':
            F = F.toFormex()
            
        draw(F,color=colors[i])
        if i == 0:
            drawVertexNumbers(F)


    clear()
    F = M.getBorderMesh()
    draw(F)
         
        
if __name__ == "draw":

    ElemList = []
    for ndim in [0,1,2,3]:
        ElemList += elementTypes(ndim)
        
    res = askItems([
        I('Element Type',choices=['All',]+ElemList),
        I('Deformed',False),
        I('Reduced dimensionality',True),
        I('Draw as',None,itemtype='radio',choices=['Mesh','Formex',]),
        ])
    if not res:
        exit()
        
    eltype = res['Element Type']
    deformed = res['Deformed']
    reduced = res['Reduced dimensionality']
    drawas = res['Draw as']
    if eltype == 'All':
        ellist = ElemList
    else:
        ellist = [eltype]
    clear()
    delay(1)
    for el in ellist:
        showElement(el,deformed,reduced,drawas)
    
    
# End
