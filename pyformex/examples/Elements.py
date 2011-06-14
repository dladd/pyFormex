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

def showElement(eltype,options):
    clear()
    drawText("Element type: %s" %eltype,100,200,font='times',size=18,color=black)
    el = elementType(eltype)
    #print el.report()
    M = el.toMesh()
    
    ndim = el.ndim

    if ndim == 3:
        view('iso')
        smooth()
    else:
        view('front')
        if options['Force dimensionality']:
            flatwire()
        else:
            smoothwire()
        
    #print options['Deformed']
    if options['Deformed']:
        M.coords = M.coords.addNoise(rsize=0.1)
        if options['Force dimensionality']:
            if ndim < 3:
                M.coords[...,2] = 0.0
            if ndim < 2:
                M.coords[...,1] = 0.0

    i = 'xyz'.find(options['Mirrored'])
    if i>=0:
        M = M.trl(i,0.2)
        M = M + M.reflect(i)

    M.setProp([5,6])
    
    if options['Draw as'] == 'Formex':
        M = M.toFormex()
    elif options['Draw as'] == 'Border':
        M = M.getBorderMesh()
        
    draw(M.coords,wait=False)
    drawNumbers(M.coords)


    if options['Color setting'] == 'prop':
        draw(M)
    else:
        draw(M,color=red,bkcolor=blue)
        
         
        
if __name__ == "draw":

    ElemList = []
    for ndim in [0,1,2,3]:
        ElemList += elementTypes(ndim)

    res = {
        'Deformed':True,
        'Mirrored':'No',
        'Draw as':'Mesh',
        'Color setting':'direct',
        'Force dimensionality':False,
        }
    res.update(pf.PF.get('Elements_data',{}))
    #print res
        
    res = askItems(
        store=res,
        items=[
            I('Element Type',choices=['All',]+ElemList),
            I('Deformed',itemtype='bool'),
            I('Mirrored',itemtype='radio',choices=['No','x','y','z']),
            I('Draw as',itemtype='radio',choices=['Mesh','Formex','Border']),
            I('Color setting',itemtype='radio',choices=['direct','prop']),
            I('Force dimensionality',itemtype='bool'),
            ])
    if not res:
        exit()

    #print "RESULT",res
    pf.PF['Elements_data'] = res
    
    eltype = res['Element Type']
    if eltype == 'All':
        ellist = ElemList
    else:
        ellist = [eltype]
    clear()
    delay(1)
    for el in ellist:
        showElement(el,res)

        
    
    
# End
