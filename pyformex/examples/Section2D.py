#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
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
#
"""Section2D

Computing geometrical properties of plane sections.

level = 'normal'
topics = ['geometry','section2d']
techniques = []

"""

from plugins.section2d import *
from mesh import Mesh
import simple,connectivity,mydict


def showaxes(C,angle,size,color):
    H = Formex(simple.Pattern['plus']).scale(0.6*size).rot(angle/Deg).trl(C)
    draw(H,color=color)


def square_example(scale=[1.,1.,1.]):
    P = Formex([[[1,1]]]).rosette(4,90).scale(scale)
    return sectionize.connectPoints(P,close=True)

def rectangle_example():
    return square_example(scale=[2.,1.,1.])

def circle_example():
    return simple.circle(5.,5.)

def close_loop_example():
    # one more example, originally not a closed loop curve
    F = Formex('l:11').replic(2,1,1) + Formex('l:2').replic(2,2,0)
    M = F.toMesh()
    draw(M,color='green')
    drawNumbers(M,color=red)
    drawNumbers(M.coords,color=blue)

    print "Original elements:",M.elems
    conn = connectivity.connectedLineElems(M.elems)
    if len(conn) > 1:
        message("This curve is not a closed circumference")
        return None
    
    sorted = conn[0]
    print "Sorted elements:",sorted

    showInfo('Click to continue')
    clear()
    M = Mesh(M.coords,sorted)
    drawNumbers(M)
    return M.toFormex()


clear()
flat()
reset()
examples = { 'Square'    : square_example,
             'Rectangle' : rectangle_example,
             'Circle'    : circle_example,
             'CloseLoop' : close_loop_example,
             }

from gui.widgets import simpleInputItem as I
res = askItems([
    I('example',text='Select an example',choices=examples.keys()),
    ])
if res:
    F = examples[res['example']]()
    if F is None:
        exit()
    draw(F)
    S = sectionChar(F)
    S.update(extendedSectionChar(S))
    print mydict.CDict(S)
    G = Formex([[[S['xG'],S['yG']]]])
    draw(G,bbox='last')
    showaxes([S['xG'],S['yG'],0.],S['alpha'],F.dsize(),'red')

# End
