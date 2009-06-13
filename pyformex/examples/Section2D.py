#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
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
#
"""Section2D

Computing geometrical properties of plane sections.

level = 'normal'
topics = ['geometry']
techniques = []

"""

from plugins.section2d import *
import simple,connectivity,mydict


def showaxes(C,angle,size,color):
    H = Formex(simple.Pattern['plus']).scale(0.6*size).rot(angle/rad).trl(C)
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
    F = Formex(pattern('11')).replic(2,1,1) + Formex(pattern('2')).replic(2,2,0)
    nodes,elems = F.feModel()

    FN = Formex(nodes)
    drawNumbers(FN,color=blue)

    F = Formex(nodes[elems])
    draw(F,color='green')
    drawNumbers(F,color=red)

    print "Original elements:",elems
    ret, sorted = connectivity.closedLoop(elems)
    print "Sorted elements:",sorted

    showInfo('Click to continue')
    clear()
    F = Formex(nodes[sorted])
    drawNumbers(F)
    return F


clear()
reset()
examples = { 'Square'    : square_example,
             'Rectangle' : rectangle_example,
             'Circle'    : circle_example,
             'CloseLoop' : close_loop_example,
             }

res = askItems([('Select an example',None,'select',examples.keys())])
if res:
    F = examples[res['Select an example']]()
    draw(F)
    S = sectionChar(F)
    S.update(extendedSectionChar(S))
    print mydict.CDict(S)
    G = Formex([[[S['xG'],S['yG']]]])
    draw(G,bbox='last')
    showaxes([S['xG'],S['yG'],0.],S['alpha'],F.dsize(),'red')

# End
