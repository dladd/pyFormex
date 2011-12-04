#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Edit Global Variables

level = 'advanced'
topics = ['editing']
techniques = ['persistence','interactive']
"""
clear()
alfabet = {
    'A': 'l:22144/61',
    'C': 'l:/13221',
   }

 
def plotChar(char):
    try:
        draw(Formex(alfabet[char]))
    except:
        raise ValueError,"Can not plot character %s" % char



data = dict(
    a = 1,
    b = 1.5,
    c = [0.,0.,0.],
    d = 'red',
    f = Formex('l:121212')
    )
export(data)
globals().update(data)
#draw(f)

plotChar('A')
plotChar('C')


right = array([1.,0.,0.])
print right
left = -right
print left
from plugins.curve import *
clear()
x = Formex('3:065').coords.reshape(-1,3)
C =  BezierSpline(x)
print C.pointsOn()
draw(C.pointsOn())
draw(C,color=red)
C =  BezierSpline(x.trl(0,1.),deriv=[left,right])
print C.pointsOn()
draw(C.pointsOn())
draw(C,color=blue)
exit()
x = Coords([[0,0],[0,2],[0,2],[1,1.5],[0,1],[0,1],[1,0.5],[0,0]])
C = BezierSpline(x)
draw(C.pointsOn())
draw(C,color=red)


# End
