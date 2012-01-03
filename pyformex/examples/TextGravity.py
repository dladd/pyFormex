# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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
"""TextGravity

Show the use of the text gravity parameter.

level = 'beginner'
topics = []
techniques = ['text']

"""

clear()
lights(False)
x = pf.canvas.width()//2
y = pf.canvas.height()//2
x,y = 600,300
print x,y

from gui.decors import Grid

d = 50

G = Grid(x-50,y-50,x+50,y+50,2,2)
decorate(G)

delay(2)
for g in [ 'NW','N','NE','W','C','E','SW','S','SE']:
    T = drawText("XXX  %s  XXX"%g,x,y,gravity=g)
    wait()
    undecorate(T)


delay(1)
from gui.gluttext import GLUTFONTS
for f in GLUTFONTS.keys():
    S = drawText(f,20,20,font='hv18')
    T = drawText('X',x,y,font=f,gravity='C')
    wait()
    undecorate(S)
    undecorate(T)
    
# End
