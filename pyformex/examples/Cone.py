#!/usr/bin/env python pyformex.py
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

"""Cone

level = 'beginner'
topics = ['geometry','surface']
techniques = ['dialog', 'colors']

"""

import simple

reset()
smooth()
r=3.
h=15.
n=64

F = simple.sector(r,360.,n,n,h=h,diag=None)
#F.coords = F.coords.flare(h/4,r/2,dir=[2,0],end=1,exp=5.)
F.setProp(0)
draw(F,view='bottom')
setDrawOptions({'bbox':None})
zoomall()
zoom(3)

print map(str,range(4))
ans = ask('How many balls do you want?',['0','1','2','3'])

try:
    nb = int(ans)
except:
    nb = 3
    
if nb > 0:
    B = simple.sphere3(n,n,r=0.9*r,bot=-90,top=90)
    B1 = B.translate([0.,0.,0.95*h])
    B1.setProp(1)
    draw(B1)

if nb > 1:
    B2 = B.translate([0.2*r,0.,1.15*h])
    B2.setProp(2)
    draw(B2)

if nb > 2:
    B3 = B.translate([-0.2*r,0.1*r,1.25*h])
    B3.setProp(6)
    draw(B3)

zoomall()
zoom(3)
