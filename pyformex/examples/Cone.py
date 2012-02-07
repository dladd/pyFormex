# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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

"""Cone

level = 'beginner'
topics = ['geometry','surface']
techniques = ['dialog', 'color']

"""
from gui.draw import *
import simple

def run():
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
    zoomAll()
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

    #sleep(10)

    if nb > 1:
        B2 = B.translate([0.2*r,0.,1.15*h])
        B2.setProp(2)
        draw(B2)

    if nb > 2:
        B3 = B.translate([-0.2*r,0.1*r,1.25*h])
        B3.setProp(6)
        draw(B3)

    zoomAll()
    zoom(3)

    focus(B1)

if __name__ == 'draw':
    run()
# End
