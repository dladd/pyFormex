#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
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
"""Props

level = 'beginner'
topics = ['geometry']
techniques = ['viewports', 'colors', 'symmetry']

A demonstration of propagating property numbers.
Also shows the use of multiple viewports.
"""

def vp(i):
    viewport(i)
    smooth()
    lights(False)
    clear()
    
if __name__ == "draw":

    layout(4)
    F0 = Formex(mpattern('12-34'),[1,3])
    F1 = F0.replic2(2,2)
    F2 = F1 + F1.mirror(1)
    F3 = F2 + F2.rotate(180.,1)
    
    for i,F in enumerate([F0,F1,F2,F3]):
        vp(i)
        draw(F)
        drawtext("F%s"%i,10,10,'hv18')
    
    
# End
