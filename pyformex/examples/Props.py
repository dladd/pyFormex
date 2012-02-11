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
"""Props

This example demonstrates the propagation of property numbers through
the pyFormex geometrical transformations.
It also shows how multiple viewports can be used to show different objects
at the same time.

Property numbers can be assigned to most Geometry objects in pyFormex.
They define a single integer number for each basic element in the object.
The property numbers can be used for whatever the user sees fit, but are
commonly used to display different elements in another color.

When a pyFormex transformation generates new elements out of old ones, each
new element will get the same property number as its parent, thus making it
possible to track from what original element a resulting one originated, or to
pass data to child elements.

This process is illustrated in this example.

- First, a Formex is constructed consisting of two triangles,
  and given the property numbers 1, resp. 3, which are displayed by default
  in the colors red, resp. blue (F0).

- The structure is then replicated 3 times in x-direction and twice in
  y-direction (F1). Remark how the colors get inherited.

- Next, a reflection of F1 in the y-direction is added to F1 to yield F2.

- Finally F3 is obtained by rotating F2 over 180 degrees around the z-axis
  (and adding it to F2).
"""
_status = 'checked'
_level = 'beginner'
_topics = ['geometry']
_techniques = ['viewport', 'color', 'symmetry']

from gui.draw import *
    
def run():
    layout(4)
    F0 = Formex('3:012934',[1,3])
    F1 = F0.replic2(3,2)
    F2 = F1 + F1.reflect(1)
    F3 = F2 + F2.rotate(180.,1)
    
    for i,F in enumerate([F0,F1,F2,F3]):
        viewport(i)
        flat()
        clear()
        vp(i)
        draw(F)
        drawText("F%s"%i,10,20,size=18)
        pf.canvas.update()
    
    
if __name__ == 'draw':
    run()
# End
