# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 15:24:17 CET 2012)
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
"""Koch line

This example illustrates the use of the 'lima' plugin to create subsequent
generations of a Koch line. The Koch line is a line with fractal properties.
Six generations of the Koch line are created. They are drawn in one of three
ways:

- all on top of each other
- in a series one above the other
- as radii of an n-pointed star  

The actual draw method is choosen randomly. Execute again to see another one.
"""
from __future__ import print_function
_status = 'checked'
_level = 'beginner'
_topics = ['illustration']
_techniques = ['color','lime']

from gui.draw import *
from plugins.lima import lima

def run():
    clear()
    wireframe()
    view('front')
    linewidth(2)
    n = 6 # number of generations

    # We use the lima module to create six generations of the Koch line
    F = [ Formex(lima("F",{"F":"F*F//F*F"},i,
                      { 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' }),i)
          for i in range(n) ]

    # scale each Formex individually to obtain same length
    sc = [ 3**(-i) for i in range(n) ]
    sz = sc[0]/3.
    F = [F[i].scale(sc[i]) for i in range(n)] 

    # display all lines in one (randomly choosen) of three ways 
    mode = random.randint(3)
    if mode == 0:
        # all on top of each other
        draw([F[i].translate([0,sz*(i-1),0]) for i in range(n)])

    elif mode == 1:
        # one above the other
        draw([F[i].translate([0,sz*n,0]) for i in range(n)])

    else:
        # as radii of an n-pointed star
        draw([F[i].rotate(360.*i/n) for i in range(n)])

    zoomAll()

if __name__ == 'draw':
    run()
# End
