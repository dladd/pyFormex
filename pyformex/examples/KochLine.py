#!/usr/bin/env pyformex --gui
# $Id: KochLine.py 66 2006-02-20 20:08:47Z bverheg $
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
"""Koch line

level = 'beginner'
topics = ['geometry']
techniques = ['colors']

"""

from plugins.lima import lima

wireframe()
linewidth(2)
n = 6 # number of generations

# We use the lima module to create six generations of the Koch line
F = [ Formex(lima("F",{"F":"F*F//F*F"},i,
                  { 'F' : 'fd();', '*' : 'ro(60);', '/' : 'ro(-60);' }),i)
      for i in range(n) ]
# and display them in series
clear()
# scale each Formex individually to obtain same length
sc = [ 3**(-i) for i in range(n) ]
sz = sc[0]/3.

F = [F[i].scale(sc[i]) for i in range(n)] 


mode = random.randint(3)
if mode == 0:
    # on top of each other
    draw([F[i].translate([0,sz*(i-1),0]) for i in range(n)])

elif mode == 1:
    # one above the other
    draw([F[i].translate([0,sz*n,0]) for i in range(n)])

else:
    # as radii of an n-pointed star
    draw([F[i].rotate(360.*i/n) for i in range(n)])

zoomAll()

# End
