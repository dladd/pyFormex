#!/usr/bin/env pyformex --gui
# $Id: KochLine.py 66 2006-02-20 20:08:47Z bverheg $
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Koch line"""

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
