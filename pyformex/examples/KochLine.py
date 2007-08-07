#!/usr/bin/env python pyformex.py
# $Id: KochLine.py 66 2006-02-20 20:08:47Z bverheg $
##
## This file is part of pyFormex 0.5 Release Mon Jul 30 13:38:48 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Koch line"""

from plugins.lima import lima

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
draw(Formex.concatenate([ 
    F[i].scale(sc[i]).translate([0,sz*(i-1),0]) for i in range(n)]))
draw(Formex.concatenate([ 
    F[i].scale(sc[i]).translate([0,sz*n,0]) for i in range(n)]))
# a variant which dispays the lines as radii of a six-pointed star
#clear()
#draw(Formex.concatenate([F[i].rotate(60*i).scale(math.pow(3,5-i)) for i in range(6)]).translate()
