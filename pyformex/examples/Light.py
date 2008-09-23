#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

"""Light

level = 'beginner'
topics = ['geometry']
techniques = ['dialog', 'colors', 'persistence']

"""

from gui.prefMenu import setRender

smooth()

Shape = { 'triangle':'16',
          'quad':'123',
          }
color2 = array([red,green,blue]) # 3 base colors
F = Formex(mpattern(Shape['triangle'])).replic2(8,4)
color3 = resize(color2,F.shape())
draw(F,color=color3)


setRender()

for a in [ 'ambient', 'specular', 'emission', 'shininess' ]:
    v = getattr(GD.canvas,a)
    print "  %s: %s" % (a,v)

# End
