#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
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
"""ColorImage

level = 'normal'
topics = ['image']
techniques = ['colors']

"""

from gui.imageColor import *

smooth()
lights(False)

filename = 'butterfly.ppm'  
filename = 'butterfly2.png'  

chdir(__file__)
im = QtGui.QImage(filename)
if im.isNull():
    warning("Could not load image '%s'" % filename)
    exit()

res = askItems([('Width',40),('Height',32)])
if not res:
    exit()

nx = res['Width']
ny = res['Height']

F = Formex(mpattern('123')).replic2(nx,ny)

clear()
R = max(nx,ny)
draw(F.translate(2,R/2).projectOnCylinder(R,1,center=[nx/2,ny/2,0.]),color=image2glcolor(im.scaled(nx,ny)))


drawtext('Created with pyFormex',10,10)
# End
