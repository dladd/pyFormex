#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
"""World

level = 'normal'
topics = ['image']
techniques = ['colors']

"""

from gui.imagecolor import *

clear()
smooth()
lights(False)
view('front')

fn = os.path.join(getcfg('datadir'),'Equirectangular-projection-400.jpg')
fn = askFilename(cur=fn,filter=utils.fileDescription('img'),)
if not fn:
    exit()

im = QtGui.QImage(fn)
if im.isNull():
    warning("Could not load image '%s'" % fn)
    exit()

nx,ny = im.width(),im.height()
#nx,ny = 200,200

# Create the colors
color,colormap = image2glcolor(im.scaled(nx,ny))


part = ask("How shall I show the image?",["Plane","Half Sphere","Full Sphere"])

# Create a 2D grid of nx*ny elements
F = Formex(mpattern('123')).replic2(nx,ny).centered().translate(2,1.)

#color = [ 'yellow' ]*(nx-2) + ['orange','red','orange']

if part == "Plane":
    G = F
else:
    if part == "Half Sphere":
        sx = 180.
    else:
        sx = 360.
    G = F.spherical(scale=[sx/nx,180./ny,2.*max(nx,ny)]).rollAxes(-1)
draw(G,color=color,colormap=colormap)
drawtext('Created with pyFormex',10,10)
# End
