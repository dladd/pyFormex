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

from gui.imagecolor import *

flat()
lights(False)
transparent(False)
view('front')

filename = getcfg('datadir')+'/butterfly.png'
image = None
scaled_image = None
w,h = 200,200

def selectFile():
    """Select an image file."""
    global filename
    filename = askFilename(filename,filter=utils.fileDescription('img'),multi=False,exist=True)
    if filename:
        currentDialog().updateData({'filename':filename})
        loadImage()


def loadImage():
    global image, scaled_image
    image = QtGui.QImage(filename)
    if image.isNull():
        warning("Could not load image '%s'" % filename)
        return None

    w,h = image.width(),image.height()
    print "size = %sx%s" % (w,h)

    diag = currentDialog()
    if diag:
        diag.updateData({'nx':w,'ny':h})

    maxsiz = 40000.
    if w*h > maxsiz:
        scale = sqrt(maxsiz/w/h)
        w = int(w*scale)
        h = int(h*scale)


transforms = {
    'flat': lambda F: F,
    'cylindrical': lambda F: F.cylindrical([2,0,1],[2.,90./float(nx),1.]).rollAxes(-1),
    'spherical': lambda F: F.spherical(scale=[1.,90./float(nx),2.]).rollAxes(-1),
    'projected_on_cylinder': lambda F: F.projectOnCylinder(2*R,1),
    }

res = askItems([
    ('filename',filename,{'buttons':[('Select File',selectFile)]}),
    ('nx',w,{'text':'width'}),
    ('ny',h,{'text':'height'}),
    ('transform',None,'vradio',{'choices':transforms.keys()}),
    ])

if not res:
    exit()

globals().update(res)

import timer
tim = timer.Timer()


tim.reset()
if image is None:
    loadImage()
print "Loading image: %.2f" % tim.seconds(False)

if image is None:
    exit()

# Create the colors
tim.reset()
color,colortable = image2glcolor(image.scaled(nx,ny))
print "Image to colors: %.2f" % tim.seconds(False)

tim.reset()
# Create a 2D grid of nx*ny elements
R = float(nx)/pi
L = float(ny)
F = Formex(mpattern('123')).replic2(nx,ny).centered()
F = F.translate(2,R)
trf = transforms[transform]
F = trf(F)
print "Creating grid : %.2f" % tim.seconds(False)

clear()
tim.reset()
draw(F,color=color,colormap=colortable)
print "drawing time = %.2f" % tim.seconds(False)
drawtext('Created with pyFormex',10,10)

# End
