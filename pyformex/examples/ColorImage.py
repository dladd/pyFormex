# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""ColorImage

This example illustrates how to display and transform 2D images.
First, a 2D image is read from a file. You can select a file by clicking ont
the filename button. 
Then a grid type geometry is constructed. The size of the grid can be set,
and the image will be rescaled to that size.
The individual elements of the grid are attributed colors corresponding with
the image pixel values.
Finally, before the result is drawn, the geometry can be transformed into some
other shape or be projected on a another surface.
"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['image']
_techniques = ['color','image']

from gui.draw import *
from gui.widgets import ImageView
from plugins.imagearray import *
from odict import ODict

resetAll()

def selectImage(fn):
    fn = askImageFile(fn)
    if fn:
        viewer.showImage(fn)
        loadImage(fn)
    return fn


def loadImage(fn):
    global image, scaled_image
    image = QImage(fn)
    if image.isNull():
        warning("Could not load image '%s'" % fn)
        return None

    w,h = image.width(),image.height()
    print("size = %sx%s" % (w,h))

    diag = currentDialog()
    if diag:
        diag.updateData({'nx':w,'ny':h})

    maxsiz = 40000.
    if w*h > maxsiz:
        scale = sqrt(maxsiz/w/h)
        w = int(w*scale)
        h = int(h*scale)
    return w,h


def run():
    global image, scaled_image
    flat()
    lights(False)
    transparent(False)
    view('front')

    # default image file
    filename = getcfg('datadir')+'/butterfly.png'
    image = None
    scaled_image = None
    w,h = 200,200

    # image viewer widget
    viewer = ImageView(filename)

    transforms = ODict([
        ('flat', lambda F: F),
        ('cylindrical', lambda F: F.cylindrical([2,0,1],[2.,90./float(nx),1.]).rollAxes(-1)),
        ('spherical', lambda F: F.spherical(scale=[1.,90./float(nx),2.]).rollAxes(-1)),
        ('projected_on_cylinder', lambda F: F.projectOnCylinder(2*R,1)),
        ])

    res = askItems([
        _I('filename',filename,text='Image file',itemtype='button',func=selectImage),
        _I('viewer',viewer,itemtype='widget'),  # the image previewing widget
        _I('nx',w,text='width'),
        _I('ny',h,text='height'),
        _I('transform',itemtype='vradio',choices=transforms.keys()),
        ])

    if not res:
        return

    globals().update(res)

    if image is None:
        print("Loading image")
        loadImage(filename)

    if image is None:
        return

    # Create the colors
    color,colortable = image2glcolor(image.scaled(nx,ny))
    print("Converting image to color array")

    # Create a 2D grid of nx*ny elements
    print("Creating grid")
    R = float(nx)/pi
    L = float(ny)
    F = Formex('4:0123').replic2(nx,ny).centered()
    F = F.translate(2,R)

    # Transform grid and draw
    def drawTransform(transform):
        print("Transforming grid")
        trf = transforms[transform]
        G = trf(F)
        clear()
        print("Drawing Colored grid")
        draw(G,color=color,colormap=colortable)
        drawText('Created with pyFormex',20,20,size=24)

    drawTransform(transform)
    zoomAll()

    
    ## layout(2)
    ## viewport(0)
    ## drawTransform('cylindrical')
    ## zoomAll()

    ## viewport(1)
    ## drawTransform('spherical')
    ## zoomAll()

if __name__ == 'draw':
    run()
# End
