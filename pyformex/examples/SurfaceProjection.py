# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
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

"""SurfaceProjection.py

This example illustrates the use of Coords.projectOnSurface and
trisurface.intersectSurfaceWithLines as a method to render a 2D image
onto a 3D surface.

"""
from __future__ import print_function
_status = 'checked'
_level = 'normal'
_topics = ['surface']
_techniques = ['transform','projection','dialog','image','isopar']

from gui.draw import *
from plugins.trisurface import TriSurface
import elements
from gui.widgets import ImageView
from gui.imagearray import *


def selectImage(fn):
    global wviewer
    fn = askImageFile(fn)
    if fn:
        wviewer.showImage(fn)
    return fn


def loadImage(fn):
    global image, scaled_image
    image = QImage(fn)
    if image.isNull():
        warning("Could not load image '%s'" % fn)
        return None
    return image


def makeGrid(nx,ny,eltype):
    """Create a 2D grid of nx*ny elements of type eltype.

    The grid is scaled to unit size and centered.
    """
    elem = getattr(elements,eltype)
    return elem.toFormex().replic2(nx,ny).resized(1.).centered()


def drawImage(grid,base,patch):
    """Draw the image on the specified patch grid.

    The image colors are specified in the global variable pcolor.
    grid is a Formex with px*py Quad8 elements.
    Each element of grid will be filled by a kx*ky patch of colors.
    """
    mT = [ patch.isopar('quad8',x,base) for x in grid.coords ]
    return [ draw(i,color=c,alpha=0.99,bbox='last',nolight=True,wait=False) for i,c in zip (mT,pcolor)]


def intersectSurfaceWithSegments2(s1, segm, atol=1.e-5, max1xperline=True):
    """it takes a TriSurface ts and a set of segments (-1,2,3) and intersect the segments with the TriSurface.
    It returns the points of intersections and, for each point, the indices of the intersected segment and triangle. If max1xperline is True, only 1 intersection per line is returned (in order to remove multiple intersections due to the tolerance) together with the index of line and triangle (at the moment the selection of one intersection among the others is random: it does not take into account the distances). If some segments do not intersect the surface, their indices are also returned."""
    segm = segm.coords
    p, il, it=trisurface.intersectSurfaceWithLines(s1, segm[:, 0], normalize(segm[:, 1]-segm[:, 0]))
    win= length(p-segm[:, 0][il])+ length(p-segm[:, 1][il])< length(segm[:, 1][il]-segm[:, 0][il])+atol
    px, ilx, itx=p[win], il[win], it[win]
    if max1xperline:
        ip= inverseIndex(ilx.reshape(-1, 1))
        sp=sort(ip, axis=1)[:, -1]
        w= where(sp>-1)[0]
        sw=sp[w]
        return px[sw], w, itx[sw], delete( arange(len(segm)),  w)
    else:return px, ilx, itx


def run():
    global wviewer,pcolor,px,py
    clear()
    smooth()
    lights(True)
    transparent(False)
    view('iso')

    image = None
    scaled_image = None

    # read the teapot surface
    T = TriSurface.read(getcfg('datadir')+'/teapot.off')
    xmin,xmax = T.bbox()
    T= T.trl(-T.center()).scale(4./(xmax[0]-xmin[0])).setProp(2)
    draw(T)

    # default image file
    dfilename = getcfg('datadir')+'/benedict_6.jpg'
    wviewer = ImageView(dfilename,maxheight=200)

    res = askItems([
        _I('filename',dfilename,text='Image file',itemtype='button',func=selectImage),
        _I('viewer',wviewer,itemtype='widget'),  # the image previewing widget
        _I('px',4,text='Number of patches in x-direction'),
        _I('py',6,text='Number of patches in y-direction'),
        _I('kx',30,text='Width of a patch in pixels'), 
        _I('ky',30,text='Height of a patch in pixels'),
        _I('scale',1.0,text='Scale factor'),
        _I('trl',[-0.4,-0.1,2.],itemtype='point',text='Translation'),
        _I('method',choices=['projection','intersection']),
        ])

    if not res:
        return

    globals().update(res)

    nx,ny = px*kx,py*ky # pixels
    print('The image is reconstructed with %d x %d pixels'%(nx, ny))

    F = Formex('4:0123').replic2(nx,ny).centered()
    if image is None:
        print("Loading image")
        image = loadImage(filename)
        wpic,hpic = image.width(),image.height()
        print("Image size is %sx%s" % (wpic,hpic))

    if image is None:
        return

    # Create the colors
    color,colortable = image2glcolor(image.scaled(nx,ny))
    # Reorder by patch
    pcolor = color.reshape((py,ky,px,kx,3)).swapaxes(1,2).reshape(-1,kx*ky,3)
    print("Shape of the colors array: %s" % str(pcolor.shape))

    mH = makeGrid(px,py,'Quad8')

    try:
        ratioYX = float(hpic)/wpic
        mH = mH.scale(ratioYX,1) # Keep original aspect ratio
    except:
        pass

    mH0 = mH.scale(scale).translate(trl)

    dg0 = draw(mH0,mode='wireframe')
    zoomAll()
    zoom(0.5)
    print("Create %s x %s patches" % (px,py))

    # Create the transforms
    base = makeGrid(1,1,'Quad8').coords[0]
    patch = makeGrid(kx,ky,'Quad4').toMesh()
    d0 = drawImage(mH0,base,patch)

    if method == 'projection':
        pts = mH0.coords.projectOnSurface(T,[0.,0.,1.],'-f')
        dg1 = d1 = [] # to allow dummy undraw 


    else:
        mH1 = mH.rotate(-30.,0).scale(0.5).translate([0.,-.7,-2.])
        dg1 = draw(mH1,mode='wireframe')
        d1 = drawImage(mH1,base,patch)

        x = connect([mH0.points(), mH1.points()])
        dx = draw(x)
        print("Creating intersection with surface")
        pts, il, it, mil=intersectSurfaceWithSegments2(T, x, max1xperline=True)

        if len(x) != len(pts):
            print("Some of the lines do not intersect the surface:")
            print(" %d lines, %d intersections %d missing" % (len(x),len(pts),len(mil)))
            return

    dp = draw(pts, marksize=6, color='white')
    #print pts.shape
    mH2 = Formex(pts.reshape(-1,8,3))

    if method == 'projection':
        x = connect([mH0.points(),mH2.points()])
        dx = draw(x)

    print("Create projection mapping using the grid points")
    d2 = drawImage(mH2.trl([0.,0.,0.01]),base,patch)
    # small translation to make sure the image is above the surface, not cutting it

    print("Finally show the finished image")
    undraw(dp);
    undraw(dx);
    undraw(d0);
    undraw(d1);
    undraw(dg0);
    undraw(dg1);
    view('front')
    zoomAll()
    pause(1)
    transparent()

if __name__ == 'draw':
    run()
# End
