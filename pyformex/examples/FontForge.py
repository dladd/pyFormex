# $Id$    *** pyformex ***
##
##  This file is part of pyFormex 0.7.3 Release Mon Dec 29 15:32:01 2008
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
"""FontForge

This example demonstrates the use of FontForge library to render text. To be
able to run it, you need to have the FontForge library and its Python bindings
installed. On Debian GNU/Linux you can achieve this by installing the package
'python-fontforge'.


"""
from __future__ import print_function
_status = 'checked'
_level = 'advanced'
_topics = ['curve', 'font']
_techniques = ['bezier','borderfill']

from gui.draw import *
import odict
from plugins.curve import BezierSpline,PolyLine
from simple import connectCurves
from plugins.trisurface import fillBorder
from geomtools import closestPoint
import utils
import os,sys

try:
    import fontforge
except ImportError:
    warning("You do not have fontforge and its Python bindings.\nPlease install python-fontforge and then try again.")
    exit()


# Define some extra font files
extra_fonts = odict.ODict([
    ('blippo',"/mnt/work/local/share/fonts/blippok.ttf"),
    ('blimpo',"/home/bene/tmp/Blimpo-Regular.ttf"),
    ('verdana',"/var/lib/defoma/x-ttcidfont-conf.d/dirs/TrueType/Verdana.ttf"),
    ])


def glyphCurve(c):
    """Convert a glyph contour to a list of quad bezier curves."""
    points = []
    control = []
    P0 = c[0]
    points.append([P0.x,P0.y])
    for i in (arange(len(c))+1) % len(c):
        P = c[i]
        if P0.on_curve and P.on_curve:
            # straight segment
            control.append([0.5*(P0.x+P.x),0.5*(P0.y+P.y)])
            points.append([P.x,P.y])
            P0 = P
            continue
        elif P0.on_curve and not P.on_curve:
            # undecided
            P1 = P0
            P0 = P
            continue
        elif not P0.on_curve and P.on_curve:
            # a single quadratic segment
            control.append([P0.x,P0.y])
            points.append([P.x,P.y])
            P0 = P
            continue
        else: # not P0.on_curve and not P.on_curve:
            # two quadratic segments, central point to be interpolated
            PM = fontforge.point()
            PM.x = 0.5*(P0.x+P.x)
            PM.y = 0.5*(P0.y+P.y)
            PM.on_curve = True
            points.append([PM.x,PM.y])
            control.append([P0.x,P0.y])
            P1 = PM
            P0 = P
            continue
    
    return Coords(points),Coords(control)

    
            
def contourCurve(c):
    """Convert a fontforge contour to a pyFormex curve""" 
    points,control = glyphCurve(c)
    return BezierSpline(coords=points[:-1],control=control,degree=2,closed=True)

def charContours(fontfile,character):
    font = fontforge.open(fontfile,5)
    print("FONT INFO FOR %s" % font)
    print(dir(font))
    print(font.gpos_lookups)

    g = font[ord(character)]
    print("GLYPH INFO FOR %s" % g)
    print(dir(g))
    print(g.getPosSub)


    l = g.layers[1]
    print("Number of curves: %s" % len(l))
    ## c = l[0]
    ## print(c)
    ## #print(dir(c))
    ## print(c.closed)
    ## print(c.is_quadratic)
    ## print(c.isClockwise())
    ## print(len(c))
    ## #print(c.reverseDirection())

    ## #if c.isClockwise():
    ## #    c = c.reverseDirection()

    return l


def connect2curves(c0,c1):
    x0 = c0.coords
    x1 = c1.coords
    i,j,d = closestPoint(x0,x1)
    x = concatenate([roll(x0,-i,axis=0),roll(x1,-j,axis=0)])
    return BezierSpline(control=x,degree=2,closed=True)
    
    
            
def charCurves(fontfile,character):
    l = charContours(fontfile,character)
    c = [ contourCurve(li) for li in l ]
    fontname = utils.projectName(fontfile)
    export({'%s-%s'%(fontname,character):c})
    return c


def drawCurve(curve,color,fill=False,with_border=True,with_points=True):
    if fill:
        border = curve.approx(24)
        if with_border:
            draw(border,color=red)
        #drawNumbers(border.coords,color=red)
        surface = fillBorder(border,'planar')
        draw(surface,color=color)
        #drawNumbers(surface)
    else:
        draw(curve,color=color)
    if with_points:
        drawNumbers(curve.pointsOn())
        drawNumbers(curve.pointsOff(),color=red)


def drawCurve2(curve,color,fill=False,with_border=True,with_points=True):
    if fill:
        curve = connect2curves(*curve)
        drawCurve(curve,blue,fill)
    else:
        drawCurve(curve[0],color,fill=False,with_border=with_border,with_points=with_points)
        drawCurve(curve[1],color,fill=False,with_border=with_border,with_points=with_points)


def show(fontname,character,fill=False):

    curve = charCurves(fontname,character)
    size = curve[0].pointsOn().bbox().dsize()
    clear()

    if not fill:
        for c in curve:
            drawCurve(c,blue,fill=False)

    else:
        if len(curve) == 1:
            drawCurve(curve[0],blue,fill=True)
        elif len(curve) == 2:
            drawCurve2(curve,blue,fill=True)
    
    return


# Initialization

chdir (__file__)

print(dir(fontforge))

fonts = utils.listFontFiles() + [ f for f in extra_fonts if os.path.exists(f) ]
print("Number of available fonts: %s" % len(fonts))

fontname = None
character = 'S'
connect = False
fill = True


def run():
    res = askItems([
        _I('fontname',fontname1,choices=fonts),
        _I('character',character1,max=1),
        _I('fill',fill),
        ],enablers=[
#        ('connect',True,'fontname2','character2')
        ])

    if not res:
        return

    globals().update(res)
    
    show(**res)
    

    

if __name__ == 'draw':
    run()


# End
