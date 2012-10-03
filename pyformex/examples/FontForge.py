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


def charContour(fontfile,character):
    font = fontforge.open(fontfile,5)
    print("FONT INFO FOR %s" % font)
    print(dir(font))
    print(font.gpos_lookups)

    g = font[ord(character)]
    print("GLYPH INFO FOR %s" % g)
    print(dir(g))
    print(g.getPosSub)


    l = g.layers[1]
    print(len(l))
    c = l[0]
    print(c)
    print(dir(c))
    print(c.closed)
    print(c.is_quadratic)
    print(c.isClockwise())
    print(len(c))
    print(c.reverseDirection())

    if c.isClockwise():
        c = c.reverseDirection()

    return c
    
            
def charCurve(fontfile,character):
    c = charContour(fontfile,character)
    points,control = glyphCurve(c)
    curve =  BezierSpline(coords=points[:-1],control=control,degree=2,closed=True)
    fontname = utils.projectName(fontfile)
    export({'%s-%s'%(fontname,character):curve})
    return curve


def drawCurve(curve,color,fill=False,with_border=True,with_points=True):
    if fill:
        border = curve.approx(24)
        if with_border:
            draw(border,color=red)
        #drawNumbers(border.coords,color=red)
        surface = fillBorder(border,'border')
        draw(surface,color=color)
        #drawNumbers(surface)
    else:
        draw(curve,color=color)
    if with_points:
        drawNumbers(curve.pointsOn())
        drawNumbers(curve.pointsOff(),color=red)


def show(fontname1,character1,fontname2=None,character2=None,connect=False,fill=False):

    curve1 = charCurve(fontname1,character1)
    size = curve1.pointsOn().bbox().dsize()
    clear()

    drawCurve(curve1,blue,fill)
    
    return

    if connect:
        curve2 = charCurve(fontname2,character2)
        curve2.coords = curve2.coords.trl([0.,0.,size])
        drawCurve(curve2,red,fill)
    return

    print(curve1.nparts)
    print(curve2.nparts)

    F0 = curve1.toFormex()
    F1 = curve2.toFormex()

    F = connectCurves(F0,F1,4)
    draw(F,color=black)


# Initialization

chdir (__file__)

print(dir(fontforge))

fonts = utils.listFontFiles() + [ f for f in extra_fonts if os.path.exists(f) ]
print("Number of available fonts: %s" % len(fonts))

fontname1 = None
fontname2 = None
character1 = 'S'
character2 = 'p'
connect = False
fill = True


def run():
    res = askItems([
        _I('fontname1',fontname1,choices=fonts),
        _I('character1',character1,max=1),
        _I('fill',fill),
# TODO: CONNECT NOT YET WORKING
#        _I('connect',connect),
#        _I('fontname2',fontname2,choices=fonts),
#        _I('character2',character2,max=1),
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
