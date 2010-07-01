#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
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

"""BezierSpline

level = 'normal'
topics = ['geometry', 'curve']
techniques = ['spline','dialog']

.. Description

Bezier Spline
=============

This example shows a collection of Bezier Spline curves thropugh a number of points. The points used are the corners and midside points of a unit square.
The user is asked for a number of points to use.
The image shows open (left) and closed (right) BezierSpline curves of
degrees 1(red), 2(magenta) and 3(blue).

"""
from gui.widgets import simpleInputItem as I
from plugins.curve import BezierSpline

clear()
linewidth(2)

# Predefined set of points
pts = Coords([
    [1.,0.,0.],
    [1.,1.,0.],
    [0.,1.,0.],
    [-1.,1.,0.],
    [-1.,0.,0.],
    [-1.,-1.,0.],
    [0.,-1.,0.],
    [1.,-1.,0.],
    [1.,0.,0.],
    ])

# Ask the user how many points he wants to use
res = askItems([I('npts',5,text='How many points to use (2..%s)' % len(pts))],legacy=False)
if not res:
    exit()

# Keep only the requested number of points
npts = res['npts']
pts = pts[:npts]

# Show open and closed Bezier Splines, for degrees 1,2,3
degrees = [1,2,3]
colors = [red,green,blue]
collection = {}
for closed in [False,True]:
    draw(pts)
    drawNumbers(pts)
    for d,c in zip(degrees,colors):
        print "DEGREE %s, %s" % (d,closed)
        B = BezierSpline(coords=pts,closed=closed,degree=d)
        draw(B,color=c)
        collection["BezierSpline-degree:%s-closed:%s" % (d,closed)] = B
    # Translate the points to the right
    pts = pts.trl(0,2.5)#[:-1]

zoomAll()
export(collection)

# End
