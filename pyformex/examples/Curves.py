#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
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

from plugins.curve import *
from odict import ODict

"""Curves

Examples showing the use of the 'curve' plugin

level = 'normal'
topics = ['geometry','curves']
techniques = ['spline','solve']
"""


method = {
    'Natural Spline': NaturalSpline,
    'Cardinal Spline': CardinalSpline,
    'Bezier': BezierCurve,
    'Polyline': PolyLine
}
        
open_or_closed = { True:'A closed', False:'An open' }

TA = None

def drawCurve(ctype,dset,closed,tension,curl,ndiv,extend):
    global TA
    P = dataset[dset]
    text = "%s %s with %s points" % (open_or_closed[closed],ctype.lower(),len(P))
    if TA is not None:
        undecorate(TA)
    TA = drawtext(text,10,10)
    draw(P, color='black',marksize=3)
    drawNumbers(Formex(P))
    kargs = {'closed':closed}
    if ctype in ['Cardinal Spline']:
        kargs['tension'] = tension
    if ctype in ['Bezier']:
        kargs['curl'] = curl
    S = method[ctype](P,**kargs)
    X = S.points(ndiv,extend)
    draw(X, color='red',marksize=3)
    draw(PolyLine(X,closed=closed), color='blue', linewidth=1)


dataset = [
    Coords([[6., 7., 12.],[9., 5., 6.],[11., -2., 6.],  [9.,  -4., 14.]]),
    Coords([[-5., -10., -4.], [-3., -5., 2.],[-4., 0., -4.], [-4.,  5, 4.],
            [6., 3., -1.], [6., -9., -1.]]),
    Coords([[-1., 7., -14.], [-4., 7., -8.],[-7., 5., -14.],[-8., 2., -14.],
            [-7.,  0, -6.], [-5., -3., -11.], [-7., -4., -11.]]),
    Coords([[-1., 1., -4.], [1., 1., 2.],[2.6, 2., -4.], [2.9,  3.5, 4.],
            [2., 4., -1.],[1.,3., 1.], [0., 0., 0.], [0., -3., 0.],
            [2., -1.5, -2.], [1.5, -1.5, 2.], [0., -8., 0.], [-1., -8., -1.],
            [3., -3., 1.]]),
    ]

data_items = [
    ['DataSet','3','select',map(str,range(len(dataset)))], 
    ['CurveType',None,'select',method.keys()],
    ['Closed',True],
    ['Nintervals',10],
    ['ExtendAtStart',0.0],
    ['ExtendAtEnd',0.0],
    ['Tension',0.0],
    ['Curl',0.5],
    ['Clear',True],
    ]
globals().update([i[:2] for i in data_items])


clear()
setDrawOptions({'bbox':'auto','view':'front'})
while not GD.dialog_timeout:
    for i,it in enumerate(data_items):
        data_items[i][1] = globals()[it[0]]
    res = askItems(data_items)
    if not res:
        break
    globals().update(res)
    if Clear:
        clear()
    drawCurve(CurveType,int(DataSet),Closed,Tension,Curl,Nintervals,
              [ExtendAtStart,ExtendAtEnd])


# End
