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
from gui import widgets

"""Curves

Examples showing the use of the 'curve' plugin

level = 'normal'
topics = ['geometry','curves','spline',]
techniques = ['solve','widgets','persistence']
"""

def BezierCurve(X,curl=None,closed=False):
    """Create a Bezier curve between 4 points"""
    ns = (X.shape[0]-1) / 3
    ip = 3*arange(ns+1)
    P = X[ip]
    ip = 3*arange(ns)
    ic = column_stack([ip+1,ip+2]).ravel()
    C = X[ic].reshape(-1,2,3)
    return BezierSpline(P,control=C,closed=closed)
    

method = ODict([
    ('Natural Spline', NaturalSpline),
    ('Bezier Spline', BezierSpline),
    ('Cardinal Bezier Spline', CardinalSpline),
    ('Cardinal Spline', CardinalSpline2),
    ('Polyline', PolyLine),
    ('Bezier Curve', BezierCurve),
])

method_color = [ 'red','green','blue','cyan','magenta','yellow' ] 
point_color = [ 'black','white' ] 
        
open_or_closed = { True:'A closed', False:'An open' }

TA = None



def drawCurve(ctype,dset,closed,endcond,tension,curl,interpoints,ndiv,extend,spread):
    global TA
    P = dataset[dset]
    text = "%s %s with %s points" % (open_or_closed[closed],ctype.lower(),len(P))
    if TA is not None:
        undecorate(TA)
    TA = drawText(text,10,10)
    draw(P, color='black',marksize=3)
    drawNumbers(Formex(P))
    kargs = {'closed':closed}
    if ctype in ['Natural Spline']:
        kargs['endcond'] = [endcond,endcond]
    if ctype.startswith('Cardinal'):
        kargs['tension'] = tension
    if ctype in ['Bezier Spline']:
        curl = float(curl)
        kargs['curl'] = curl
    S = method[ctype](P,**kargs)
    if interpoints == 'subPoints':
        X = S.subPoints(ndiv,extend)
        point_color = 'black'
        msize = 3
    elif interpoints == 'pointsAt':
        npts = ndiv*S.nparts + (extend[1]-extend[0]) * ndiv + 1
        X = S.pointsAt(extend[0] + arange(npts)/ndiv)
        point_color = 'white'
        msize = 4
    else: #if interpoints == 'random':
        at = sort(random.rand(ndiv*S.nparts))*S.nparts
        print at
        X = S.pointsAt(at)
        point_color = 'blue'
        msize = 5
    
    PL = PolyLine(X,closed=closed)

    if spread:
        at = PL.atLength(PL.nparts)
        X = PL.pointsAt(at)
        PL = PolyLine(X,closed=closed)
        
    draw(X, color=point_color,marksize=msize)
    im = method.keys().index(ctype)
    draw(PL, color=method_color[im])


dataset = [
    Coords([[1., 0., 0.],[0., 1., 0.],[-1., 0., 0.],  [0., -1., 0.]]),
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
    ['DataSet','0','select',map(str,range(len(dataset)))], 
    ['CurveType',None,'select',method.keys()],
    ['Closed',False],
    ['EndCondition',None,'select',['notaknot','secder']],
    ['Tension',0.0],
    ['Curl',0.5],
    ['InterPoints',None,'select',['subPoints','pointsAt','random']],
    ['Nintervals',10],
    ['SpreadEqually',False],
    ['ExtendAtStart',0.0],
    ['ExtendAtEnd',0.0],
#    ['FreeSpaced',[-0.1,0.0,0.1,0.25,1.5,2.75]],
    ['Clear',True],
    ]
globals().update([i[:2] for i in data_items])
if GD.PF.has_key('_Curves_data_'):
    globals().update(GD.PF['_Curves_data_'])


clear()
setDrawOptions({'bbox':'auto','view':'front'})
linewidth(2)

for i,it in enumerate(data_items):
    data_items[i][1] = globals()[it[0]]

dialog = None

def save():
    """Save the data"""
    keys = [ i[0] for i in data_items ]
    values = [ globals()[k] for k in keys ]
    export({'_Curves_data_':dict(zip(keys,values))})


def close():
    if dialog:
        dialog.close()
    save()

def show(all=False):
    dialog.acceptData()
    globals().update(dialog.result)
    if Clear:
        clear()
    if all:
        Types = method.keys()
    else:
        Types = [CurveType]
    setDrawOptions({'bbox':'auto'})
    for Type in Types:
        drawCurve(Type,int(DataSet),Closed,EndCondition,Tension,Curl,InterPoints,Nintervals,[ExtendAtStart,ExtendAtEnd],SpreadEqually)
        setDrawOptions({'bbox':None})

def showAll():
    show(all=True)

    
dialog = widgets.InputDialog(data_items,caption='Curve parameters',actions = [('Close',close),('Show All',showAll),('Show',show)],default='Show')
dialog.show()

# End
