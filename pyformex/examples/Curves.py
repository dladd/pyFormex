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
topics = ['geometry','curves']
techniques = ['spline','solve','widgets']
"""

class BezierCurve(Curve):
    """A class representing a Bezier curve."""
    coeffs = matrix([[-1.,  3., -3., 1.],
                     [ 3., -6.,  3., 0.],
                     [-3.,  3.,  0., 0.],
                     [ 1.,  0.,  0., 0.]]
                    )#pag.440 of open GL

    def __init__(self,pts,deriv=None,curl=0.5,control=None,closed=False):
        """Create a Bezier curve through the given points.

        The curve is defined by the points and the directions at these points.
        If no directions are specified, the average of the segments ending
        in that point is used, and in the end points of an open curve, the
        direction of the end segment.
        The curl parameter can be set to influence the curliness of the curve.
        curl=0.0 results in straight segment.
        """
        pts = Coords(pts)
        self.coords = pts
        self.nparts = self.coords.shape[0]
        if not closed:
            self.nparts -= 1
            
        if control is None:
            P = PolyLine(pts,closed=closed)
            if deriv is None:
                deriv = P.avgDirections()
                if not closed:
                    atends = P.directions()
                    deriv = Coords.concatenate([atends[:1],deriv,atends[-1:]])
                curl = curl * P.lengths()
                if not closed:
                    curl = concatenate([curl,curl[-1:]])
                print curl.shape
                print deriv.shape
                curl = curl.reshape(-1,1)
                deriv *= curl
                print deriv.shape
                control = concatenate([P+deriv,P-deriv])
                print control.shape
        self.control = Coords(control)
        self.closed = closed


    def sub_points(self,t,j):
        n = self.coords.shape[0]
        ind = [j,(j+1)%n]
        P = self.coords[ind]
        D = self.control[ind]
##         if self.curl is None:
##             D = P + self.deriv[ind]
##         else:
##             print self.curl,type(self.curl)
##             print self.deriv,type(self.deriv)
##             D = P + (self.curl[ind]*array([1.,-1.])).reshape(-1,1)*self.deriv[ind]
        P = concatenate([ P[0],D[0],D[1],P[1] ],axis=0).reshape(-1,3)
        C = self.coeffs * P
        U = column_stack([t**3., t**2., t, ones_like(t)])
        X = dot(U,C)
        return X

def Bezier4(X,curl=0.5,closed=False):
    """Create a Bezier curve between 4 points"""
    draw(PolyLine(X))
    nx = X.shape[0]
    P = X[arange(0,nx,3)]
          
    if curl is not None:
        D = Coords([X[1]-X[0],X[3]-X[2]])
        print "D: %s, type %s" % (D,type(D))
        D = normalize(D)
        print "D: %s, type %s" % (D,type(D))
    else:
        D = Coords([X[1]-X[0],X[2]-X[3]])
    return BezierCurve(P,D,curl,closed)
    

defaultDataSet = '0'
defaultCurveType = 'Bezier'

method = ODict([
    ('Natural Spline', NaturalSpline),
    ('Cardinal Spline', CardinalSpline),
    ('Bezier', BezierCurve),
    ('Polyline', PolyLine),
    ('Bezier4', Bezier4),
])

method_color = [ 'red','green','blue','cyan', 'yellow' ] 
point_color = [ 'black','white' ] 
        
open_or_closed = { True:'A closed', False:'An open' }

TA = None


    

def drawCurve(ctype,dset,closed,endcond,tension,curl,interpoints,ndiv,extend):
    global TA
    P = dataset[dset]
    text = "%s %s with %s points" % (open_or_closed[closed],ctype.lower(),len(P))
    if TA is not None:
        undecorate(TA)
    TA = drawtext(text,10,10)
    draw(P, color='black',marksize=3)
    drawNumbers(Formex(P))
    kargs = {'closed':closed}
    if ctype in ['Natural Spline']:
        kargs['endcond'] = [endcond,endcond]
    if ctype in ['Cardinal Spline']:
        kargs['tension'] = tension
    if ctype in ['Bezier','Bezier4']:
        if curl == 'None':
            curl = None
        else:
            curl = float(curl)
        kargs['curl'] = curl
    S = method[ctype](P,**kargs)
    if interpoints == 'subPoints':
        X = S.subPoints(ndiv,extend)
        point_color = 'back'
    else:
        npts = ndiv*S.nparts + (extend[1]-extend[0]) * ndiv + 1
        X = S.pointsAt(extend[0] + arange(npts)/ndiv)
        point_color = 'white'
    draw(X, color=point_color,marksize=3)
    im = method.keys().index(ctype)
    draw(PolyLine(X,closed=closed), color=method_color[im], linewidth=1)


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
    ['DataSet',defaultDataSet,'select',map(str,range(len(dataset)))], 
    ['CurveType',defaultCurveType,'select',method.keys()],
    ['Closed',False],
    ['EndCondition',None,'select',['notaknot','secder']],
    ['Tension',0.0],
    ['Curl','0.5'], # A string, to allow None
    ['InterPoints',None,'select',['subPoints','pointsAt']],
    ['Nintervals',10],
    ['ExtendAtStart',0.0],
    ['ExtendAtEnd',0.0],
    ['Clear',True],
    ]
globals().update([i[:2] for i in data_items])


clear()
setDrawOptions({'bbox':'auto','view':'front'})

for i,it in enumerate(data_items):
    data_items[i][1] = globals()[it[0]]

dialog = None

def close():
    if dialog:
        dialog.close()

def show():
    dialog.acceptData()
    globals().update(dialog.result)
    if Clear:
        clear()
    drawCurve(CurveType,int(DataSet),Closed,EndCondition,Tension,Curl,InterPoints,Nintervals,[ExtendAtStart,ExtendAtEnd])
    

    
dialog = widgets.InputDialog(data_items,caption='Curve parameters',actions = [('Close',close),('Show',show)],default='Show')
dialog.show()

#while not GD.dialog_timeout:
#    sleep(5)
#    GD.app.processEvents()

# End
