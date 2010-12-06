#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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
"""Curves

Examples showing the use of the 'curve' plugin

level = 'normal'
topics = ['geometry','curve']
techniques = ['solve','widgets','persistence','import','spline']
"""

from plugins.curve import *
from odict import ODict
from gui.widgets import InputDialog, simpleInputItem as I



def BezierCurve(X,curl=None,closed=False):
    """Create a Bezier curve give 4 points

    This currently does not allow closed==True
    """
    ns = (X.shape[0]-1) / 3
    ip = 3*arange(ns+1)
    P = X[ip]
    ip = 3*arange(ns)
    ic = column_stack([ip+1,ip+2]).ravel()
    C = X[ic].reshape(-1,2,3)
    # always use False
    return BezierSpline(P,control=C,closed=False)
    

method = ODict([
    ('Bezier Spline', BezierSpline),
    ('Quadratic Bezier Spline', QuadBezierSpline),
    ('Cardinal Spline', CardinalSpline),
    ('Cardinal Spline2', CardinalSpline2),
    ('Natural Spline', NaturalSpline),
    ('Polyline', PolyLine),
    ('Bezier Curve', BezierCurve),
])

method_color = [ 'red','green','blue','cyan','magenta','yellow','white' ] 
point_color = [ 'black','white' ] 
        
open_or_closed = { True:'A closed', False:'An open' }

TA = None



def drawCurve(ctype,dset,closed,endcond,tension,curl,ndiv,ntot,extend,spread,drawtype,cutWP=False,scale=None,directions=False):
    global TA
    P = dataset[dset]
    text = "%s %s with %s points" % (open_or_closed[closed],ctype.lower(),len(P))
    if TA is not None:
        undecorate(TA)
    TA = drawText(text,10,10,font='sans',size=20)
    draw(P, color='black')
    drawNumbers(Formex(P))
    kargs = {'closed':closed}
    if ctype in ['Natural Spline','Bezier Spline','Cardinal Spline']:
        kargs['endzerocurv'] = (endcond,endcond)
    if ctype.startswith('Cardinal'):
        kargs['tension'] = tension
    if ctype in ['Bezier Spline']:
        curl = float(curl)
        kargs['curl'] = curl
    S = method[ctype](P,**kargs)

    if scale:
        S = S.scale(scale)

    print "%s points on the curve" % S.pointsOn().shape[0]
    draw(S.pointsOff(),color=red)
    #print "coeffs %s" % S.coeffs

    if spread:
        #print ndiv,ntot
        PL = S.approx(ndiv=ndiv,ntot=ntot)
    else:
        #print ndiv,ntot
        PL = S.approx(ndiv=ndiv)
        
    im = method.keys().index(ctype)
    if cutWP:
        PC = PL.cutWithPlane([0.,0.42,0.],[0.,1.,0.])
        draw(PC[0],color=red)
        draw(PC[1],color=green)
    else:
        draw(PL, color=method_color[im])
    draw(PL.pointsOn(),color=black)
    
    if directions:
        t = arange(2*S.nparts+1)*0.5
        ipts = S.pointsAt(t)
        draw(ipts)
        idir = S.directionsAt(t)
        drawVectors(ipts,0.2*idir)
        
    

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
    Coords([[0., 1., 0.],[0., 0.1, 0.],[0.1, 0., 0.],  [1., 0., 0.]]),
    Coords([[0., 1., 0.],[0.,0.,0.],[0.,0.,0.],[1., 0., 0.]]),
    #Coords([[0., 1., 0.],[1., 0., 0.]]),
    ]

data_items = [
    I('DataSet','0',choices=map(str,range(len(dataset)))), 
    I('CurveType',choices=method.keys()),
    I('Closed',False),
    I('EndCurvatureZero',False),
    I('Tension',0.0),
    I('Curl',0.5),
    I('Ndiv',10),
    I('SpreadEvenly',False),
    I('Ntot',40),
    I('ExtendAtStart',0.0),
    I('ExtendAtEnd',0.0),
    I('Scale',[1.0,1.0,1.0]),
    I('DrawAs',None,'hradio',choices=['Curve','Polyline']),
    I('Clear',True),
    I('ShowDirections',False),
    I('CutWithPlane',False),
    ]

clear()
setDrawOptions({'bbox':'auto','view':'front'})
linewidth(2)

dialog = None


def close():
    global dialog
    if dialog:
        dialog.close()
        dialog = None


def show(all=False):
    dialog.acceptData()
    globals().update(dialog.results)
    export({'_Curves_data_':dialog.results})
    if Clear:
        clear()
    if all:
        Types = method.keys()
    else:
        Types = [CurveType]
    setDrawOptions({'bbox':'auto'})
    for Type in Types:
        drawCurve(Type,int(DataSet),Closed,EndCurvatureZero,Tension,Curl,Ndiv,Ntot,[ExtendAtStart,ExtendAtEnd],SpreadEvenly,DrawAs,CutWithPlane,Scale,ShowDirections)
        setDrawOptions({'bbox':None})

def showAll():
    show(all=True)

def timeOut():
    showAll()
    close()
    
dialog = widgets.InputDialog(
    data_items,
    caption='Curve parameters',
    actions = [('Close',close),('Clear',clear),('Show All',showAll),('Show',show)],
    default='Show')

if pf.PF.has_key('_Curves_data_'):
    #print pf.PF['_Curves_data_']
    dialog.updateData(pf.PF['_Curves_data_'])

dialog.timeout = timeOut
dialog.show()
       

# End
