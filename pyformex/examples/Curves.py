# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
techniques = ['widgets','persistence','import','spline']
"""

from plugins.curve import *
from plugins.nurbs import *
from odict import ODict
from gui.widgets import InputDialog

ctype_color = [ 'red','green','blue','cyan','magenta','yellow','white' ] 
point_color = [ 'black','white' ] 
        
open_or_closed = { True:'A closed', False:'An open' }

TA = None


curvetypes = [
    'PolyLine',
    'Quadratic Bezier Spline',
    'Cubic Bezier Spline',
    'Natural Spline',
    'Nurbs Curve',
]


def drawCurve(ctype,dset,closed,endcond,curl,ndiv,ntot,extend,spread,drawtype,cutWP=False,scale=None,directions=False):
    global S,TA
    P = dataset[dset]
    text = "%s %s with %s points" % (open_or_closed[closed],ctype.lower(),len(P))
    if TA is not None:
        undecorate(TA)
    TA = drawText(text,10,10,font='sans',size=20)
    draw(P, color='black',nolight=True)
    drawNumbers(Formex(P))
    if ctype == 'PolyLine':
        S = PolyLine(P,closed=closed)
    elif ctype == 'Quadratic Bezier Spline':
        S = BezierSpline(P,degree=2,closed=closed,curl=curl,endzerocurv=(endcond,endcond))
    elif ctype == 'Cubic Bezier Spline':
        S = BezierSpline(P,closed=closed,curl=curl,endzerocurv=(endcond,endcond))
    elif ctype == 'Natural Spline':
        S = NaturalSpline(P,closed=closed,endzerocurv=(endcond,endcond))
        directions = False
    elif ctype == 'Nurbs Curve':
        S = NurbsCurve(P,closed=closed)#,blended=closed)
        scale = None
        directions = False
        drawtype = 'Curve'

    if scale:
        S = S.scale(scale)

    im = curvetypes.index(ctype)
    print "%s control points" % S.coords.shape[0]
    #draw(S.coords,color=red,nolight=True)

    if drawtype == 'Curve':
        draw(S,color=ctype_color[im],nolight=True)

    else:
        if spread:
            #print ndiv,ntot
            PL = S.approx(ndiv=ndiv,ntot=ntot)
        else:
            #print ndiv,ntot
            PL = S.approx(ndiv=ndiv)

        if cutWP:
            PC = PL.cutWithPlane([0.,0.42,0.],[0.,1.,0.])
            draw(PC[0],color=red)
            draw(PC[1],color=green)
        else:
            draw(PL, color=ctype_color[im])
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
    _I('DataSet','0',choices=map(str,range(len(dataset)))), 
    _I('CurveType',choices=curvetypes),
    _I('Closed',False),
    _I('EndCurvatureZero',False),
    _I('Curl',1./3.),
    _I('Ndiv',10),
    _I('SpreadEvenly',False),
    _I('Ntot',40),
    _I('ExtendAtStart',0.0),
    _I('ExtendAtEnd',0.0),
    _I('Scale',[1.0,1.0,1.0]),
    _I('DrawAs',None,'hradio',choices=['Curve','Polyline']),
    _I('Clear',True),
    _I('ShowDirections',False),
    _I('CutWithPlane',False),
    ]

clear()
setDrawOptions({'bbox':'auto','view':'front'})
linewidth(2)
flat()

dialog = None

import script


def close():
    global dialog
    if dialog:
        dialog.close()
        dialog = None
    # Release scriptlock
    scriptRelease(__file__)



def show(all=False):
    dialog.acceptData()
    globals().update(dialog.results)
    export({'_Curves_data_':dialog.results})
    if Clear:
        clear()
    if all:
        Types = curvetypes
    else:
        Types = [CurveType]
    setDrawOptions({'bbox':'auto'})
    for Type in Types:
        drawCurve(Type,int(DataSet),Closed,EndCurvatureZero,Curl,Ndiv,Ntot,[ExtendAtStart,ExtendAtEnd],SpreadEvenly,DrawAs,CutWithPlane,Scale,ShowDirections)
        setDrawOptions({'bbox':None})

def showAll():
    show(all=True)

def timeOut():
    showAll()
    wait()
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
# Block other scripts 
scriptLock(__file__)



# End
