#!/usr/bin/env pyformex --gui
# $Id$

"""Nurbs

level = 'advanced'
topics = ['geometry', 'curve']
techniques = ['nurbs','border']

.. Description

Nurbs
=====
"""

import simple
from plugins.curve import *
from plugins.nurbs import *


def expandNurbsCurve():
    def directions(self,u=None,n=10,d=1):
        print "DERIV",u
        if u is None:
            umin = self.knots[0]
            umax = self.knots[-1]
            u = umin + arange(n+1) * (umax-umin) / n
        
        ctrl = self.coords.astype(double)
        knots = self.knots.astype(double)
        u = asarray(u).astype(double)
        
        try:
            pts = nurbs.bspdeval(self.degree,ctrl,knots,u[0],1)
            if isnan(pts).any():
                print "We got a NaN"
                print pts
                raise RuntimeError
        except:
            raise RuntimeError,"Some error occurred during the evaluation of the Nurbs curve"

        #print pts
        if pts.shape[-1] == 4:
            #pts = Coords4(pts).toCoords()
            pts = Coords(pts[...,:3])
            x = pts[0]
            d = normalize(pts[1])
        else:
            pts = Coords(pts)
        return x,d
    NurbsCurve.directions = directions


def drawThePoints(N,n,color=None):
    umin = N.knots[N.degree]
    umax = N.knots[-N.degree-1]
    #umin = N.knots[0]
    #umax = N.knots[-1]
    print "Umin = %s, Umax = %s" % (umin,umax)
    u = umin + arange(n+1) * (umax-umin) / float(n)
    print u
    #u = [0.25,0.5,0.75]
    P = N.pointsAt(u)    
    draw(P,color=color,marksize=5)
    drawNumbers(P,color=color)

    for ui in u:
        #x = N.pointsAt([ui])
        #draw(x,marksize=10,color=yellow)
        x,d = N.directions([ui])
        print "Point %s: Dir %s" % (x,d)
        draw(x,marksize=10,color=yellow)
        draw(Formex([[x,x+0.5*d]]),color=yellow,linewidth=3)

expandNurbsCurve()                         
clear()
linewidth(2)
flat()

#    3*0    -     2*1     -    3*2    : 8 = 5+3
#    nctrl = nparts * degree + 1 
#    nknots = nctrl + degree + 1
#    nknots = (nparts+1) * degree + 2
#
# degree  nparts  nctrl   nknots
#    2      1       3        6
#    2      2       5        8
#    2      3       7       10
#    2      4       9       12
#    3      1       4        8
#    3      2       7       11
#    3      3      10       14
#    3      4      13       17
#    4      1       5       10 
#    4      2       9       14
#    4      3      13       18
#    5      1       6       12
#    5      2      11       17
#    5      3      16       22
#    6      1       7       14       
#    6      2      13       20
#    7      1       8       16
#    8      1       9       18

# This should be a valid combination of ntrl/degree
# drawing is only done if degree <= 7

def demo_weights():
    nctrl = 8
    degree = 7
    pts = allpts[:nctrl]
    knots = None
    L = {}
    draw(pts)
    drawNumbers(pts)
    draw(PolyLine(pts))
    for w,c in zip(weight,colors):
        qc = Coords4(pts)
        for i in range(1,degree):
            qc[i::degree].deNormalize(w)
        print qc
        C = NurbsCurve(qc,knots=knots,degree=degree,closed=False)
        draw(C,color=c)
        drawThePoints(C,10,color=c)
        L["wt-%s" % w] = C
    export(L)


C = Formex(pattern('51414336')).toCurve()

def drawControl():
    pts = C.coords
    draw(C)
    nctrl = C.ncoords()
    draw(pts)
    drawNumbers(pts)
    draw(PolyLine(pts),bbox='auto',view='front')


def drawNurbs(degree,closed,blended,weighted=False):
    pts = C.coords
    if closed:
        pts = pts[:-1]
        blended=True
    if not blended:
        npts = ((len(pts)-1) // degree) * degree + 1
        pts = pts[:npts]
    if weighted:
        wts = array([1.]*len(pts))
        wts[1::2] = 0.5
        print wts,wts.shape
    else:
        wts=None
    print pts.shape
    N = NurbsCurve(pts,wts=wts,degree=degree,closed=closed,blended=blended)
    draw(N,color=red)
    drawThePoints(N,20,color=black)


clear()
setDrawOptions({'bbox':None})
linewidth(2)
flat()
drawControl()

dialog = None


def close():
    global dialog
    if dialog:
        dialog.close()
        dialog = None


def show():
    dialog.acceptData()
    res = dialog.results
    if res['clear']:
        clear()
        drawControl()
    export({'_Nurbs_data_':res})
    del res['clear']  # remove non-argument items
    drawNurbs(**res)


def showAll():
    for closed,blended in [ (True,True), (False,True), (False,False)]:
        for degree in range(6):
            drawNurbs(degree=degree,closed=closed,blended=blended)
            

def timeOut():
    showAll()
    close()


data_items = [
    _I('degree',2),
    _I('closed',False),
    _I('blended',True,enabled=False),
    _I('weighted',False),
    _I('clear',True),
    ]
input_enablers = [
    ('closed',False,'blended'),
    ]
  
dialog = Dialog(
    data_items,
    enablers = input_enablers,
    caption = 'Nurbs parameters',
    actions = [('Close',close),('Clear',clear),('Show All',showAll),('Show',show)],
    default = 'Show',
    )

if pf.PF.has_key('_Nurbs_data_'):
    dialog.updateData(pf.PF['_Nurbs_data_'])

dialog.timeout = timeOut
dialog.show()
       

exit()

draw(simple.circle(2,4),linewidth=3)
xm = 0.5*(pts[0]+pts[2])
xn = 0.5*(xm+pts[1])
draw([xm,pts[1],xn],marksize=10)

pause()
clear()
wireframe()
transparent(False)
lights(False)
grid = simple.rectangle(4,3,diag='d')
draw(grid,color=blue)
print grid.shape()
grid.eltype='nurbs'
draw(grid,color=red)
print grid.shape()
grid.eltype='curve'
draw(grid,color=yellow)
print grid.shape()

exit()
## for order,color in zip(range(2,5),[cyan,magenta,yellow]):
##     if order > len(pts):
##         continue
##     N[order] = NurbsActor(pts,closed=closed,order=order,color=color)
##     drawActor(N[order])
for order,color in zip(range(2,5),[red,green,blue]):
    if order > len(pts):
        continue
    print wts.shape
    N[order] = NurbsActor(pts,wts=wts,closed=False,order=order,color=color)
    drawActor(N[order])

exit()
drawNumbers(pts*wts,color=red)
for order,color in zip(range(2,5),[cyan,magenta,yellow]):
    N[order] = NurbsActor(pts*wts,closed=closed,order=order,color=color)
    drawActor(N[order])

exit()


NO = NurbsActor(pts*wts,closed=closed,order=3,color=blue)
drawActor(NO)

NO = NurbsActor(pts*wts,closed=closed,order=2,color=cyan)
drawActor(NO)

## def distance(C):
##     d = C.approx(100).pointsOn().distanceFromPoint(origin())
##     return 1.-d.min(),1.-d.max()

## clear()
## S = NaturalSpline(pts*wts,closed=True)
## draw(S,color=magenta)
## draw(S.pointsOn(),color=magenta)
## drawNumbers(S.pointsOn(),color=magenta)
## print distance(S)

## #from gui.colors import *
## for c,curl in zip([black,red,green,blue],[ 0.375055, 0.375058, 0.37506  ]):
##     C = BezierSpline(pts*wts,closed=True,curl=curl)
##     draw(C,color=c)
##     print distance(C)


zoomAll()

# En1
