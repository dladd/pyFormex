#!/usr/bin/pyformex --gui
# $Id$

"""NurbsCurve

level = 'advanced'
topics = ['geometry', 'curve']
techniques = ['nurbs','connect','border']

.. Description

Nurbs
=====
"""

import simple
from plugins.curve import *
from plugins.nurbs import *
    


clear()
linewidth(2)
flat()


def drawThePoints(N,n,color=None):
    umin = N.knots[N.degree]
    umax = N.knots[-N.degree-1]
    #print "Umin = %s, Umax = %s" % (umin,umax)
    u = umin + arange(n+1) * (umax-umin) / float(n)
    P = N.pointsAt(u)    
    draw(P,color=color,marksize=5)
    drawNumbers(P,color=color)
    
    XD = N.derivatives(u,5)[:4]
    if XD.shape[-1] == 4:
        XD = XD.toCoords()
    x,d1,d2,d3 = XD[:4]
    e1,e2,e3,k,t = frenet(d1,d2,d3)
    #print t

    #k = 1./k
    #k[isnan(k)] = 0.
    k /= k[isnan(k) == 0].max()
    tmax = t[isnan(t) == 0].max()
    if tmax > 0:
        t /= tmax
    #print t
    s = 0.3
    x1 = x+s*e1
    x2 = x+s*e2
    x3 = x+s*e3
    x2k = x+k.reshape(-1,1)*e2 # draw curvature along normal
    x3t = x+t.reshape(-1,1)*e3
    draw(x,marksize=10,color=yellow)
    draw(connect([Formex(x),Formex(x1)]),color=yellow,linewidth=3)
    draw(connect([Formex(x),Formex(x2)]),color=cyan,linewidth=2)
    draw(connect([Formex(x),Formex(x2k)]),color=blue,linewidth=5)
    draw(connect([Formex(x),Formex(x3)]),color=magenta,linewidth=2)
    draw(connect([Formex(x),Formex(x3t)]),color=red,linewidth=5)


def drawNurbs(points,pointtype,degree,strategy,closed,blended,weighted=False,Clear=False):
    if Clear:
        clear()

    C = Formex(pattern(points)).toCurve()
    X = C.coords
    draw(C)
    draw(X,marksize=10)
    drawNumbers(X,leader='P',trl=[0.02,0.02,0.])
    if closed:
        # remove last point if it coincides with first
        x,e = Coords.concatenate([X[0],X[-1]]).fuse()
        if x.shape[0] == 1:
            X = X[:-1]
        blended=True
    draw(PolyLine(X,closed=closed),bbox='auto',view='front')
    if not blended:
        nX = ((len(X)-1) // degree) * degree + 1
        X = X[:nX]
    if weighted:
        wts = array([1.]*len(X))
        wts[1::2] = 0.5
        #print wts,wts.shape
    else:
        wts=None
    if pointtype == 'Control':
        N = NurbsCurve(X,wts=wts,degree=degree,closed=closed,blended=blended)
    else:
        N = globalInterpolationCurve(X,degree=degree,strategy=strategy)
    draw(N,color=red)
    #drawThePoints(N,11,color=black)


clear()
setDrawOptions({'bbox':None})
linewidth(2)
flat()

dialog = None


def close():
    global dialog
    if dialog:
        dialog.close()
        dialog = None
    # Release scriptlock
    scriptRelease(__file__)


def show():
    dialog.acceptData()
    res = dialog.results
    export({'_Nurbs_data_':res})
    drawNurbs(**res)

def showAll():
    dialog.acceptData()
    res = dialog.results
    export({'_Nurbs_data_':res})
    for points in predefined:
        print res
        res['points'] = points
        drawNurbs(**res)


def timeOut():
    showAll()
    wait()
    close()


predefined = [
    '51414336',
    '51i4143I36',
    '2584',
    '25984',
    '184',
    '514',
    '1234',
    '5858585858',
    '12345678',
    '121873',
    '1218973',
    '8585',
    '85985',
    '214121',
    '214412',
    '151783',
    'ABCDABCD',
    ]
    
data_items = [
    _I('points',text='Point set',choices=predefined),
    _I('pointtype',text='Point type',itemtype='select',choices=['Control','OnCurve']),
    _I('degree',2),
    _I('strategy',0.5),
    _I('closed',False),
    _I('blended',True,enabled=False),
    _I('weighted',False),
    _I('Clear',True),
    ]
input_enablers = [
    ('pointtype','OnCurve','strategy'),
    ('pointtype','Control','closed'),
    ('pointtype','Control','blended'),
    ('pointtype','Control','weighted'),
#    ('closed',False,'blended'),
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

# Block other scripts 
scriptLock(__file__)
       

# End
