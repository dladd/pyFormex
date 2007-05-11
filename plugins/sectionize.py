#!/usr/bin/env python
# $Id$

"""sectionize.py

Create, measure and approximate cross section of a Formex.
"""

##import globaldata as GD
##from globaldata import PF
##import utils
##import timer
##from plugins import f2abq, stl, tetgen, stl_abq
##from gui import widgets,actors,colors
from gui.draw import *
from formex import *



def sectionize(F):
    """Sectionize a Formex along 0 axis."""
    clear()
    linewidth(1)
    draw(F,color='yellow')
    bb = F.bbox()
    GD.message("Bounding box = %s" % bb)

    itemlist = [['number of sections',20],['relative thickness',0.1]]
    res,accept = widgets.inputDialog(itemlist,'Sectioning Parameters').getResult()
    sections = []
    ctr = []
    diam = []
    if accept:
        n = int(res[0][1])
        th = float(res[1][1])
        xmin = bb[0][0]
        xmax = bb[1][0]
        dx = (xmax-xmin) / n
        dxx = dx * th
        X = xmin + arange(n+1) * dx
        GD.message("Sections are taken at X-values: %s" % X)

        c = zeros([n,3],float)
        d = zeros([n,1],float)
        linewidth(2)

        for i in range(n+1):
            G = F.clip(F.test(nodes='any',dir=0,min=X[i]-dxx,max=X[i]+dxx))
            draw(G,color='blue',view=None)
            GD.canvas.update()
            C = G.center()
            H = Formex(G.f-C)
            x,y,z = H.x(),H.y(),H.z()
            D = 2 * sqrt((x*x+y*y+z*z).mean())
            GD.message("Section Center: %s; Diameter: %s" % (C,D))
            sections.append(G)
            ctr.append(C)
            diam.append(D)
    return sections,ctr,diam


def drawCircles(sections,ctr,diam):
    """Draw circles as approximation of Formices."""
    import simple
    circle = simple.circle().rotate(-90,1)
    cross = Formex(simple.Pattern['plus']).rotate(-90,1)
    circles = []
    n = len(sections)
    for i in range(n):
        C = cross.translate(ctr[i])
        B = circle.scale(diam[i]/2).translate(ctr[i])
        S = sections[i]
        print C.bbox()
        print B.bbox()
        print S.bbox()
        clear()
        draw(S,view='left',wait=False)
        draw(C,color='red',bbox=None,wait=False)
        draw(B,color='blue',bbox=None)
        circles.append(B)
    return circles


def drawAllCircles(F,circles):
    clear()
    linewidth(1)
    draw(F,color='yellow',view='front')
    linewidth(2)
    for circ in circles:
        bb = circ.bbox()
        d = (bb[1] - bb[0]) * 0.5
        bb[0] -= d
        bb[1] += d
        draw(circ,color='blue',bbox=bb)
    zoomAll()


def connectPoints(ptlist):
    """Create a Formex connecting all points in the ptlist.

    ptlist is any (n,3) shaped structure (sequence|list|array) of floats.
    """
    Fc = Formex(array(ptlist).reshape((-1,1,3)))
    return connect([Fc,Fc],bias=[0,1])


# End
