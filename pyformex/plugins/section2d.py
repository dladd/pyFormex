#!/usr/bin/env python pyformex.py
#
"""Some functions operating on 2D structures.

This is a plugin for pyFormex.
(C) 2002 B. Verhegghe
"""

from plugins import sectionize
from numpy import *

## We should probably turn this into a class planeSection

class planeSection(object):
    """A class describing a general 2D section.

    The 2D section is the area inside a closed curve in the (x,y) plane.
    The curve is decribed by a finite number of points and by straight
    segments connecting them.
    """

    def __init__(F):
        """Initialize a plane section.

        Initialization can be done either by a list of points or a set of line
        segments.

        1. By Points
        Each point is connected to the following one, and (unless they are
        very close) the last one back to the first. Traversing the resulting
        path should rotate positively around the z axis to yield a positive
        surface.

        2. By Segments
        It is the responsibilty of the user to ensure that the segments
        form a closed curve. If not, the calculated section data will be
        rather meaningless.
        """
        if F.nplex() == 1:
            self.F = sectionize.connectPoints(F,close=True)
        elif F.nplex() == 2:
            self.F = F
        else:
            raise ValueError,"Expected a plex-1 or plex-2 Formex"

        
def loopCurve(elems):
    """Check if a set of line elements form a closed curve.

    elems is a connection table of line elements, such as obtained
    from the feModel() method on a plex-2 Formex.

    A new connection table is returned which is equivalent to the input
    if it forms a closed loop, but has the elements in order and sense
    along the curve.
    """
    srt = zeros_like(elems) - 1
    ie = 0
    je = 0
    rev = False
    while True:
        if rev:
            srt[ie] =  elems[je][[1,0]]
        else:
            srt[ie] =  elems[je]
        elems[je] = [ -1,-1 ] # Done with this one
        #print srt
        #print elems
        j = srt[ie][1]
        if j == 0:
            #print "Finished"
            break
        w = where(elems == j)
        if w[0].size == 0:
            print "No match found"
        je = w[0][0]
        ie += 1
        rev = w[1][0] == 1
    if any(srt == -1):
        print "The curve is not closed"
    return srt


def sectionChar(F):
    """Compute characteristics of plane sections.

    The plane sections are described by their circumference, consisting of a
    sequence of straight segments.
    The segment end point data are gathered in a plex-1 Formex.
    The z-value of the coordinates does not have to be specified,
    and will be ignored if it is.
    Each point is connected to the following one, and the last one to the first.
    The resulting path through the points should rotate positively around the
    z axis to yield a positive surface.

    The return value is a dict with the following characteristics:
    'L'   : circumference,
    'A'   : enclosed surface,
    'Sx'  : first area moment around global x-axis
    'Sy'  : first area moment around global y-axis
    'Ixx' : second area moment around global x-axis
    'Iyy' : second area moment around global y-axis
    'Ixy' : product moment of area around global x,y-axes
    """
    if F.nplex() != 2:
        raise ValueError, "Expected a plex-1 Formex!"
    GD.debug("The circumference has %d segments" % F.nelems())
    x = F.x()
    y = F.y()
    x0 = x[:,0]
    y0 = y[:,0]
    x1 = x[:,1]
    y1 = y[:,1]
    a = (x0*y1 - x1*y0) / 2
    return {
        'L'   : sqrt((x1-x0)**2 + (y1-y0)**2).sum(),
        'A'   : a.sum(),
        'Sx'  : (a*(y0+y1)).sum()/3,
        'Sy'  : (a*(x0+x1)).sum()/3,
        'Ixx' : (a*(y0*y0+y0*y1+y1*y1)).sum()/6,
        'Iyy' : (a*(x0*x0+x0*x1+x1*x1)).sum()/6,
        'Ixy' :-(a*(x0*y0+x1*y1+(x0*y1+x1*y0)/2)).sum()/6,
        }


def extendedSectionChar(S):
    """Computes extended section characteristics for the given section.

    S is a dict with section basic section characteristics as returned by
    sectionChar().
    This function computes and reutrns a dict with the following:
    'xG', 'yG' : coordinates of the center of gravity G of the plane section
    'IGxx', 'IGyy', 'IGxy' : second area moments and product around axes
       through G and  parallel with the global x,y-axes
    'alpha' : angle(in radians) between the glabla x,y axes and the principal
       axes (X,Y) of the section (X and Y always pass through G)
    'IXX','IYY': principal second area moments around X,Y respectively. (The
       second area product is always zero.)
    """
    xG =  S['Sy']/S['A']
    yG =  S['Sx']/S['A']
    IGxx = S['Ixx'] - S['A'] * yG**2
    IGyy = S['Iyy'] - S['A'] * xG**2
    IGxy = S['Ixy'] + S['A'] * xG*yG
    alpha,IXX,IYY = princTensor2D(IGxx,IGyy,IGxy)
    return {
        'xG'   : xG,
        'yG'   : yG,
        'IGxx' : IGxx,
        'IGyy' : IGyy,
        'IGxy' : IGxy,
        'alpha': alpha,
        'IXX'  : IXX,
        'IYY'  : IYY,
        }


def princTensor2D(Ixx,Iyy,Ixy):
    """Compute the principal values and directions of a 2D tensor.

    Returns a tuple with three values:
    - alpha: angle (in radians) from x-axis to principal X-axis
    - IXX,IYY: principal values of the tensor
    """
    from math import sqrt,atan2
    C = (Ixx+Iyy) * 0.5
    D = (Ixx-Iyy) * 0.5
    R = sqrt(D**2 + Ixy**2)
    IXX = C+R
    IYY = C-R
    alpha = atan2(Ixy,D) * 0.5
    return alpha,IXX,IYY
    

    
if __name__ == "draw":

    import simple

    
    def showaxes(C,angle,size,color):
        H = Formex(simple.Pattern['plus']).scale(0.6*size).rot(angle/rad).trl(C)
        draw(H,color=color)


    def square_example(scale=[1.,1.,1.]):
        P = Formex([[[1,1]]]).rosette(4,90).scale(scale)
        F = sectionize.connectPoints(P,close=True)
        draw(F)
        return sectionChar(F)

    def rectangle_example():
        return square_example(scale=[2.,1.,1.])

    def circle_example():
        H = simple.circle(5.,5.)
        draw(H)
        return sectionChar(H)
   
    
    def close_loop_example():
        # one more example, originally not a closed loop curve
        F = Formex(pattern('11')).replic(2,1,1) + Formex(pattern('2')).replic(2,2,0)
        nodes,elems = F.feModel()

        FN = Formex(nodes)
        drawNumbers(FN,color=blue)

        F = Formex(nodes[elems])
        draw(F)
        drawNumbers(F,color=red)

        print nodes
        print elems

        sorted = loopCurve(elems)

        print sorted

        ask('Click to continue',['Continue'])
        clear()
        F = Formex(nodes[sorted])
        draw(F,color=blue)
        print F.f
        return sectionChar(F)
    

    examples = { 'Square'    : square_example,
                 'Rectangle' : rectangle_example,
                 'Circle'    : circle_example,
                 'CloseLoop' : close_loop_example,
                 }
    
    res = askItems([('Select an example',examples.keys(),'select')])
    if res:
        S = examples[res['Select an example']]()
        S.update(extendedSectionChar(S))
        print S
        G = Formex([[[S['xG'],S['yG']]]])
        draw(G,bbox=None)
        showaxes([S['xG'],S['yG'],0.],S['alpha'],F.size(),'red')
 
