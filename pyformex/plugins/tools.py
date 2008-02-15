# $Id$

"""tools.py

Graphic Tools for pyFormex.
"""

from coords import *

class Plane(object):

    def __init__(self,point,normal):
        P = Coords(point)
        n = Coords(normal)

        if P.shape != (3,) or n.shape != (3,):
            raise ValueError,"point or normal does not have correct shape"

        self.P = P 
        self.n = n


    def point(self):
        return self.P

    def normal(self):
        return self.n
    
    def bbox(self):
        return self.P.bbox()

    def __str__(self):
        return 'P:%s n:%s' % (list(self.P),tuple(self.n)) 


################# Report information about picked objects ################

def report(K):
    if K.obj_type == 'elements':
        return reportElements(K)
    elif K.obj_type == 'points':
        return reportElements(K)
    else:
        return ''

def reportElements(K):
    s = "Element report\n"
    for k in K.keys():
        v = K[k]
        A = GD.canvas.actors[k]
        t = A.atype()
        s += "Actor %s (type %s); Elements %s\n" % (k,t,v)
        if t == 'Formex':
            e = A
        elif t == 'Surface':
            e = A.getElems()
        for p in v:
            s += "  Element %s: %s\n" % (p,e[p])
    return s


def reportPoints(K):
    s = "Point report\n"
    for k in K.keys():
        v = K[k]
        A = GD.canvas.actors[k]
        s += "Actor %s (type %s); Points %s\n" % (k,A.atype(),v)
        x = A.vertices()
        for p in v:
            s += "  Point %s: %s\n" % (p,x[p]) 
    return s

# End
