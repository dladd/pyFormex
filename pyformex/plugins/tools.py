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

# End
