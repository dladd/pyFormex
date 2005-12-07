#!/usr/bin/env python
# $Id$
"""Color mapping of a range of values."""

from colors import *
from utils import stuur

# predefined color palettes
Palette = {
    'RGB' : [ red,green,blue ],
    'BGR' : [ blue,green,red ],
    'RWB' : [ red,white,blue ],
    'BWR' : [ blue,white,red ],
    'RWG' : [ red,white,green ],
    'GWR' : [ green,white,red ],
    'GWB' : [ green,white,blue ],
    'BWG' : [ blue,white,green ],
    'BW'  : [ black,None,white ],
    'WB'  : [ white,grey(0.5),black ],
}

class ColorScale:
    def __init__(self,palet,minval=0.,maxval=1.,midval=None):
        if type(palet) == str:
            self.palet = Palette.get(palet,Palette['RGB'])
        if self.palet[1] == None:
            self.palet[1] = [ 0.5*(p+q) for p,q in zip(self.palet[0],self.palet[2]) ]
        self.xmin = minval
        self.xmax = maxval
        if midval:
            self.x0 = midval
        else:
            self.x0 = 0.5*(minval+maxval)

    def scale(self,val):
        """Scale a value to the range -1...1."""
        return stuur(val,[self.xmin,self.x0,self.xmax],[-1.,0.,1.],1.)

    def color(self,val):
        """Return the color representing a value val.

        The returned color is a tuple of three RGB values in the range 0-1.
        """
        x = self.scale(val)
        c0 = self.palet[1]
        if x == 0.:
            return c0
        if x < 0:
            c1 = self.palet[0]
            x = -x
        else:
            c1 = self.palet[2]
        return tuple( [ (1.-x)*p + x*q for p,q in zip(c0,c1) ] )

        

if __name__ == "__main__":

    for palet in [ 'RGB', 'BW' ]:
        CS = ColorScale(palet,-50.,250.)
        for x in [ -50+10.*i for i in range(31) ]:
            print x,": ",CS.color(x)
    
