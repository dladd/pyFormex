#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.6 Release Sun Sep 30 14:33:15 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
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
    """A colorscale maps a range of values into colors."""
    
    def __init__(self,palet,minval=0.,maxval=1.,midval=None,exp=1.0,exp2=None):
        """Create a colorscale to map a range of values into colors.

        The values range from minval to maxval (default 0.0..1.0).

        A midval may be specified to set the value corresponding to
        the midle of the color scale. It defaults to the middle value
        of the range. It is especially useful if the range extends over
        negative and positive values to set 0.0 as the middle value. 

        The palet is a list of 3 colors, corresponding to the minval,
        midval and maxval respectively. The middle color may be given
        as None, in which case it will be set to the middle color
        between the first and last.

        Some useful palets are predefined in Palette.

        Mapping values to colors is linear by default. Nonlinear
        mappings can be obtained by specifying an exponent. Mapping
        is done with the stuur function from utils.py.
        If 2 exponents are given, mapping is done independently in the
        minval..midval range with exp and in the midval..maxval range
        with exp2.
        """
        #print palet,minval,maxval,midval
        if type(palet) == str:
            self.palet = Palette.get(palet.upper(),Palette['RGB'])
        else:
            self.palet = palet
        if self.palet[1] == None:
            self.palet[1] = [ 0.5*(p+q) for p,q in zip(self.palet[0],self.palet[2]) ]
        self.xmin = minval
        self.xmax = maxval
        if midval == None:
            self.x0 = 0.5*(minval+maxval)
        else:
            self.x0 = midval
        self.exp = exp
        self.exp2 = exp2

    def scale(self,val):
        """Scale a value to the range -1...1."""
        if self.exp2 == None:
            return stuur(val,[self.xmin,self.x0,self.xmax],[-1.,0.,1.],self.exp)
        return self.scale2(val)
    
    def scale2(self,val):
        """Scale a value to the range -1...1.

        This scales indepently in one of the intervals
        xmin..x0 or x0..xmax.
        """
        if val < self.x0:
            return stuur(val,[self.xmin,(self.x0+self.xmin)/2,self.x0],[-1.,-0.5,0.],self.exp)
        else:
            return stuur(val,[self.x0,(self.x0+self.xmax)/2,self.xmax],[0.,0.5,1.0],1./self.exp2)


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


class ColorLegend:
    """A colorlegend is a colorscale divided in a number of subranges."""

    def __init__(self,colorscale,n):
        """Create a color legend dividing a colorscale in n subranges.

        The full value range of the colorscale is divided in n subranges,
        each half range being divided in n/2 subranges.
        This sets n+1 limits of the subranges.
        The n colors of the subranges correspond to the subrange middle value.
        """
        self.cs = colorscale
        n = int(n)
        r = float(n)/2
        m = (n+1)/2
        vals = [ (self.cs.xmin*(r-i)+self.cs.x0*i)/r for i in range(m) ]
        val2 = [ (self.cs.xmax*(r-i)+self.cs.x0*i)/r for i in range(m) ]
        val2.reverse()
        if n % 2 == 0:
            vals += [ self.cs.x0 ]
        vals += val2
        midvals = [ (vals[i] + vals[i+1])/2 for i in range(n) ]
        self.limits = vals 
        self.colors = map(self.cs.color,midvals)
        self.underflowcolor = None
        self.overflowcolor = None

    def overflow(self,oflow=None):
        """Raise a runtime error if oflow == None, else return oflow."""
        if oflow==None:
            raise RuntimeError, "Value outside colorscale range"
        else:
            return oflow

    def color(self,val):
        """Return the color representing a value val.

        The color is that of the subrange holding the value. If the value
        matches a subrange limit, the lower range color is returned.
        If the value falls outside the colorscale range, a runtime error
        is raised, unless the corresponding underflowcolor or overflowcolor
        attribute has been set, in which case this attirbute is returned.
        Though these attributes can be set to any not None value, it will
        usually be set to some color value, that will be used to show
        overflow values.
        The returned color is a tuple of three RGB values in the range 0-1.
        """
        i = 0
        while self.limits[i] < val:
            i += 1
            if i >= len(self.limits):
                return self.overflow(self.overflowcolor)
        if i==0:
            return self.overflow(self.underflowcolor)
        return self.colors[i-1]
        

if __name__ == "__main__":

    for palet in [ 'RGB', 'BW' ]:
        CS = ColorScale(palet,-50.,250.)
        for x in [ -50+10.*i for i in range(31) ]:
            print x,": ",CS.color(x)
    
    CS = ColorScale('RGB',-50.,250.,0.)
    CL = ColorLegend(CS,5)
    print CL.limits
    for x in [ -45+10.*i for i in range(30) ]:
        print x,": ",CL.color(x)
    CL.underflowcolor = black
    CL.overflowcolor = white

    print CL.color(-55)
    print CL.color(255)
