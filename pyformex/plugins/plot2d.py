# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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
"""plot2d.py

Generic 2D plotting functions for pyFormex.
"""

import pyformex
from pyformex import utils
from numpy import *

def showHistogram(x,y,txt,**options):
    """Show a histogram of x,y data.

    """
    plot2d_system = pyformex.cfg['gui/plot2d']

    if plot2d_system == 'gnuplot':
        if not utils.hasModule('gnuplot'):
            error("You do not have the Python Gnuplot module installed.\nI can not draw the requested plot.")
            return
        
        import Gnuplot
        maxlen = min(len(x),len(y))
        data = Gnuplot.Data(x[:maxlen],y[:maxlen],title=txt, with_='histeps') 
        g = Gnuplot.Gnuplot(persist=1)
        g.title('pyFormex histogram: %s' % txt)
        g.plot(data)
        
    elif plot2d_system == 'qwt':
        pass
        #from PyQt4.Qwt5.qplt import *


def createHistogram(data,cumulative=False,**kargs):
    """Create a histogram from data

    """
    y,x = histogram(data,**kargs)
    if cumulative:
        y = y.cumsum()
    return y,x

# End
