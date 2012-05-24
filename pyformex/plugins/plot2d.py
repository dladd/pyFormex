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

import pyformex as pf
from pyformex import utils
from arraytools import *


def showStepPlot(x,y,label='',title=None,plot2d_system=None):
    """Show a step plot of x,y data.

    """
    if title is None:
        title = 'pyFormex step plot: %s' % label
    maxlen = min(len(x),len(y))
    x = x[:maxlen]
    y = y[:maxlen]

    if plot2d_system is None:
        plot2d_system = pf.cfg['plot2d']
        if not utils.hasModule(plot2d_system):
            pf.error("I can not draw the requested plot. You need to install one of the supported plotting libraries (python-gnuplot or python-matplotlib) and set the appropriate preference in you pyformex configuration file or via the Settings->Settings Dialog menu item.")
            return

    if plot2d_system == 'gnuplot':
        import Gnuplot
        data = Gnuplot.Data(x,y,title=label, with_='steps') 
        g = Gnuplot.Gnuplot(persist=1)
        g.title(title)
        g.plot(data)
        
    elif plot2d_system == 'qwt':
        pass
        #from PyQt4.Qwt5.qplt import *

    elif plot2d_system == 'matplotlib':
        import matplotlib.pyplot as plt
        plt.step(x,y,where='post',label=label)
        plt.title(title)
        plt.legend()
        plt.show()


def showHistogram(x,y,label,cumulative=False,plot2d_system=None):
    """Show a histogram of x,y data.

    """
    if cumulative:
        fill = y[-1]
    else:
        fill = y[0]
    y = growAxis(y,len(x)-len(y),fill=fill)
    showStepPlot(x,y,label,plot2d_system=plot2d_system)


def createHistogram(data,cumulative=False,**kargs):
    """Create a histogram from data

    """
    y,x = histogram(data,**kargs)
    if cumulative:
        y = y.cumsum()
    return y,x

# End
