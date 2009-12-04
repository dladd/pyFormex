# $Id$
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
        print(x)
        print(y)
        print(len(x))
        print(len(y))
        maxlen = min(len(x),len(y))
        data = Gnuplot.Data(x[:maxlen],y[:maxlen],title=txt, with_='histeps') 
        g = Gnuplot.Gnuplot(persist=1)
        g.title('pyFormex histogram: %s' % txt)
        g.plot(data)
        
    elif plot2d_system == 'qwt':
        pass
        #from PyQt4.Qwt5.qplt import *


def createHistogram(data,cumulative=False):
    """Create a histogram from data

    """
    y,x = histogram(data)
    if cumulative:
        y = y.cumsum()
    return y,x

# End
