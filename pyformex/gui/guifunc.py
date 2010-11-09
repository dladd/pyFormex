# $Id$
##

"""GUI support functions.

This module defines a collection of function which are the equivalent of
functions defined in the draw module, but are executed in the viewport with
the current GUI focus, instead of the script viewport.
"""

import pyformex as pf
import draw

######## decorator function #############

def viewport_function(func):
    """Perform a function on the current GUI viewport.

    This is a decorator function executing a function on the
    current GUI viewport instead of on the current script viewport.
    """
    draw_func = getattr(draw,func.__name__)
    def newf(*args,**kargs):
        """Performs the draw.func on the current GUI viewport"""
        #print "SAVED script canvas %s" % pf.canvas
        save = pf.canvas
        pf.canvas = pf.GUI.viewports.current
        #print "SET script canvas %s" % pf.canvas
        draw_func(*args,**kargs)
        pf.canvas = save
        #print "RESTORED script canvas %s" % pf.canvas

    newf.__name__ = func.__name__
    newf.__doc__ = draw_func.__doc__
    return newf

@viewport_function
def renderMode(*args,**kargs):
        pass

@viewport_function
def zoomAll(*args,**kargs):
        pass

def inGUIVP(func,*args,**kargs):
    """Execute a draw function in the current GUI viewport."""
    draw_func = getattr(draw,func.__name__)
    #print "inGUI SAVED script canvas %s" % pf.canvas
    save = pf.canvas
    pf.canvas = pf.GUI.viewports.current
    #print "inGUI SET script canvas %s" % pf.canvas
    draw_func(*args,**kargs)
    pf.canvas = save
    #print "inGUI RESTORED script canvas %s" % pf.canvas
        

# End
