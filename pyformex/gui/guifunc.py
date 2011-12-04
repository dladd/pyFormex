# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
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

"""GUI support functions.

This module defines a collection of functions which are the equivalent of
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
