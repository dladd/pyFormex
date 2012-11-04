# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""pyFormex C library module initialisation.

This tries to load the compiled libraries, and replaces those that failed
to load with the (slower) Python versions.
"""
from __future__ import print_function

__all__ = [ 'misc', 'nurbs', 'drawgl', 'accelerated' ]

misc = nurbs = drawgl = None
accelerated = []
required_drawgl_version = 1

    
import pyformex as pf


accelerate = gui = False
if pf.options:
    # testing for not False makes other values than T/F (like None) pass
    accelerate = pf.options.uselib is not False
    gui = pf.options.gui

if accelerate:

    try:
        import misc_ as misc
        pf.debug("Succesfully loaded the pyFormex compiled misc library",pf.DEBUG.LIB)
        accelerated.append(misc)
    except ImportError:
        pf.debug("Error while loading the pyFormex compiled misc library",pf.DEBUG.LIB)

    try:
        import nurbs_ as nurbs
        pf.debug("Succesfully loaded the pyFormex compiled nurbs library",pf.DEBUG.LIB)
        accelerated.append(nurbs)
    except ImportError:
        pf.debug("Error while loading the pyFormex compiled nurbs library",pf.DEBUG.LIB)

    if gui: 
        try:
            import drawgl_ as drawgl
            pf.debug("Succesfully loaded the pyFormex compiled drawgl library",pf.DEBUG.LIB)
            drawgl_version = drawgl.get_version()
            pf.debug("Drawing library version %s" % drawgl_version,pf.DEBUG.LIB)
            if not drawgl_version == required_drawgl_version:
                raise RuntimeError,"Incorrect acceleration library version (have %s, required %s)\nIf you are running pyFormex directly from sources, this might mean you have to run 'make lib' in the top directory of your pyFormex source tree.\nElse, this probably means pyFormex was not correctly installed."
            accelerated.append(drawgl)
        except ImportError:
            pf.debug("Error while loading the pyFormex compiled drawgl library",pf.DEBUG.LIB)

if misc is None:
    pf.debug("Using the (slower) Python misc functions",pf.DEBUG.LIB)
    import misc

if nurbs is None:
    pf.debug("Using the (slower) Python nurbs functions",pf.DEBUG.LIB)
    import nurbs

if gui and drawgl is None:
    pf.debug("Using the (slower) Python draw functions",pf.DEBUG.LIB)
    import drawgl


pf.debug("Accelerated: %s" % accelerated,pf.DEBUG.LIB|pf.DEBUG.INFO)

# End
