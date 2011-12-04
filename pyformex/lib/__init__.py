# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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
"""pyFormex C library module initialisation.

This tries to load the compiled libraries, and replaces those that failed
to load with the (slower) Python versions.
"""

__all__ = [ 'misc', 'nurbs', 'drawgl', 'accelerated' ]

misc = nurbs = drawgl = None
accelerated = []
required_drawgl_version = 1

    
from pyformex import options,debug


accelerate = gui = False
if options:
    # testing for not False makes other values than T/F (like None) pass
    accelerate = options.uselib is not False
    gui = options.gui

if accelerate:

    try:
        import misc_ as misc
        debug("Succesfully loaded the pyFormex compiled misc library")
        accelerated.append(misc)
    except ImportError:
        debug("Error while loading the pyFormex compiled misc library")

    try:
        import nurbs_ as nurbs
        debug("Succesfully loaded the pyFormex compiled nurbs library")
        accelerated.append(nurbs)
    except ImportError:
        debug("Error while loading the pyFormex compiled nurbs library")

    if gui: 
        try:
            import drawgl_ as drawgl
            debug("Succesfully loaded the pyFormex compiled drawgl library")
            drawgl_version = drawgl.get_version()
            debug("Drawing library version %s" % drawgl_version)
            if not drawgl_version == required_drawgl_version:
                raise RuntimeError,"Incorrect acceleration library version (have %s, required %s)\nIf you are running pyFormex directly from sources, this might mean you have to run 'make lib' in the top directory of your pyFormex source tree.\nElse, this probably means pyFormex was not correctly installed."
            accelerated.append(drawgl)
        except ImportError:
            debug("Error while loading the pyFormex compiled drawgl library")

if misc is None:
    debug("Using the (slower) Python misc functions")
    import misc

if nurbs is None:
    debug("Using the (slower) Python nurbs functions")
    import nurbs

if gui and drawgl is None:
    debug("Using the (slower) Python draw functions")
    import drawgl


debug("Accelerated: %s" % accelerated)

# End
