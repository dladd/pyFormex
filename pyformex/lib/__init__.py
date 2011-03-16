#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
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


# End
