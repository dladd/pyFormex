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

__all__ = [ 'drawgl', 'misc', 'accelerated' ]

drawgl = misc = None
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
        import misc
        debug("Succesfully loaded the pyFormex compiled misc library")
        accelerated.append(misc)
    except ImportError:
        debug("Error while loading the pyFormex compiled misc library")

    if gui: 
        try:
            import drawgl
            debug("Succesfully loaded the pyFormex compiled draw library")
            drawgl_version = drawgl.get_version()
            debug("Drawing library version %s" % drawgl_version)
            if not drawgl_version == required_drawgl_version:
                raise RuntimeError,"Incorrect acceleration library version (have %s, required %s)\nIf you are running pyFormex directly from sources, this might mean you have to run 'make lib' in the top directory of your pyFormex source tree.\nElse, this probably means pyFormex was not correctly installed."
            accelerated.append(drawgl)
        except ImportError:
            debug("Error while loading the pyFormex compiled draw library")

if misc is None:
    debug("Using the (slower) Python misc functions")
    import pyformex.misc as misc

if gui and drawgl is None:
    debug("Using the (slower) Python draw functions")
    import pyformex.gui.drawgl as drawgl


# End
