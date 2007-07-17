#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Tue Jul 10 13:43:12 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Global data for pyFormex."""

# Set pyformex version
__version__ = "0.5-alpha"
Version = 'pyFormex version %s' % __version__
Copyright = 'Copyright (C) 2007 Benedict Verhegghe'

# versions of detected modules/external commands
version = {'pyformex':__version__}
external = {}

# initialize some global variables used for communication between modules

options = None     # the options found on the command line
print_help = None  # the function to print the pyformex help text (pyformex -h)

cfg = None         # the current configuration
refcfg = None      # the reference configuration 
preffile = None    # the file where current configuration will be saved

# Qt4 GUI parts
app = None         # the Qapplication 
gui = None         # the QMainWindow
canvas = None      # the OpenGL Drawing widget
board = None       # the message board

PF = {}            # globals that will be offered to scripts

# These image format should probably be moved to image.py
image_formats_qt = []
image_formats_qtr = []
image_formats_gl2ps = []
image_formats_fromeps = []

#multisave = False


# define last rescue versions of message and debug
def message(s):
    print s

def debug(s):
    if hasattr(options,'debug'):
        if options.debug:
            print "DEBUG: %s" % str(s)


# we couldn't put these in gui.py because we can't import gui in other modules

canPlay = False
scriptName = None

def savePreferences():
    """Save the preferences.

    The name of the preferences file was set in GD.preffile.
    If a local preferences file was read, it will be saved there.
    Otherwise, it will be saved as the user preferences, possibly
    creating that file.
    """
    f = preffile
    del cfg['__ref__']
    debug("!!!Saving config:\n%s" % cfg)

    try:
        fil = file(f,'w')
        fil.write("%s" % cfg)
        fil.close()
        res = "Saved"
    except:
        res = "Could not save"
    debug("%s preferences to file %s" % (res,f))

# End
