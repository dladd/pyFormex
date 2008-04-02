#!/usr/bin/env python
# $Id$
# $URL$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Global data for pyFormex."""

# Set pyformex version
__version__ = "0.7-a2"
__revision__ = "$Rev$"
Version = 'pyFormex %s' % __version__
Copyright = 'Copyright (C) 2004-2007 Benedict Verhegghe'

# versions of detected modules/external commands
version = {'pyformex':__version__}
external = {}

# set start date
import datetime
date = datetime.datetime.today()

# initialize some global variables used for communication between modules

options = None     # the options found on the command line
print_help = None  # the function to print the pyformex help text (pyformex -h)

cfg = None         # the current configuration
refcfg = None      # the reference configuration 
preffile = None    # the file where current configuration will be saved

# Qt4 GUI parts
app_started = False
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

def warning(s):
    if gui:
        from gui import draw
        draw.warning(s)
    else:
        import script
        script.warning(s)
    

def debug(s):
    #if hasattr(options,'debug'):
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
    If GD.preffile is None, preferences are not saved.
    """
    f = preffile
    if not f:
        return
    
    del cfg['__ref__']

    # Dangerous to set permanently!
    del cfg['input/timeout']
    
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
