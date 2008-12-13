#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""pyFormex GUI module initialisation.

Currently, this does nothing. The file should be kept though, because it is
needed to flag this directory as a Python package.
"""

__version__ = "0.7.3-a7"
__revision__ = "$Rev$"
Version = 'pyFormex %s' % __version__
Copyright = 'Copyright (C) 2004-2007 Benedict Verhegghe'


# The GUI parts
app_started = False
app = None         # the Qapplication 
gui = None         # the QMainWindow
canvas = None      # the OpenGL Drawing widget
board = None       # the message board

# set start date/time
import time,datetime
StartTime = datetime.datetime.now()

# initialize some global variables used for communication between modules

options = None     # the options found on the command line
print_help = None  # the function to print the pyformex help text (pyformex -h)

cfg = None         # the current configuration
refcfg = None      # the reference configuration 
preffile = None    # the file where current configuration will be saved

PF = {}            # globals that will be offered to scripts
    
scriptName = None

# Output status of the draw.askItems() function
dialog_timeout = False
dialog_accepted = False


# define last rescue versions of message, warning and debug
def message(s):
    print s

warning = message

## def warning(s):
##     if gui:
##         from gui import draw
##         draw.warning(s)
##     else:
##         import script
##         script.warning(s)

def debug(s):
    if options.debug:
        print "DEBUG: %s" % str(s)

def debugt(s):
    if options.debug:
        print "%.3f: %s" % (time.time(),str(s))


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


### End
