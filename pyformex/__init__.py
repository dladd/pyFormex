#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""pyformex.gui module initialisation.

"""

__version__ = "0.8.1-a4"
__revision__ = ""
Version = 'pyFormex %s' % __version__
Copyright = 'Copyright (C) 2004-2009 Benedict Verhegghe'
Url = 'http://pyformex.org'

# The GUI parts
app_started = False
app = None         # the Qapplication 
GUI = None         # the QMainWindow
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


# define last rescue versions of message, warning and debug
def message(s):
    print s

warning = message

def debug(s,lead="DEBUG"):
    """Print a debug message"""
    try: # to make sure that debug() can be used before options are set
        if options.debug:
            raise
        pass
    except:
        print "%s: %s" % (lead,str(s))


def debugt(s):
    """Print a debug message with timer"""
    debug(s,time.time())

def savePreferences():
    """Save the preferences.

    The name of the preferences file was set in GD.preffile.
    If a local preferences file was read, it will be saved there.
    Otherwise, it will be saved as the user preferences, possibly
    creating that file.
    If GD.preffile is None, preferences are not saved.
    """
    if preffile is None:
        return
    
    del cfg['__ref__']

    # Dangerous to set permanently!
    del cfg['input/timeout']
    
    debug("!!!Saving config:\n%s" % cfg)

    try:
        fil = file(preffile,'w')
        fil.write("%s" % cfg)
        fil.close()
        res = "Saved"
    except:
        res = "Could not save"
    debug("%s preferences to file %s" % (res,preffile))


### End
