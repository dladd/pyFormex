#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.7.3 Release Tue Dec 30 20:45:35 2008
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

__version__ = "0.7.4-a1"
__revision__ = "$Rev$"
Version = 'pyFormex %s' % __version__
Copyright = 'Copyright (C) 2004-2008 Benedict Verhegghe'


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
