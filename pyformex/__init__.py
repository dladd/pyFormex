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
"""pyFormex core module initialisation.

This module initializes the pyFormex global variables and
defines a few essential functions.
"""

# This is the very first thing that is executed when starting pyFormex
# It is loaded even before main.

__version__ = "0.8.6-a1"
__revision__ = '2141:2143M'
Version = 'pyFormex %s' % __version__
Copyright = 'Copyright (C) 2004-2011 Benedict Verhegghe'
Url = 'http://pyformex.org'
Description = "pyFormex is a tool for generating, manipulating and transforming large geometrical models of 3D structures by sequences of mathematical transformations."
svnversion = False

# The GUI parts
app_started = False
interactive = False
app = None         # the Qapplication 
GUI = None         # the GUI QMainWindow
canvas = None      # the OpenGL Drawing widget controlled by the running script
#board = None       # the message board

# set start date/time
import time,datetime
StartTime = datetime.datetime.now()

# initialize some global variables used for communication between modules

options = None     # the options found on the command line
   
print_help = None  # the function to print(the pyformex help text (pyformex -h))

cfg = {}         # the current session configuration
prefcfg = None     # the preferenced configuration 
refcfg = None      # the reference configuration 
preffile = None    # the file where the preferenced configuration will be saved

PF = {}            # explicitely exported globals
#_PF_ = {}          # globals that will be offered to scripts
    
scriptName = None


# define last rescue versions of message, warning and debug
def message(s):
    print(s)

warning = message

def debug(s,lead="DEBUG",level=-1):
    """Print a debug message"""
    try: # to make sure that debug() can be used before options are set
        if options.debug < 0 or (options.debug % level > 0):
            raise
        pass
    except:
        print("%s: %s" % (lead,str(s)))


def debugt(s):
    """Print a debug message with timer"""
    debug(s,time.time())




### End
