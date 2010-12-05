#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""Calpy interface for pyFormex.

Currently this is only used to detect the installation of calpy and
add the path of the calpy module to sys.path.

Importing this module will automatically check the availabilty of calpy
and set the sys.path accordingly.
"""

import pyformex as pf

import utils
import sys,os

calpy_path = None  # never tried to detect

def detect(trypaths=None):
    """Check if we have calpy and if so, add its path to sys.path."""

    global calpy_path
    
    calpy = utils.checkExternal('calpy')
    if not calpy:
        return
    
    pf.message("You have calpy version %s" % calpy)
    path = ''
    calpy = calpy.split('-')[0]  # trim the version trailer
    if utils.checkVersion('calpy','0.3.4-rev3',external=True) >= 0:
        sta,out = utils.runCommand('calpy --whereami')
        if not sta:
            path = out
            pf.debug("I found calpy in %s" % path)
    if not path:
        if trypaths is None:
            trypaths = [ '/usr/local/lib', '/usr/local' ]
        for p in trypaths:
            path = '%s/calpy-%s' % (p,calpy)
            if os.path.exists(path):
                pf.debug('path exists: %s' % path)
                break
            else:
                pf.debug('path does not exist: %s' % path)
                path = ''
    if path:
        #path += '/calpy'
        pf.message("I found calpy in '%s'" % path)
        sys.path.append(path)

    calpy_path = path


def check(trypaths=None):
    """Warn the user that calpy was not found."""
    if calpy_path is None:
        detect(trypaths)

    try:
        import calpy
    except ImportError:
        pass
    
    if utils.hasModule('calpy',check=True):
        return True
    else:
        pf.warning("sys.path=%s\nSorry, I can not run this example, because you do not have calpy installed (at least not in a place where I can find it)." % sys.path)
        return False
 


if __name__ == "__main__":
    print(__doc__)
else:
    pf.debug("Loading plugin %s" % __file__)
    check()


### End
