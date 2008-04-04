#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.7 Release Fri Apr  4 18:41:11 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

"""Calpy interface for pyFormex.

Currently this is only used to detect the installation of calpy and
add the path of the calpy module to sys.path.

Importing this module will automatically check the availabilty of calpy
and set the sys.path accordingly.
"""

import globaldata as GD

import utils
import sys,os

calpy_path = None  # never tried to detect

def detect(trypaths=None):
    """Check if we have calpy and if so, add its path to sys.path."""

    global calpy_path
    
    calpy = utils.hasExternal('calpy')
    print "AAAA",calpy
    if not calpy:
        return
    
    GD.message("You have calpy version %s" % calpy)
    path = ''
    calpy = calpy.split('-')[0]  # trim the version trailer
    if utils.checkVersion('calpy','0.3.4-rev3',external=True) >= 0:
        sta,out = utils.runCommand('calpy --whereami')
        if not sta:
            path = out
            GD.debug("I found calpy in %s" % path)
    if not path:
        if trypaths is None:
            trypaths = [ '/usr/local/lib', '/usr/local' ]
        for p in trypaths:
            path = '%s/calpy-%s' % (p,calpy)
            if os.path.exists(path):
                GD.debug('path exists: %s' % path)
                break
            else:
                GD.debug('path does not exist: %s' % path)
                path = ''
    if path:
        #path += '/calpy'
        GD.message("I found calpy in '%s'" % path)
        sys.path.append(path)

    calpy_path = path


def check(trypaths=None):
    """Warn the user that calpy was not found."""
    if calpy_path is None:
        detect(trypaths)

    import calpy
    
    if not utils.hasModule('calpy',check=True):
        GD.warning("sys.path=%s\nSorry, I can not run this example, because you do not have calpy installed (at least not in a place where I can find it)." % sys.path)
        exit()


if __name__ == "__main__":
    print __doc__
else:
    GD.debug("Loading plugin %s" % __file__)
    check()


### End
