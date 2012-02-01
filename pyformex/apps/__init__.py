# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
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
"""pyFormex apps initialization.

This module contains the functions to detect and load the pyFormex
applications.
"""

import pyformex as pf
import utils

import os


def load(appname,refresh=False,autorun=True):
    """Load and run the named app"""
    import sys
    name = 'apps.'+appname
    try:
        __import__(name)
        app = sys.modules[name]
        if pf.GUI:
            pf.GUI.setcurfile(app)
        if refresh:
            reload(module)
        if autorun and hasattr(app,'run'):
            app.run()
        return app
    except:
        raise
    

def available_apps():
    appdir = os.path.dirname(__file__)
    files = utils.listTree(appdir,listdirs=False,excludedirs=['.*'],includefiles=['.*\.py$'])
    apps = [ os.path.basename(f) for f in files ]
    apps = [ os.path.splitext(f)[0] for f in apps if f[0] not in '._' ]
    return apps


def create_app_menu(parent=None,before=None): 	 
    from gui import menu
    loadmenu = menu.Menu('&Run Applications',parent=parent,before=before)
    loadactions = menu.ActionList(function=load,menu=loadmenu) 	 
    for name in _available_apps:
        descr = name.capitalize().replace('_',' ')
        loadactions.add(name,icon=None,text=descr)
        
    return loadactions


# Initialize this module

_available_apps = available_apps()

# End
