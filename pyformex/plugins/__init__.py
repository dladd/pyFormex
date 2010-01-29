#!/usr/bin/env python
# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
"""pyFormex plugins initialisation.

Currently, this does nothing. The file should be kept though, because it is
needed to flag this directory as a Python module.
"""

from types import ModuleType
from gettext import gettext as _


def load(plugin):
    """Load the named plugin"""
    # imports are placed here to defer loading until possible
    import formex_menu
    import surface_menu
    import mesh_menu
    import tools_menu
    import draw2d
    import jobs_menu
    import postproc_menu
    module = globals().get(plugin,None)
    if type(module) is ModuleType and hasattr(module,'show_menu'):
        module.show_menu()


def refresh(plugin):
    """Reload the named plugin"""
    module = globals().get(plugin,None)
    reload(module)


def refresh_menu(plugin):
    """Reload the named plugin"""
    module = globals().get(plugin,None)
    reload(module)


def loaded_modules():
    d = [ k for k in globals() if type(globals()[k]) is ModuleType ]
    d.sort()
    return d

plugin_menus = [
    (_('Formex menu'),'formex_menu'),
    (_('Surface menu'),'surface_menu'),
    (_('Mesh menu'),'mesh_menu'),
    (_('Tools menu'),'tools_menu'),
    (_('Draw menu'),'draw2d'),
    (_('Jobs menu'),'jobs_menu'),
    (_('Postproc menu'),'postproc_menu'),
    ]

def create_plugin_menus(parent=None,before=None):
    plugin_text =  [ k for k,v in plugin_menus ]
    plugin_names = [ v for k,v in plugin_menus ]

    from gui import menu
    
    loadmenu = menu.Menu('&Load plugins',parent=parent,before=before)
    reloadmenu = menu.Menu('&Reload plugins',parent=parent,before=before)

    loadactions = menu.ActionList(function=load,menu=loadmenu)
    reloadactions = menu.ActionList(function=refresh,menu=reloadmenu)

    for text,name in plugin_menus:
        loadactions.add(name,icon=None,text=text)
        reloadactions.add(name,icon=None,text=text)

    return loadactions,reloadactions

# End
