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

import pyformex as GD
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
    #reload(module)
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


def loadConfiguredPlugins(ok_plugins=None):
    if ok_plugins is None:
        ok_plugins = GD.cfg['gui/plugins']
    for n,p in plugin_menus:
        if p in ok_plugins:
            load(p)
        else:
            module = globals().get(p,None)
            if hasattr(module,'close_menu'):
                module.close_menu()

#################### EXPERIMENTAL STUFF BELOW !! ################
import odict

_registered_plugins = odict.ODict() 

def show_menu(name,before='help'):
    """Show the menu."""
    if not GD.GUI.menu.action(_menu):
        create_menu(before=before)

def close_menu(name):
    """Close the menu."""
    name.replace('_menu','')
    print "CLOSING MENU %s" % name 
    GD.GUI.menu.removeItem(name)


def register_plugin_menu(name,menudata,before=['help']):
    menudata.extend([
        ("---",None),
#        ("&Reload Menu",reload_menu,{'data':name}),
        ("&Close Menu",close_menu,{'data':name}),
        ])
    w = menu.Menu(name,items=menudata,parent=GD.GUI.menu,before=before[0])
    _registered_plugins[name] = w
    return w


def reload_menu(name):
    """Reload the menu."""
    before = GD.GUI.menu.nextitem(_menu)
    print "Menu %s was before %s" % (_menu,before)
    close_menu()
    import plugins
    plugins.refresh('draw2d')
    show_menu(before=before)
    setDrawOptions({'bbox':'last'})
    print GD.GUI.menu.actionList()

# End
