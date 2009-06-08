# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
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
"""Menus for the pyFormex GUI."""

import pyformex
from pyformex.gui import *

from gettext import gettext as _
import fileMenu
import cameraMenu
import prefMenu
import viewportMenu
import toolbar
import help
import image
import draw
import script

save = NotImplemented
saveAs = NotImplemented

def editor():
    if pyformex.GUI.editor:
        print "Close editor"
        pyformex.GUI.closeEditor()
    else:
        print "Open editor"
        pyformex.GUI.showEditor()

 
def resetGUI():
    pyformex.GUI.setBusy(False)
    pyformex.GUI.actions['Play'].setEnabled(True)
    pyformex.GUI.actions['Step'].setEnabled(True)
    pyformex.GUI.actions['Continue'].setEnabled(False)
    pyformex.GUI.actions['Stop'].setEnabled(False)

  

def addViewport():
    """Add a new viewport."""
    n = len(pyformex.GUI.viewports.all)
    if n < 4:
        pyformex.GUI.viewports.addView(n/2,n%2)

def removeViewport():
    """Remove a new viewport."""
    n = len(pyformex.GUI.viewports.all)
    if n > 1:
        pyformex.GUI.viewports.removeView()


def viewportSettings():
    """Interactively set the viewport settings."""
    pass

            
# The menu actions can be simply function names instead of strings, if the
# functions have already been defined here.

def printwindow():
    pyformex.app.syncX()
    r = pyformex.GUI.frameGeometry()
    print "Qt4 geom(w,h,x,y): %s,%s,%s,%s" % (r.width(),r.height(),r.x(),r.y())
    print "According to xwininfo, (x,y) is %s,%s" % pyformex.GUI.XPos()


_geometry=None

def saveGeometry():
    global _geometry
    _geometry = pyformex.GUI.saveGeometry()

def restoreGeometry():
    pyformex.GUI.restoreGeometry(_geometry)


def moveCorrect():
    pyformex.GUI.move(*pyformex.GUI.XPos())


ActionMenuData = [
    (_('&Step'),draw.step),
    (_('&Continue'),draw.fforward), 
    (_('&Reset GUI'),resetGUI),
    (_('&Force Finish Script'),draw.force_finish),
    (_('&ListFormices'),script.printall),
    (_('&PrintGlobalNames'),script.printglobalnames),
    (_('&PrintGlobals'),script.printglobals),
    (_('&PrintConfig'),script.printconfig),
    (_('&Print Detected Software'),script.printdetected),
    (_('&PrintBbox'),draw.printbbox),
    (_('&Print Viewport Settings'),draw.printviewportsettings),
    (_('&Print Window Geometry'),printwindow),
    (_('&Correct the Qt4 Geometry'),moveCorrect),
    (_('&Save Geometry'),saveGeometry),
    (_('&Restore Geometry'),restoreGeometry),
#    (_('&Add Project to Status Bar'),gui.addProject),
    ]
             

MenuData = [
    (_('&File'),fileMenu.MenuData),
    (_('&Actions'),ActionMenuData),
    (_('&Help'),help.MenuData)
    ]


def createMenuData():
    """Returns the full data menu."""
    # Insert configurable menus
    if pyformex.cfg.get('gui/prefsmenu','True'):
        MenuData[1:1] = prefMenu.MenuData
    if pyformex.cfg.get('gui/viewportmenu','True'):
        MenuData[2:2] = viewportMenu.MenuData
    if pyformex.cfg.get('gui/cameramenu','True'):
        MenuData[3:3] = [(_('&Camera'),cameraMenu.MenuData)]
    
# End
