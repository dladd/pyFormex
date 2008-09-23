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
from plugins import surface_menu,formex_menu,tools_menu,postproc_menu


save = NotImplemented
saveAs = NotImplemented

def editor():
    if pyformex.gui.editor:
        print "Close editor"
        pyformex.gui.closeEditor()
    else:
        print "Open editor"
        pyformex.gui.showEditor()

 
def resetGUI():
    pyformex.gui.setBusy(False)
    pyformex.gui.actions['Play'].setEnabled(True)
    pyformex.gui.actions['Step'].setEnabled(True)
    pyformex.gui.actions['Continue'].setEnabled(False)
    pyformex.gui.actions['Stop'].setEnabled(False)

  

def addViewport():
    """Add a new viewport."""
    n = len(pyformex.gui.viewports.all)
    if n < 4:
        pyformex.gui.viewports.addView(n/2,n%2)

def removeViewport():
    """Remove a new viewport."""
    n = len(pyformex.gui.viewports.all)
    if n > 1:
        pyformex.gui.viewports.removeView()


def viewportSettings():
    """Interactively set the viewport settings."""
    pass


def setOptions():
    options = ['test','debug','uselib','safelib','fastencode','fastfuse']
    items = [ (o,getattr(pyformex.options,o)) for o in options ]
    res = draw.askItems(items)
    if res:
        for o in options:
            setattr(pyformex.options,o,res[o])
            
# The menu actions can be simply function names instead of strings, if the
# functions have already been defined here.
#


FileMenuData = [
    (_('&Start new project'),fileMenu.createProject),
    ('&Open existing project',fileMenu.openProject),
    ('&Save project',fileMenu.saveProject),
    ('&Save and close project',fileMenu.closeProject),
    ('---',None),
    (_('&Create new script'),fileMenu.createScript),
    (_('&Open existing script'),fileMenu.openScript),
    (_('&Play script'),draw.play),
    (_('&Edit script'),fileMenu.editScript),
    (_('&Change workdir'),draw.askDirname),
    (_('---1'),None),
    (_('&Save Image'),fileMenu.saveImage),
    (_('Start &MultiSave'),fileMenu.startMultiSave),
    (_('Save &Next Image'),image.saveNext),
    (_('Create &Movie'),image.createMovie),
    (_('&Stop MultiSave'),fileMenu.stopMultiSave),
    (_('&Save Icon'),fileMenu.saveIcon),
    (_('---2'),None),
    (_('Load &Plugins'),[
        (_('Surface menu'),surface_menu.show_menu),
        (_('Formex menu'),formex_menu.show_menu),
        (_('Tools menu'),tools_menu.show_menu),
        (_('Postproc menu'),postproc_menu.show_menu),
        ]),
    (_('&Options'),setOptions),
    (_('---3'),None),
    (_('E&xit'),draw.closeGui),
]

def printwindow():
    pyformex.app.syncX()
    r = pyformex.gui.frameGeometry()
    print "Qt4 geom(w,h,x,y): %s,%s,%s,%s" % (r.width(),r.height(),r.x(),r.y())
    print "According to xwininfo, (x,y) is %s,%s" % pyformex.gui.XPos()


_geometry=None

def saveGeometry():
    global _geometry
    _geometry = pyformex.gui.saveGeometry()

def restoreGeometry():
    pyformex.gui.restoreGeometry(_geometry)


def moveCorrect():
    pyformex.gui.move(*pyformex.gui.XPos())


ActionMenuData = [
    (_('&Step'),draw.step),
    (_('&Continue'),draw.fforward), 
    (_('&Reset GUI'),resetGUI),
    (_('&Force Finish Script'),draw.force_finish),
    (_('&ListFormices'),script.printall),
    (_('&PrintGlobalNames'),script.printglobalnames),
    (_('&PrintGlobals'),script.printglobals),
    (_('&PrintConfig'),script.printconfig),
    (_('&Print Detected Software'),script.printDetected),
    (_('&PrintBbox'),draw.printbbox),
    (_('&Print Viewport Settings'),draw.printviewportsettings),
    (_('&Print Window Geometry'),printwindow),
    (_('&Correct the Qt4 Geometry'),moveCorrect),
    (_('&Save Geometry'),saveGeometry),
    (_('&Restore Geometry'),restoreGeometry),
#    (_('&Add Project to Status Bar'),gui.addProject),
    ]

CameraMenuData = [
    (_('&LocalAxes'),draw.setLocalAxes),
    (_('&GlobalAxes'),draw.setGlobalAxes),
    (_('&Projection'),toolbar.setProjection),
    (_('&Perspective'),toolbar.setPerspective),
    (_('&Zoom All'),draw.zoomAll), 
    (_('&Zoom In'),cameraMenu.zoomIn), 
    (_('&Zoom Out'),cameraMenu.zoomOut), 
    (_('&Dolly In'),cameraMenu.dollyIn), 
    (_('&Dolly Out'),cameraMenu.dollyOut), 
    (_('&Translate'),[
        (_('Translate &Right'),cameraMenu.transRight), 
        (_('Translate &Left'),cameraMenu.transLeft), 
        (_('Translate &Up'),cameraMenu.transUp),
        (_('Translate &Down'),cameraMenu.transDown),
        ]),
    (_('&Rotate'),[
        (_('Rotate &Right'),cameraMenu.rotRight),
        (_('Rotate &Left'),cameraMenu.rotLeft),
        (_('Rotate &Up'),cameraMenu.rotUp),
        (_('Rotate &Down'),cameraMenu.rotDown), 
        (_('Rotate &ClockWise'),cameraMenu.twistRight),
        (_('Rotate &CCW'),cameraMenu.twistLeft),
        ]),
    ]
             

MenuData = [
    (_('&File'),FileMenuData),
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
        MenuData[3:3] = [(_('&Camera'),CameraMenuData)]
    
# End
