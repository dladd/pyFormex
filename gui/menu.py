#!/usr/bin/env python
# $Id$
"""Menus for the pyFormex GUI."""

import os
from gettext import gettext as _
from PyQt4 import QtCore, QtGui
import globaldata as GD
import fileMenu
import cameraMenu
import prefMenu
import help
import draw
from plugins import stl_menu


def addMenuItems(menu, items=[]):
    """Add a list of items to a menu.

    Each item is a tuple of two to five elements:
       Item Text, Action, [ ShortCut, Icon, ToolTip ].

    Item text is the text that will be displayed in the menu. An optional '&'
    may be used to flag the next character as the shortcut key. The '&' will
    be stripped off before displaying the text.

    Action can be any of the following:
      - a Python function or instance method : it will be called when the
        item is selected,
      - a list of Menu Items: a popup Menu will be created that will appear
        when the item is selected,
      - a string '---' : this will create a separator item with no action,
      - a string that will evaluate to one of the above.
    
    ShortCut is an optional key combination to select the item.
    """
    for item in items:
        txt,val = item[:2]
        if val == '---':
            menu.addSeparator()
            continue
        if type(val) == str:
            val = eval(val)
        if isinstance(val, list):
            pop = QtGui.QMenu(txt,menu)
            addMenuItems(pop,val)
            menu.addMenu(pop)
        else:
            a = menu.addAction(txt,val)
            if len(item) >= 5:
                a.setToolTip(item[4])



save = NotImplemented
saveAs = NotImplemented

def editor():
    if GD.gui.editor:
        print "Close editor"
        GD.gui.closeEditor()
    else:
        print "Open editor"
        GD.gui.showEditor()


def formex_menu():
    draw.play(os.path.join(GD.cfg['pyformexdir'],'plugins','formex_menu.py'))

def addViewport():
    n = len(GD.canvas.views)
    if n < 4:
        GD.canvas.addView(n/2,n%2)
    

# The menu actions can be simply function names instead of strings, if the
# functions have already been defined here.
#

MenuData = [
    (_('&File'),[
        (_('&New'),fileMenu.newFile),
        (_('&Open'),fileMenu.openFile),
        (_('&Play'),fileMenu.play),
        (_('&Edit'),fileMenu.edit),
#        (_('&Save'),'save'),
#        (_('Save &As'),'saveAs'),
        (_('---'),'---'),
        (_('Save &Image'),'fileMenu.saveImage'),
        (_('Start &MultiSave'),'fileMenu.startMultiSave'),
        (_('Stop &MultiSave'),'fileMenu.stopMultiSave'),
        (_('Save &Next Image'),'draw.saveNext'),
        (_('---'),'---'),
        (_('Load &Plugins'),[
            (_('STL menu'),stl_menu.show_menu),
            (_('Formex menu'),formex_menu),
            ]),
        (_('---'),'---'),
        (_('E&xit'),'GD.app.exit'),
        ]),
    (_('&Settings'),[
        (_('&Appearance'),'prefMenu.setAppearance'), 
        (_('&Font'),'prefMenu.setFont'), 
        (_('Toggle &Triade'),'draw.toggleTriade'), 
        (_('&Drawwait Timeout'),'prefMenu.setDrawtimeout'), 
        (_('&Background Color'),'prefMenu.setBGcolor'), 
        (_('Line&Width'),'prefMenu.setLinewidth'), 
        (_('&Canvas Size'),'prefMenu.setCanvasSize'), 
        (_('&RotFactor'),'prefMenu.setRotFactor'),
        (_('&PanFactor'),'prefMenu.setPanFactor'),
        (_('&ZoomFactor'),'prefMenu.setZoomFactor'),
        (_('&Wireframe'),'draw.wireframe'),
        (_('&Flat'),'draw.flat'),
        (_('&Smooth'),'draw.smooth'),
        (_('&Render'),'prefMenu.setRender'),
        (_('&Light0'),'prefMenu.setLight0'),
        (_('&Light1'),'prefMenu.setLight1'),
        (_('&Commands'),'prefMenu.setCommands'),
        (_('&Help'),'prefMenu.setHelp'),
        (_('&Save Preferences'),'GD.savePreferences'), ]),
    (_('&Camera'),[
        (_('&LocalAxes'),'draw.setLocalAxes'),
        (_('&GlobalAxes'),'draw.setGlobalAxes'),
        (_('&Projection'),'cameraMenu.setProjection'),
        (_('&Perspective'),'cameraMenu.setPerspective'),
        (_('&Zoom All'),'draw.zoomAll'), 
        (_('&Zoom In'),'cameraMenu.zoomIn'), 
        (_('&Zoom Out'),'cameraMenu.zoomOut'), 
        (_('&Dolly In'),'cameraMenu.dollyIn'), 
        (_('&Dolly Out'),'cameraMenu.dollyOut'), 
        (_('Translate &Right'),'cameraMenu.transRight'), 
        (_('Translate &Left'),'cameraMenu.transLeft'), 
        (_('Translate &Up'),'cameraMenu.transUp'),
        (_('Translate &Down'),'cameraMenu.transDown'),
        (_('Rotate &Right'),'cameraMenu.rotRight'),
        (_('Rotate &Left'),'cameraMenu.rotLeft'),
        (_('Rotate &Up'),'cameraMenu.rotUp'),
        (_('Rotate &Down'),'cameraMenu.rotDown'), 
        (_('Rotate &ClockWise'),'cameraMenu.twistRight'),
        (_('Rotate &CCW'),'cameraMenu.twistLeft'),
        ]),
    (_('&Actions'),[
        (_('&Step'),'draw.step'),
        (_('&Continue'),'draw.fforward'), 
        (_('&Clear'),'draw.clear'),
        (_('&Redraw'),'draw.redraw'),
        (_('&DrawSelected'),'draw.drawSelected'),
        (_('&ListFormices'),'draw.printall'),
        (_('&PrintBbox'),'draw.printbbox'),
        (_('&PrintGlobals'),'draw.printglobals'),
        (_('&PrintConfig'),'draw.printconfig'),
        ]),
    (_('&Help'),[
##        (_('&Help'),'help.help'),
        (_('&Manual'),help.manual),
        (_('&PyDoc'),help.pydoc,None,None,'Autogenerated documentation from the pyFormex sources'),
        (_('pyFormex &Website'),help.website),
        (_('&Description'),help.description), 
        (_('&About'),help.about), 
        (_('&Warning'),help.testwarning),
        ])
    ]


ViewportMenuData = [
    (_('&Viewport'),[
        (_('&Add new viewport'),addViewport), 
        ]),
    ]


def insertViewportMenu():
    """Insert the viewport menu data into the global menu data."""
    MenuData[2:2] = ViewportMenuData

    
# End
