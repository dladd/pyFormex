#!/usr/bin/env python
# $Id $
"""Menus for the pyFormex GUI."""

from PyQt4 import QtCore, QtGui
import globaldata as GD
import fileMenu
import viewMenu
import help
import draw


def addMenuItems(menu, items=[]):
    """Add a list of items to a menu.

    Each item is a tuple of three strings : Type, Text, Value.
    Type specifies the menu item type and must be one of
    'Sep', 'Popup', 'Action', 'VAction'.
    
    'Sep' is a separator item. Its Text and Value fields are not used.
    
    For the other types, Text is the string that will be displayed in the
    menu. It can include a '&' character to flag the hotkey.
    
    'Popup' is a popup submenu item. Its value should be an item list,
    defining the menu to pop up when activated.
    
    'Action' is an active item. Its value is a python function that will be
    executed when the item is activated. It should be a global function
    without arguments.
    
    'VAction' is an active item where the value is a tuple of an function
    and an integer argument. When activated, the function will be executed
    with the specified argument. With 'VAction', you can bind multiple
    menu items to the same function.
    """
#
# Using ("SAction","text",foo) is almost equivalent to
# using ("Action","text","foo"), but the latter allows for functions
# that have not been defined yet in this scope!
#

    for key,txt,val in items:
        if key == "Sep":
            menu.addSeparator()
        elif key == "Popup":
            pop = QtGui.QMenu(txt,menu)
            addMenuItems(pop,val)
            menu.addMenu(pop)
        elif key == "Action":
            menu.addAction(txt,eval(val))
##        elif key == "VAction":
##            id = menu.insertItem(txt,eval(val[0]))
##            menu.setItemParameter(id,val[1])
##        elif key == "SAction":
##            menu.insertItem(txt,val)
        else:
            raise RuntimeError, "Invalid key %s in menu item"%key

MenuData = [
    ("Popup","&File",[
        ("Action","&New","fileMenu.newFile"),
        ("Action","&Open","fileMenu.openFile"),
        ("Action","&Play","fileMenu.play"),
        ("Action","&Edit","fileMenu.edit"),
#        ("Action","&Save","save"),
#        ("Action","Save &As","saveAs"),
        ("Sep",None,None),
        ("Action","Save &Image","fileMenu.saveImage"),
        ("Action","Toggle &MultiSave","fileMenu.multiSave"),
        ("Sep",None,None),
        ("Action","E&xit","GD.app.exit"), ]),
##    ("Popup","&Settings",[
###        ("Action","&Preferences","preferences"), 
##        ("Action","Toggle &Triade","draw.toggleTriade"), 
##        ("Action","&Drawwait Timeout","prefDrawtimeout"), 
##        ("Action","&Background Color","prefBGcolor"), 
##        ("Action","Line&Width","prefLinewidth"), 
##        ("Action","&Canvas Size","prefCanvasSize"), 
##        ("Action","&LocalAxes","localAxes"),
##        ("Action","&GlobalAxes","globalAxes"),
##        ("Action","&Wireframe","draw.wireframe"),
##        ("Action","&Flat","draw.flat"),
##        ("Action","&Smooth","draw.smooth"),
##        ("Action","&Render","prefRender"),
##        ("Action","&Light0","prefLight0"),
##        ("Action","&Light1","prefLight1"),
##        ("Action","&Help","prefHelp"),
##        ("Action","&Save Preferences","draw.savePreferences"), ]),
    ("Popup","&Camera",[
        ("Action","&Zoom In","viewMenu.zoomIn"), 
        ("Action","&Zoom Out","viewMenu.zoomOut"), 
        ("Action","&Dolly In","viewMenu.dollyIn"), 
        ("Action","&Dolly Out","viewMenu.dollyOut"), 
        ("Action","Pan &Right","viewMenu.transRight"), 
        ("Action","Pan &Left","viewMenu.transLeft"), 
        ("Action","Pan &Up","viewMenu.transUp"),
        ("Action","Pan &Down","viewMenu.transDown"),
        ("Action","Rotate &Right","viewMenu.rotRight"),
        ("Action","Rotate &Left","viewMenu.rotLeft"),
        ("Action","Rotate &Up","viewMenu.rotUp"),
        ("Action","Rotate &Down","viewMenu.rotDown"),  ]),
    ("Popup","&Actions",[
        ("Action","&Step","draw.step"),
        ("Action","&Continue","draw.fforward"), 
        ("Action","&Clear","draw.clear"),
        ("Action","&Redraw","draw.redraw"),
        ("Action","&DrawSelected","draw.drawSelected"),
        ("Action","&ListFormices","draw.printall"),
        ("Action","&PrintGlobals","draw.printglobals"),  ]),
    ("Popup","&Help",[
        ("Action","&Help","dohelp"),
        ("Action","&About","help.about"), 
        ("Action","&Warning","help.testwarning"), ]) ]

## We need this because the menufunctions take an argument and the
## help.help function has a default argument
##
def dohelp():
    help.help()


# End
