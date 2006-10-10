#!/usr/bin/env python
# $Id$
"""Menus for the pyFormex GUI."""

from PyQt4 import QtCore, QtGui
import globaldata as GD
import fileMenu
import cameraMenu
import prefMenu
import help
import draw


###################### Actions #############################################
# Actions are just python functions, preferably without arguments
# Actions are often used as slots, which are triggered by signals,
#   e.g. by clicking a menu item or a tool button.
# Since signals have no arguments:
# Can we use python functions with arguments as actions ???
# - In menus we can have the menuitems send an integer id.
# - For other cases (like toolbuttons), we can subclass QAction and let it send
#   a signal with appropriate arguments 
#
# The above might no longer be correct for QT4! 

class FAction(QtGui.QAction):
    """A FAction is a QAction that calls a function when triggered.

    Most often QActions are created and connected automatically by some
    action insertion function of QT4. But for these cases where you want
    to create and connect the QAction before adding it, it is convenient
    to have this single line option.
    """
    
    def __init__(self,name,func,icon=None,tip=None,key=None):
        """Create a new FAction connected to method func.

        If the FAction is used in a menu, a name and func is sufficient.
        For use in a toolbar, you will probably want to specify an icon.
        Additionally, you can set a tooltip and shortcut key.
        When the action is triggered, the func is called.
        """
        QtGui.QAction.__init__(self,name,None)
        if icon:
            self.setIcon(icon)
        if tip:
            self.setToolTip(tip)
        self.connect(self,QtCore.SIGNAL("triggered()"),func)


class DAction(QtGui.QAction):
    """A DAction is a QAction that emits a signal with a string parameter.

    When triggered, this action sends a signal 'Clicked' with a custom
    string as parameter. The connected slot can then act depending on this
    parameter.
    """
    
    def __init__(self,name,icon=None,data=None):
        """Create a new DAction with name, icon and string data.

        If the DAction is used in a menu, a name is sufficient. For use
        in a toolbar, you will probably want to specify an icon.
        When the action is triggered, the data is sent as a parameter to
        the SLOT function connected with the 'Clicked' signal.
        If no data is specified, the name is used as data. 
        
        See the views.py module for an example.
        """
        QtGui.QAction.__init__(self,name,None)
        if icon:
            self.setIcon(icon)
        if not data:
            data = name
        self.setData(QtCore.QVariant(data))
        self.connect(self,QtCore.SIGNAL("triggered()"),self.activated)
        
    def activated(self):
        print "Clicked %s" % str(self.data().toString())
        self.emit(QtCore.SIGNAL("Clicked"), str(self.data().toString()))


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
    ("Popup","&Settings",[
#        ("Action","&Preferences","preferences"), 
        ("Action","&Font","prefMenu.setFont"), 
        ("Action","Font&Size","prefMenu.setFontSize"), 
        ("Action","Toggle &Triade","draw.toggleTriade"), 
        ("Action","&Drawwait Timeout","prefMenu.setDrawtimeout"), 
        ("Action","&Background Color","prefMenu.setBGcolor"), 
        ("Action","Line&Width","prefMenu.setLinewidth"), 
        ("Action","&Canvas Size","prefMenu.setCanvasSize"), 
        ("Action","&LocalAxes","prefMenu.setLocalAxes"),
        ("Action","&GlobalAxes","prefMenu.setGlobalAxes"),
        ("Action","&Wireframe","draw.wireframe"),
        ("Action","&Flat","draw.flat"),
        ("Action","&Smooth","draw.smooth"),
        ("Action","&Render","prefMenu.setRender"),
        ("Action","&Light0","prefMenu.setLight0"),
        ("Action","&Light1","prefMenu.setLight1"),
        ("Action","&Help","prefMenu.setHelp"),
        ("Action","&Save Preferences","GD.savePreferences"), ]),
    ("Popup","&Camera",[
        ("Action","&Zoom In","cameraMenu.zoomIn"), 
        ("Action","&Zoom Out","cameraMenu.zoomOut"), 
        ("Action","&Dolly In","cameraMenu.dollyIn"), 
        ("Action","&Dolly Out","cameraMenu.dollyOut"), 
        ("Action","Translate &Right","cameraMenu.transRight"), 
        ("Action","Translate &Left","cameraMenu.transLeft"), 
        ("Action","Translate &Up","cameraMenu.transUp"),
        ("Action","Translate &Down","cameraMenu.transDown"),
        ("Action","Rotate &Right","cameraMenu.rotRight"),
        ("Action","Rotate &Left","cameraMenu.rotLeft"),
        ("Action","Rotate &Up","cameraMenu.rotUp"),
        ("Action","Rotate &Down","cameraMenu.rotDown"), 
        ("Action","Rotate &ClockWise","cameraMenu.twistRight"),
        ("Action","Rotate &CCW","cameraMenu.twistLeft"),  ]),
    ("Popup","&Actions",[
        ("Action","&Step","draw.step"),
        ("Action","&Continue","draw.fforward"), 
        ("Action","&Clear","draw.clear"),
        ("Action","&Redraw","draw.redraw"),
        ("Action","&DrawSelected","draw.drawSelected"),
        ("Action","&ListFormices","draw.printall"),
        ("Action","&PrintGlobals","draw.printglobals"),  ]),
    ("Popup","&Help",[
##        ("Action","&Help","help.help"),
        ("Action","&Manual","help.manual"),
        ("Action","&PyDoc","help.pydoc"),
        ("Action","pyFormex &Website","help.website"),
        ("Action","&Description","help.description"), 
        ("Action","&About","help.about"), 
        ("Action","&Warning","help.testwarning"), ]) ]



#####################################################################
# Opening, Playing and Saving pyformex scripts

save = NotImplemented
saveAs = NotImplemented

def editor():
    if GD.gui.editor:
        print "Close editor"
        GD.gui.closeEditor()
    else:
        print "Open editor"
        GD.gui.showEditor()


# End
