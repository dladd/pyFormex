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

    Each item is a tuple of two to four elements:
       Item Text, Action, [ ShortCut, Icon ].

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
        print "Adding item %s: %s" % (txt,val)
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
            menu.addAction(txt,val)



save = NotImplemented
saveAs = NotImplemented

def editor():
    if GD.gui.editor:
        print "Close editor"
        GD.gui.closeEditor()
    else:
        print "Open editor"
        GD.gui.showEditor()


MenuData = [
    ('&File',[
        ('&New','fileMenu.newFile'),
        ('&Open','fileMenu.openFile'),
        ('&Play','fileMenu.play'),
        ('&Edit','fileMenu.edit'),
#        ('&Save','save'),
#        ('Save &As','saveAs'),
        ('---','---'),
        ('Save &Image','fileMenu.saveImage'),
        ('Toggle &MultiSave','fileMenu.multiSave'),
        ('Toggle &AutoSave','fileMenu.autoSave'),
        ('---','---'),
        ('E&xit','GD.app.exit'), ]),
    ('&Settings',[
#        ('&Preferences','preferences'), 
        ('&Font','prefMenu.setFont'), 
        ('Font&Size','prefMenu.setFontSize'), 
        ('Toggle &Triade','draw.toggleTriade'), 
        ('&Drawwait Timeout','prefMenu.setDrawtimeout'), 
        ('&Background Color','prefMenu.setBGcolor'), 
        ('Line&Width','prefMenu.setLinewidth'), 
        ('&Canvas Size','prefMenu.setCanvasSize'), 
        ('&LocalAxes','prefMenu.setLocalAxes'),
        ('&GlobalAxes','prefMenu.setGlobalAxes'),
        ('&RotFactor','prefMenu.setRotFactor'),
        ('&PanFactor','prefMenu.setPanFactor'),
        ('&ZoomFactor','prefMenu.setZoomFactor'),
        ('&Wireframe','draw.wireframe'),
        ('&Flat','draw.flat'),
        ('&Smooth','draw.smooth'),
        ('&Render','prefMenu.setRender'),
        ('&Light0','prefMenu.setLight0'),
        ('&Light1','prefMenu.setLight1'),
        ('&Help','prefMenu.setHelp'),
        ('&Save Preferences','GD.savePreferences'), ]),
    ('&Camera',[
        ('&Zoom In','cameraMenu.zoomIn'), 
        ('&Zoom Out','cameraMenu.zoomOut'), 
        ('&Dolly In','cameraMenu.dollyIn'), 
        ('&Dolly Out','cameraMenu.dollyOut'), 
        ('Translate &Right','cameraMenu.transRight'), 
        ('Translate &Left','cameraMenu.transLeft'), 
        ('Translate &Up','cameraMenu.transUp'),
        ('Translate &Down','cameraMenu.transDown'),
        ('Rotate &Right','cameraMenu.rotRight'),
        ('Rotate &Left','cameraMenu.rotLeft'),
        ('Rotate &Up','cameraMenu.rotUp'),
        ('Rotate &Down','cameraMenu.rotDown'), 
        ('Rotate &ClockWise','cameraMenu.twistRight'),
        ('Rotate &CCW','cameraMenu.twistLeft'),  ]),
    ('&Actions',[
        ('&Step','draw.step'),
        ('&Continue','draw.fforward'), 
        ('&Clear','draw.clear'),
        ('&Redraw','draw.redraw'),
        ('&DrawSelected','draw.drawSelected'),
        ('&ListFormices','draw.printall'),
        ('&PrintBbox','draw.printbbox'),
        ('&PrintGlobals','draw.printglobals'),
        ('&PrintConfig','draw.printconfig'),  ]),
    ('&Help',[
##        ('&Help','help.help'),
        ('&Manual','help.manual'),
        ('&PyDoc','help.pydoc'),
        ('pyFormex &Website','help.website'),
        ('&Description','help.description'), 
        ('&About','help.about'), 
        ('&Warning','help.testwarning'), ]) ]


# End
