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

def Windows():
    GD.app.setStyle('Windows')

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
        ('Start &MultiSave','fileMenu.startMultiSave'),
        ('Stop &MultiSave','fileMenu.stopMultiSave'),
        ('Save &Next Image','draw.saveNext'),
        ('---','---'),
        ('E&xit','GD.app.exit'), ]),
    ('&Settings',[
        ('&Appearance','prefMenu.setAppearance'), 
        ('&Font','prefMenu.setFont'), 
        ('Toggle &Triade','draw.toggleTriade'), 
        ('&Drawwait Timeout','prefMenu.setDrawtimeout'), 
        ('&Background Color','prefMenu.setBGcolor'), 
        ('Line&Width','prefMenu.setLinewidth'), 
        ('&Canvas Size','prefMenu.setCanvasSize'), 
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
        ('&LocalAxes','draw.setLocalAxes'),
        ('&GlobalAxes','draw.setGlobalAxes'),
        ('&Projection','draw.setProjection'),
        ('&Perspective','draw.setPerspective'),
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
