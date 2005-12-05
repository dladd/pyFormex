#!/usr/bin/env python
# $Id $
"""Graphical User Interface for pyformex."""

import globaldata as GD
import canvas
from widgets import MyQAction

import pyfotemp as PT

import sys,time,os.path,string
import qt
import qtgl


###################### Views #############################################
# Views are different camera postitions from where to view the structure.
# They can be activated from menus, or from the  view toolbox
# A number of views are predefined in the canvas class
# Any number of new views can be created, deleted, changed.
# Each view is identified by a string
    
def view(v):
    """Show a named view, either a builtin or a user defined."""
    global canvas
    if canvas.views.has_key(v):
        canvas.setView(None,v)
        canvas.update()
    else:
        warning("A view named '%s' has not been created yet" % v)
  
def initViewActions(parent,viewlist):
    """Create the initial set of view actions."""
    global views
    views = []
    for name in viewlist:
        icon = name+"view.xbm"
        Name = string.capitalize(name)
        tooltip = Name+" View"
        menutext = "&"+Name
        createViewAction(parent,name,icon,tooltip,menutext)

def createViewAction(parent,name,icon,tooltip,menutext):
    """Creates a view action and adds it to the menu and/or toolbar.

    The view action is a MyQAction which sends the name when activated.
    It is added to the viewsMenu and/or the viewsBar if they exist.
    The toolbar button has icon and tooltip. The menu item has menutext. 
    """
    global views,viewsMenu,viewsBar
    dir = GD.config['icondir']
    a = MyQAction(name,tooltip,qt.QIconSet(qt.QPixmap(os.path.join(dir,icon))),menutext,0,parent)
    qt.QObject.connect(a,qt.PYSIGNAL("Clicked"),view)
    views.append(name)
    if viewsMenu:
        a.addTo(viewsMenu)
    if viewsBar:
        a.addTo(viewsBar)
 
def addView(name,angles,icon="userview.xbm",tooltip=None,menutext=None):
    """Add a new view to the list of predefined views.

    This creates a new named view with specified angles for the canvas.
    It also creates a MyQAction which sends the name when activated, and
    adds the MyQAction to the viewsMenu and/or the viewsBar if they exist.
    """
    global views,viewsMenu,viewsBar,canvas,gui
    if tooltip == None:
        tooltip = name
    if menutext == None:
        menutext == name
    dir = GD.config['icondir']
    canvas.createView(name,angles)
    createViewAction(name,icon,tooltip,menutext)


class GUI:
    """Implements a GUI for pyformex."""

    def __init__(self):
        """Constructs the GUI.

        The GUI has a central canvas for drawing, a menubar and a toolbar
        on top, and a statusbar at the bottom.
        """
        global viewsMenu,viewsBar
        wd,ht = (GD.config['width'],GD.config['height'])
        self.main = qt.QMainWindow()
        self.main.setCaption(GD.Version)
        self.main.resize(wd,ht)
        # add widgets to the main window
        self.statusbar = self.main.statusBar()
        self.message = qt.QLabel(self.statusbar)
        self.statusbar.addWidget(self.message)
        self.showMessage(GD.Version+" (c) B. Verhegghe")
        self.menu = self.main.menuBar()
        self.toolbar = qt.QToolBar(self.main)
        # Create an OpenGL canvas with a nice frame around it
        f = qt.QHBox(self.main)
        f.setFrameStyle(qt.QFrame.Sunken | qt.QFrame.Panel)
        f.setLineWidth(2)
        f.resize(wd,ht)
        fmt = qtgl.QGLFormat.defaultFormat()
        fmt.setDirectRendering(GD.options.dri)
        self.canvas = canvas.Canvas(wd,ht,fmt,f)
        self.main.setCentralWidget(f)
        # Populate the menu ...
        PT.insertExampleMenu()
        PT.addDefaultMenu(self.menu)
        # ... and the toolbar
        self.actions = PT.addActionButtons(self.toolbar)
        if GD.config.setdefault('camerabuttons',True):
            PT.addCameraButtons(self.toolbar)
        # ... and the views menu
        viewsMenu = None
        if GD.config.setdefault('viewsmenu',True):
            viewsMenu = qt.QPopupMenu(self.menu)
            self.menu.insertItem('View',viewsMenu,-1,2)
        # ... and the views toolbar
        viewsBar = None
        if GD.config.setdefault('viewsbar',True):
            viewsBar = qt.QToolBar("Views",self.main)
        # Create View Actions for the default views provided by the canvas
        initViewActions(self.main,GD.config.setdefault('builtinviews',['front','back','left','right','top','bottom','iso']))

    def showMessage(self,s):
        """Display a permanent message in the status line."""
        self.message.setText(qt.QString(s))

    def addView(self,a):
        """Add a new view action to the Views Menu and Views Toolbar."""
        if self.has('viewsMenu'):
            a.addTo(self.viewsMenu)
        if self.has('viewsBar'):
            a.addTo(self.viewsBar)


###########################  app  ################################

def runApp(args):
    """Create and run the qt application."""
    global app_started
    GD.app = qt.QApplication(args)
    qt.QObject.connect(GD.app,qt.SIGNAL("lastWindowClosed()"),GD.app,qt.SLOT("quit()"))
    # create GUI, show it, run it
    GD.gui = GUI()
    GD.canvas = GD.gui.canvas
    GD.app.setMainWidget(GD.gui.main)
    GD.gui.main.show()
    GD.options.gui = True
    # remaining args are interpreted as scripts
    GD.app_started = False
    for arg in args:
        if os.path.exists(arg):
            playFile(arg)
    GD.app_started = True
    GD.app.exec_loop()

#### End
