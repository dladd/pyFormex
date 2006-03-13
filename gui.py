#!/usr/bin/env python
# $Id $
"""Graphical User Interface for pyformex."""

import globaldata as GD
import canvas
import script
import draw
import widgets

#try:
#    import editor         ## non-essential module under testing
#except ImportError:
#    pass

import sys,time,os.path,string

import qt
import qtgl

import pyfotemp as PT


class MyQAction(qt.QAction):
    """A MyQAction is a QAction that sends a string as parameter when clicked."""
    def __init__(self,text,*args):
        qt.QAction.__init__(self,*args)
        self.signal = text
        self.connect(self,qt.SIGNAL("activated()"),self.activated)
        
    def activated(self):
        self.emit(qt.PYSIGNAL("Clicked"), (self.signal,))


###################### Views #############################################
# Views are different camera postitions from where to view the structure.
# They can be activated from menus, or from the  view toolbox
# A number of views are predefined in the canvas class
# Any number of new views can be created, deleted, changed.
# Each view is identified by a string
  
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
    dir = GD.cfg['icondir']
    a = MyQAction(name,qt.QIconSet(qt.QPixmap(os.path.join(dir,icon))),menutext,0,parent)
    qt.QObject.connect(a,qt.PYSIGNAL("Clicked"),draw.view)
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
    global views,viewsMenu,viewsBar
    if tooltip == None:
        tooltip = name
    if menutext == None:
        menutext = name
    dir = GD.cfg['icondir']
    if not GD.canvas.views.has_key(name):
        createViewAction(GD.gui.main,name,icon,tooltip,menutext)
    GD.canvas.createView(name,angles)

def addboard():
    GD.board = GD.gui.addBoard()

class GUI:
    """Implements a GUI for pyformex."""

    def __init__(self):
        """Constructs the GUI.

        The GUI has a central canvas for drawing, a menubar and a toolbar
        on top, and a statusbar at the bottom.
        """
        global viewsMenu,viewsBar
        wd,ht = (GD.cfg.gui['width'],GD.cfg.gui['height'])
        self.main = qt.QMainWindow()
        self.main.setCaption(GD.Version)
        self.main.resize(wd,ht)
        # add widgets to the main window
        self.statusbar = self.main.statusBar()
        self.curfile = qt.QLabel(self.statusbar)
        self.statusbar.addWidget(self.curfile)
        self.smiley = qt.QLabel(self.statusbar)
        self.statusbar.addWidget(self.smiley)
        self.menu = self.main.menuBar()
        self.toolbar = qt.QToolBar(self.main)
        self.editor = None
        # Create a box for the central widgets
        self.box = qt.QVBox(self.main)
        #self.box.setFrameStyle(qt.QFrame.Sunken | qt.QFrame.Panel)
        #self.box.setLineWidth(2)
        self.main.setCentralWidget(self.box)
        # Create a splitter
        s = qt.QSplitter(self.box)
        s.setOrientation(qt.QSplitter.Vertical)
        s.setLineWidth(0)
        # Create an OpenGL canvas with a nice frame around it
        fmt = qtgl.QGLFormat.defaultFormat()
        fmt.setDirectRendering(GD.options.dri)
        c = canvas.Canvas(wd,ht,fmt,s)
        c.setMinimumHeight(100)
        c.resize(wd,ht)
        self.canvas = c
        # Create the message board
        b = qt.QTextEdit(s)
        b.setReadOnly(True) 
        b.setTextFormat(qt.Qt.PlainText)
        b.setText(GD.Version+' started')
        b.setMinimumHeight(32)
        b.setFrameStyle(qt.QFrame.Sunken | qt.QFrame.Panel)
        b.setLineWidth(0)
        self.board = b
        s.setSizes([wd,64])
        # Populate the menu ...
        PT.insertExampleMenu()
        PT.addDefaultMenu(self.menu)
        # ... and the toolbar
        self.actions = PT.addActionButtons(self.toolbar)
        if GD.cfg.gui.setdefault('camerabuttons',True):
            PT.addCameraButtons(self.toolbar)
        # ... and the views menu
        viewsMenu = None
        if GD.cfg.gui.setdefault('viewsmenu',True):
            viewsMenu = qt.QPopupMenu(self.menu)
            self.menu.insertItem('View',viewsMenu,-1,2)
        # ... and the views toolbar
        viewsBar = None
        if GD.cfg.gui.setdefault('viewsbar',True):
            viewsBar = qt.QToolBar("Views",self.main)
        # Create View Actions for the default views provided by the canvas
        initViewActions(self.main,GD.cfg.gui.setdefault('builtinviews',['front','back','left','right','top','bottom','iso']))
        #self.showMessage(GD.Version+"   (C) B. Verhegghe")


    def showMessage(self,s):
        """Append a message to the message board."""
        self.board.append(qt.QString(s))
        self.board.moveCursor(qt.QTextEdit.MoveEnd,True)
        self.board.update()

    def clearMessages(self,s):
        """Clear the message board."""
        self.board.setText("")
        self.board.update()
    
    def resize(self,wd,ht):
        """Resize the canvas."""
        self.canvas.resize(wd,ht)
        self.box.resize(wd,ht+self.board.height())
        self.main.adjustSize()

##    def addView(self,a):
##        """Add a new view action to the Views Menu and Views Toolbar."""
##        if self.has('viewsMenu'):
##            a.addTo(self.viewsMenu)
##        if self.has('viewsBar'):
##            a.addTo(self.viewsBar)
    
    def showEditor(self):
        """Start the editor."""
        if not hasattr(self,'editor'):
            self.editor = Editor(self,'Editor')
            self.editor.show()
            self.editor.setText("Hallo\n")

    def closeEditor(self):
        """Close the editor."""
        if hasattr(self,'editor'):
            self.editor.close()
            self.editor = None


def setcurfile(filename):
    """Set the current file and check whether it is a pyFormex script.

    A file that is not a pyFormex script can be loaded in the editor,
    but it can not be played as a pyFormex script.
    A file is considered to be a pyformex script if its name ends in '.py'
    and the first line of the file ends with 'pyformex'. Typically, a
    pyFormex script starts with a line: #!/usr/bin/env pyformex
    """
    GD.cfg.curfile = filename
    GD.gui.curfile.setText(os.path.basename(filename))
    GD.canPlay = script.isPyFormex(filename)
    GD.gui.actions['Play'].setEnabled(GD.canPlay)
    if GD.canPlay:
        icon = 'happy.xbm'
    else:
        icon = 'unhappy.xbm'
    GD.gui.smiley.setPixmap(qt.QPixmap(os.path.join(GD.cfg.icondir,icon)))



def messageBox(message,level='info',actions=['OK']):
    """Display a message box and wait for user response.

    The message box displays a text, an icon depending on the level
    (either 'about', 'info', 'warning' or 'error') and 1-3 buttons
    with the specified action text. The 'about' level has no buttons.

    The function returns the number of the button that was clicked.
    """
    w = qt.QMessageBox()
    if level == 'error':
        return w.critical(w,GD.Version,message,*actions)
    elif level == 'warning':
        return w.warning(w,GD.Version,message,*actions)
    elif level == 'info':
        return w.information(w,GD.Version,message,*actions)
    elif level == 'about':
        return w.about(w,GD.Version,message)



#### End
