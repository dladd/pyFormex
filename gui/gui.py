#!/usr/bin/env python
# $Id $
"""Graphical User Interface for pyformex."""

import globaldata as GD
#import pyfotemp as PT
#import decorations
#import widgets

#try:
#    import editor         ## non-essential module under testing
#except ImportError:
#    pass

import sys,time,os.path,string

from PyQt4 import QtCore, QtGui, QtOpenGL
import menu
import viewMenu
import canvas
import script
import draw


_start_message = GD.Version + ', by B. Verhegghe'

GD.image_formats_qt = map(str,QtGui.QImageWriter.supportedImageFormats())


class MyQAction(QtGui.QAction):
    """A MyQAction is a QAction that sends a string as parameter when clicked."""
    def __init__(self,text,*args):
        QtGui.QAction.__init__(self,*args)
        self.signal = text
        self.connect(self,QtCore.SIGNAL("activated()"),self.activated)
        
    def activated(self):
        self.emit(QtCore.PYSIGNAL("Clicked"), (self.signal,))


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

################### Script action toolbar ###########
def addActionButtons(toolbar):
    """Add the script action buttons to the toolbar."""
    action = {}
    dir = GD.cfg.icondir
    buttons = [ [ "Play", "next.xbm", script.play, False ],
                [ "Step", "nextstop.xbm", draw.step, False ],
                [ "Continue", "ff.xbm", draw.fforward, False ],
              ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,b[1])))
        a = toolbar.addAction(icon,b[0],b[2])
        a.setEnabled(b[3])
        action[b[0]] = a
    return action

################# Camera action toolbar ###############
def addCameraButtons(toolbar):
    """Add the camera buttons to a toolbar."""
    
    dir = GD.cfg['icondir']
    buttons = [ [ "Rotate left", "rotleft.xbm", viewMenu.rotLeft ],
                [ "Rotate right", "rotright.xbm", viewMenu.rotRight ],
                [ "Rotate up", "rotup.xbm", viewMenu.rotUp ],
                [ "Rotate down", "rotdown.xbm", viewMenu.rotDown ],
                [ "Twist left", "twistleft.xbm", viewMenu.twistLeft ],
                [ "Twist right", "twistright.xbm", viewMenu.twistRight ],
                [ "Translate left", "left.xbm", viewMenu.transLeft ],
                [ "Translate right", "right.xbm", viewMenu.transRight ],
                [ "Translate down", "down.xbm", viewMenu.transDown ],
                [ "Translate up", "up.xbm", viewMenu.transUp ],
                [ "Zoom In", "zoomin.xbm", viewMenu.zoomIn ],
                [ "Zoom Out", "zoomout.xbm", viewMenu.zoomOut ],  ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,b[1])))
        a = toolbar.addAction(icon,b[0],b[2])
        #a.setAutoRepeat(True)
    for w in toolbar.children():
        print w


def printFormat(fmt):
    """Print partial information about the OpenGL format."""
    print "Double Buffer: ",fmt.doubleBuffer()
    print "Depth Buffer: ",fmt.depth()
    print "RGBA: ",fmt.rgba()


class Board(QtGui.QTextEdit):
    """Message board for displaying read-only plain text messages."""
    
    def __init__(self,parent=None):
        """Construct the Message Board widget."""
        QtGui.QTextEdit.__init__(self,parent)
        self.setReadOnly(True) 
        self.setAcceptRichText(False)
        self.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
        # self.setLineWidth(0) # no meaning with panel
        self.setMinimumHeight(32)
##        if text:
##            self.setPlainText(text)


class QtCanvas(QtOpenGL.QGLWidget,canvas.Canvas):
    """A canvas for OpenGL rendering."""
    
    def __init__(self,*args):
        """Initialize an empty canvas with default settings.
        """
        QtOpenGL.QGLWidget.__init__(self,*args)
        self.setMinimumSize(32,32)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        w,h = 800,600
        self.resize(w,h)
        canvas.Canvas.__init__(self,w,h)
        
    def initializeGL(self):
        if GD.options.debug:
            print "initializeGL: "
            p = self.sizePolicy()
            print p.horizontalPolicy(), p.verticalPolicy(), p.horizontalStretch(), p.verticalStretch()
        self.glinit()

    def	resizeGL(self,w,h):
        if GD.options.debug:
            print "resizeGL: %s x %s" % (w,h)
        self.setSize(w,h)

    def	paintGL(self):
        if GD.options.debug:
            print "paintGL:"
        self.display()

def printsize(w):
    print "%s x %s" % (w.width(),w.height())


class GUI:
    """Implements a GUI for pyformex."""

    def __init__(self):
        """Constructs the GUI.

        The GUI has a central canvas for drawing, a menubar and a toolbar
        on top, and a statusbar at the bottom.
        """
        global viewsMenu,viewsBar
        wd,ht = (GD.cfg.gui['width'],GD.cfg.gui['height'])
        self.main = QtGui.QMainWindow()
        self.main.setWindowTitle(GD.Version)
        self.readSettings()
        # add widgets to the main window
        self.statusbar = self.main.statusBar()
        self.curfile = QtGui.QLabel('No File')
        self.curfile.setLineWidth(0)
        self.statusbar.addWidget(self.curfile)
        self.smiley = QtGui.QLabel()
        self.statusbar.addWidget(self.smiley)
        self.menu = self.main.menuBar()
        self.toolbar = self.main.addToolBar('Top ToolBar')
        self.editor = None
        # Create a box for the central widget
        self.box = QtGui.QWidget()
        self.boxlayout = QtGui.QVBoxLayout()
        self.box.setLayout(self.boxlayout)
        self.main.setCentralWidget(self.box)
        #self.box.setFrameStyle(qt.QFrame.Sunken | qt.QFrame.Panel)
        #self.box.setLineWidth(2)
        # Create a splitter
        s = QtGui.QSplitter()
        s.setOrientation(QtCore.Qt.Vertical)
        s.show()
        #s.moveSplitter(300,0)
        #s.moveSplitter(300,1)
        #s.setLineWidth(0)
        # Create an OpenGL canvas with a nice frame around it
        fmt = QtOpenGL.QGLFormat.defaultFormat()
        fmt.setDirectRendering(GD.options.dri)
        #fmt.setRgba(False)
        if GD.options.debug:
            printFormat(fmt)
        c = QtCanvas(fmt)
##        c = canvas.Canvas(wd,ht,fmt,s)
        c.setBgColor(GD.cfg['bgcolor'])
        c.resize(wd,ht)
##        if GD.options.splash:
##            c.addDecoration(decorations.TextActor(_start_message,wd/2,ht/2,font='tr24',adjust='center',color='red'))
        self.canvas = c
        # Create the message board
        self.board = Board()
        self.board.setPlainText(GD.Version+' started')
        self.boxlayout.addWidget(s)
        s.addWidget(self.canvas)
        s.addWidget(self.board)
        printsize(self.canvas)
        printsize(s)
        printsize(self.board)
        self.box.setLayout(self.boxlayout)
        # Populate the menubar
##        PT.insertExampleMenu()
##        PT.addDefaultMenu(self.menu)
        menu.addMenuItems(self.menu,menu.MenuData)
        self.menu.show()
        # ... and the toolbar
        self.actions = addActionButtons(self.toolbar)
        if GD.cfg.gui.setdefault('camerabuttons',True):
            addCameraButtons(self.toolbar)
        # ... and the views menu
##        viewsMenu = None
##        if GD.cfg.gui.setdefault('viewsmenu',True):
##            viewsMenu = QtGui.QPopupMenu(self.menu)
##            self.menu.insertItem('View',viewsMenu,-1,2)
##        # ... and the views toolbar
##        viewsBar = None
##        if GD.cfg.gui.setdefault('viewsbar',True):
##            viewsBar = QtGui.QToolBar("Views",self.main)
##        # Create View Actions for the default views provided by the canvas
##        initViewActions(self.main,GD.cfg.gui.setdefault('builtinviews',['front','back','left','right','top','bottom','iso']))
        self.showMessage(GD.Version+"   (C) B. Verhegghe")
        self.showMessage("%s" % self.statusbar.isSizeGripEnabled())


    def showMessage(self,s):
        """Append a message to the message board."""
        self.board.append(s)
        self.board.ensureCursorVisible()
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


    def update(self):
        self.main.update()

        
    def readSettings(self):
        settings = QtCore.QSettings("pyformex.berlios.de", "pyFormex")
        pos = settings.value("pos", QtCore.QVariant(QtCore.QPoint(200, 200))).toPoint()
        size = settings.value("size", QtCore.QVariant(QtCore.QSize(400, 400))).toSize()
        self.main.resize(size)
        self.main.move(pos)


    def writeSettings(self):
        settings = QtCore.QSettings("pyformex.berlios.de", "pyFormex")
        if GD.options.debug:
            print self.main.pos()
            print self.main.size()
        settings.setValue("pos", QtCore.QVariant(self.main.pos()))
        settings.setValue("size", QtCore.QVariant(self.main.size()))



def setcurfile(filename):
    """Set the current file and check whether it is a pyFormex script.

    The checking is done by the function isPyFormex().
    A file that is not a pyFormex script can be loaded in the editor,
    but it can not be played as a pyFormex script.
    """
    GD.cfg.curfile = filename
    GD.canPlay = utils.isPyFormex(filename)
    GD.gui.curfile.setText(os.path.basename(filename))
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
        ans = w.critical(w,GD.Version,message,*actions)
    elif level == 'warning':
        ans = w.warning(w,GD.Version,message,*actions)
    elif level == 'info':
        ans = w.information(w,GD.Version,message,*actions)
    elif level == 'about':
        ans = w.about(w,GD.Version,message)
    GD.gui.update()
    return ans


def runApp(args):
    """Create and run the qt application."""
    global app_started
    GD.app = QtGui.QApplication(args)
    QtCore.QObject.connect(GD.app,QtCore.SIGNAL("lastWindowClosed()"),GD.app,QtCore.SLOT("quit()"))
    # create GUI, show it, run it
    GD.gui = GUI()
##    GD.canvas = GD.gui.canvas
##    GD.app.setMainWidget(GD.gui.main)
    GD.gui.main.show()
    # remaining args are interpreted as scripts
    GD.app_started = False
    for arg in args:
        if os.path.exists(arg):
            play(arg)
    GD.app_started = True
    GD.app.exec_()

    GD.gui.writeSettings()

    #Save the preferences if they changed
    if GD.options.debug:
        if GD.prefsChanged:
            print "Saving prefs: ",GD.cfg
        else:
            print "Not saving prefs, because nothing changed" 
    if GD.prefsChanged:
        savePreferences()

#### End
