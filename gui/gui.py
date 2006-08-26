#!/usr/bin/env python
# $Id$
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
import cameraMenu
import fileMenu
import scriptsMenu
import prefMenu
import canvas
import views
import script
import draw
import utils


_start_message = GD.Version + ', by B. Verhegghe'
GD.iconType = '.xpm'


################### Script action toolbar ###########
def addActionButtons(toolbar):
    """Add the script action buttons to the toolbar."""
    action = {}
    dir = GD.cfg.icondir
    buttons = [ [ "Play", "next", fileMenu.play, False ],
                [ "Step", "nextstop", draw.step, False ],
                [ "Continue", "ff", draw.fforward, False ],
              ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,b[1])+GD.iconType))
        a = toolbar.addAction(icon,b[0],b[2])
        a.setEnabled(b[3])
        action[b[0]] = a
    return action

################# Camera action toolbar ###############
def addCameraButtons(toolbar):
    """Add the camera buttons to a toolbar."""
    dir = GD.cfg['icondir']
    buttons = [ [ "Rotate left", "rotleft", cameraMenu.rotLeft ],
                [ "Rotate right", "rotright", cameraMenu.rotRight ],
                [ "Rotate up", "rotup", cameraMenu.rotUp ],
                [ "Rotate down", "rotdown", cameraMenu.rotDown ],
                [ "Twist left", "twistleft", cameraMenu.twistLeft ],
                [ "Twist right", "twistright", cameraMenu.twistRight ],
                [ "Translate left", "left", cameraMenu.transLeft ],
                [ "Translate right", "right", cameraMenu.transRight ],
                [ "Translate down", "down", cameraMenu.transDown ],
                [ "Translate up", "up", cameraMenu.transUp ],
                [ "Zoom In", "zoomin", cameraMenu.zoomIn ],
                [ "Zoom Out", "zoomout", cameraMenu.zoomOut ],  ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,b[1])+GD.iconType))
        a = toolbar.addAction(icon,b[0],b[2])
        #a.setAutoRepeat(True)


def printFormat(fmt):
    """Print partial information about the OpenGL format."""
    print "Double Buffer: ",fmt.doubleBuffer()
    print "Depth Buffer: ",fmt.depth()
    print "RGBA: ",fmt.rgba()


################# Message Board ###############
class Board(QtGui.QTextEdit):
    """Message board for displaying read-only plain text messages."""
    
    def __init__(self,parent=None):
        """Construct the Message Board widget."""
        QtGui.QTextEdit.__init__(self,parent)
        self.setReadOnly(True) 
        self.setAcceptRichText(False)
        self.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
        self.setMinimumSize(24,24)
        self.cursor = self.textCursor()

    def add(self,s):
        """Append a message to the message board."""
        #self.cursor = self.textCursor()
        #print "Voor",self.cursor.position(),self.cursor.anchor()
        self.append(s)
        #self.cursor = self.textCursor()
        #print "Na",self.cursor.position(),self.cursor.anchor()
        self.cursor.movePosition(QtGui.QTextCursor.End)
        self.setTextCursor(self.cursor)
        #self.ensureCursorVisible()
        #self.update()


################# OpenGL Canvas ###############
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
            #print "initializeGL: "
            p = self.sizePolicy()
            print p.horizontalPolicy(), p.verticalPolicy(), p.horizontalStretch(), p.verticalStretch()
        self.glinit()

    def	resizeGL(self,w,h):
        if GD.options.debug:
            print "resizeGL: %s x %s" % (w,h)
        self.setSize(w,h)

    def	paintGL(self):
        self.display()


def printsize(w,t=None):
    print "%s %s x %s" % (t,w.width(),w.height())


class GUI:
    """Implements a GUI for pyformex."""

    def __init__(self):
        """Constructs the GUI.

        The GUI has a central canvas for drawing, a menubar and a toolbar
        on top, and a statusbar at the bottom.
        """
        wd,ht = (GD.cfg.gui['width'],GD.cfg.gui['height'])
        self.main = QtGui.QMainWindow()
        self.main.setWindowTitle(GD.Version)
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
        self.box.setLayout(self.boxlayout)
        # Create the top menu and keep a dict with the main menu items
        menu.addMenuItems(self.menu, menu.MenuData)
        self.menus = dict([ [str(a.text()),a] for a in self.menu.actions()])
        print self.menus
        # Create a menu with standard views
        # and insert it before the help menu
        self.views = None
        if GD.cfg.gui.setdefault('viewsmenu',True):
            self.viewsMenu = views.ViewsMenu()
            self.menu.insertMenu(self.menus['&Help'],self.viewsMenu)
##        # ... and the views toolbar
##        viewsBar = None
##        if GD.cfg.gui.setdefault('viewsbar',True):
##            viewsBar = QtGui.QToolBar("Views",self.main)
##        # Create View Actions for the default views provided by the canvas
##        initViewActions(self.main,GD.cfg.gui.setdefault('builtinviews',['front','back','left','right','top','bottom','iso']))
        # Install the default canvas views
        defviews = self.canvas.views.keys()
        # NO, these are not sorted, better:
        defviews = [ 'front', 'back', 'top', 'bottom', 'left', 'right', 'iso' ]
        self.views = views.Views(defviews,self.viewsMenu)
        # Create a menu with pyFormex examples
        # and insert it before the help menu
        self.examples = scriptsMenu.ScriptsMenu(GD.cfg.exampledir)
        self.menu.insertMenu(self.menus['&Help'],self.examples)
        # Display the main menubar
        if GD.options.debug:
            printsize(self.main,'Main:')
            printsize(self.canvas,'Canvas:')
            printsize(self.board,'Board:')
        self.menu.show()
        # ... and the toolbar
        self.actions = addActionButtons(self.toolbar)
        self.toolbar.addSeparator()
        if GD.cfg.gui.setdefault('camerabuttons',True):
            addCameraButtons(self.toolbar)
        self.readSettings()

    
    def resize(self,wd,ht):
        """Resize the canvas."""
        self.canvas.resize(wd,ht)
        self.box.resize(wd,ht+self.board.height())
        self.main.adjustSize()
    
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
        curfile = str(settings.value("curfile").toString())
        print "Curfile:",curfile
        if curfile:
            self.setcurfile(curfile)
        print "Will now resize main"
        self.main.resize(size)
        print "Will now move main"
        self.main.move(pos)
        print "Settings processed"


    def writeSettings(self):
        if GD.options.debug:
            print "Saving settings"
        settings = QtCore.QSettings("pyformex.berlios.de", "pyFormex")
        settings.setValue("pos", QtCore.QVariant(self.main.pos()))
        settings.setValue("size", QtCore.QVariant(self.main.size()))
        if GD.cfg.curfile:
            settings.setValue("curfile", QtCore.QVariant(GD.cfg.curfile))
        settings.setValue("help/helpdir", QtCore.QVariant('zaag'))


    def setcurfile(self,filename):
        """Set the current file and check whether it is a pyFormex script.

        The checking is done by the function isPyFormex().
        A file that is not a pyFormex script can be loaded in the editor,
        but it can not be played as a pyFormex script.
        """
        GD.cfg.curfile = filename
        GD.canPlay = utils.isPyFormex(filename)
        self.curfile.setText(os.path.basename(filename))
        self.actions['Play'].setEnabled(GD.canPlay)
        if GD.canPlay:
            icon = 'happy'
        else:
            icon = 'unhappy'
        self.smiley.setPixmap(QtGui.QPixmap(os.path.join(GD.cfg.icondir,icon)+GD.iconType))



def messageBox(message,level='info',actions=['OK']):
    """Display a message box and wait for user response.

    The message box displays a text, an icon depending on the level
    (either 'about', 'info', 'warning' or 'error') and 1-3 buttons
    with the specified action text. The 'about' level has no buttons.

    The function returns the number of the button that was clicked.
    """
    w = QtGui.QMessageBox()
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

    # Set some globals
    GD.image_formats_qt = map(str,QtGui.QImageWriter.supportedImageFormats())
    GD.image_formats_qtr = map(str,QtGui.QImageReader.supportedImageFormats())
    if GD.options.debug:
        print "Image types for saving: ",GD.image_formats_qt
        print "Image types for input: ",GD.image_formats_qtr
        
    # create GUI, show it, run it
    GD.gui = GUI()
    GD.board = GD.gui.board
    GD.canvas = GD.gui.canvas
    GD.gui.main.show()
    GD.board.add(GD.Version+"   (C) B. Verhegghe")
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
        prefMenu.savePreferences()

#### End
