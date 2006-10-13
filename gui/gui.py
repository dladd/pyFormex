#!/usr/bin/env python
# $Id$
"""Graphical User Interface for pyformex."""

import globaldata as GD

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


################### Script action toolbar ###########
def addActionButtons(toolbar):
    """Add the script action buttons to the toolbar."""
    action = {}
    dir = GD.cfg['icondir']
    buttons = [ [ "Play", "next", fileMenu.play, False ],
                [ "Step", "nextstop", draw.step, False ],
                [ "Continue", "ff", draw.fforward, False ],
              ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,b[1])+GD.cfg['gui/icontype']))
        a = toolbar.addAction(icon,b[0],b[2])
        a.setEnabled(b[3])
        action[b[0]] = a
    return action

################# Camera action toolbar ###############
def printButtonClicked():
    print "Button CLicked!"

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
    for but in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,but[1])+GD.cfg['gui/icontype']))
        a = toolbar.addAction(icon,but[0],but[2])
        b =  toolbar.children()[-1]
        b.setAutoRepeat(True)
        a.connect(b,QtCore.SIGNAL("clicked()"),QtCore.SLOT("trigger()"))
        b.setToolTip(but[0])
        

def printFormat(fmt):
    """Print partial information about the OpenGL format."""
    print "OpenGL: ",fmt.hasOpenGL()
    print "OpenGLOverlays: ",fmt.hasOpenGLOverlays()
    print "Overlay: ",fmt.hasOverlay()
    print "Plane: ",fmt.plane()
    print "Direct Rendering: ",fmt.directRendering()
    print "Double Buffer: ",fmt.doubleBuffer()
    print "Depth Buffer: ",fmt.depth()
    print "RGBA: ",fmt.rgba()
    print "Alpha: ",fmt.alpha()


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
        if not self.isValid():
            raise RuntimeError,"Could not create a valid OpenGL widget"
        self.setMinimumSize(32,32)
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        canvas.Canvas.__init__(self)
        
    def initializeGL(self):
        if GD.options.debug:
            print "initializeGL: "
            p = self.sizePolicy()
            print p.horizontalPolicy(), p.verticalPolicy(), p.horizontalStretch(), p.verticalStretch()
        self.initCamera()
        self.glinit()

    def	resizeGL(self,w,h):
        GD.debug("resizeGL: %s x %s" % (w,h))
        self.setSize(w,h)

    def	paintGL(self):
        self.display()


def printsize(w,t=None):
    print "%s %s x %s" % (t,w.width(),w.height())


################# GUI ###############
class GUI:
    """Implements a GUI for pyformex."""

    def __init__(self,size=(800,600),pos=(0,0)):
        """Constructs the GUI.

        The GUI has a central canvas for drawing, a menubar and a toolbar
        on top, and a statusbar at the bottom.
        """
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
        if GD.options.dri:
            fmt.setDirectRendering(True)
        if GD.options.nodri:
            fmt.setDirectRendering(False)
        #fmt.setRgba(False)
        if GD.options.debug:
            printFormat(fmt)
        QtOpenGL.QGLFormat.setDefaultFormat(fmt)
        c = QtCanvas()
        c.setBgColor(GD.cfg['draw/bgcolor'])
        c.resize(*GD.cfg['gui/size'])
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
        #print self.menus
        # ... and the toolbar
        self.actions = addActionButtons(self.toolbar)
        self.toolbar.addSeparator()
        if GD.cfg['gui/camerabuttons']:
            addCameraButtons(self.toolbar)
        # Create a menu with standard views
        # and insert it before the help menu
        self.viewsMenu = None
        if GD.cfg['gui/viewsmenu']:
            self.viewsMenu = views.ViewsMenu()
            self.menu.insertMenu(self.menus['&Help'],self.viewsMenu)
        # Install the default canvas views
        # defviews = self.canvas.views.keys()
        # NO, these are not sorted, better:
        defviews = [ 'front', 'back', 'top', 'bottom', 'left', 'right', 'iso' ]
        self.views = views.Views(defviews,self.viewsMenu,self.toolbar)
        # Create a menu with pyFormex examples
        # and insert it before the help menu
        self.examples = scriptsMenu.ScriptsMenu(GD.cfg['exampledir'])
        self.menu.insertMenu(self.menus['&Help'],self.examples)
        # Display the main menubar
        self.menu.show()
        self.resize(*size)
        self.moveto(pos[0],pos[1])
        if GD.options.debug:
            printsize(self.main,'Main:')
            printsize(self.canvas,'Canvas:')
            printsize(self.board,'Board:')
        
    
    def resizeCanvas(self,wd,ht):
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


    def resize(self,w,h):
        """Resize the gui main window."""
        self.main.resize(w,h)

        
    def moveto(self,x,y):
        """Move the gui main window."""
        self.main.move(x,y)


    def size(self):
        """Return the size of the main window()."""
        s = self.main.size()
        return s.width(),s.height()
    

    def pos(self):
        """Return the position of the main window()."""
        p = self.main.pos()
        return p.x(),p.y()
    

    def setcurfile(self,filename=None):
        """Set the current file and check whether it is a pyFormex script.

        The checking is done by the function isPyFormex().
        A file that is not a pyFormex script can be loaded in the editor,
        but it can not be played as a pyFormex script.
        """
        if filename:
            GD.cfg['curfile'] = filename
        else:
            filename = GD.cfg.get('curfile','')
        if filename:
            GD.canPlay = utils.isPyFormex(filename)
            self.curfile.setText(os.path.basename(filename))
            self.actions['Play'].setEnabled(GD.canPlay)
            if GD.canPlay:
                icon = 'happy'
            else:
                icon = 'unhappy'
            self.smiley.setPixmap(QtGui.QPixmap(os.path.join(GD.cfg['icondir'],icon)+GD.cfg['gui/icontype']))


    def addView(self,name,angles):
        """Create a new view and add it to the list of predefined views.

        This creates a named view with specified angles for the canvas.
        If the name already exists in the canvas views, it is overwritten
        by the new angles.
        It adds the view to the views Menu and Toolbar, if these exist and
        do not have the name yet.
        """
        if not GD.canvas.views.has_key(name):
            self.views.add(name)
        GD.canvas.createView(name,angles)


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


def setFontSize(s=None):
    """Set the main application font size to the given point size."""
    if s:
        GD.cfg['gui/fontsize'] = s
    else:
        s = GD.cfg.get('gui/fontsize',12)
    print s
    font = GD.app.font()
    font.setPointSize(int(s))
    GD.app.setFont(font)
    if GD.gui:
        GD.gui.update()


def runApp(args):
    """Create and run the qt application."""
    GD.app = QtGui.QApplication(args)
    QtCore.QObject.connect(GD.app,QtCore.SIGNAL("lastWindowClosed()"),GD.app,QtCore.SLOT("quit()"))

##    if GD.options.config:
##        GD.Cfg = settings.Settings(filename=GD.options.config)
##    else:
##        GD.Cfg = settings.Settings("pyformex.berlios.de", "pyformex")

    # Set some globals
    GD.image_formats_qt = map(str,QtGui.QImageWriter.supportedImageFormats())
    GD.image_formats_qtr = map(str,QtGui.QImageReader.supportedImageFormats())
    if GD.options.debug:
        print "Image types for saving: ",GD.image_formats_qt
        print "Image types for input: ",GD.image_formats_qtr
        
    # create GUI, show it, run it
    if GD.cfg.has_key('gui/fontsize'):
        print "I found a fontsize",GD.cfg['gui/fontsize']
        setFontSize()
    GD.gui = GUI(GD.cfg['gui/size'],GD.cfg['gui/pos'])
    GD.gui.setcurfile()
    GD.board = GD.gui.board
    GD.canvas = GD.gui.canvas
    #print "Canvas Created"
    GD.gui.main.show()   # This creates the X Error ###
    #print "GUI available"
    GD.board.add(GD.Version+"   (C) B. Verhegghe")
    # remaining args are interpreted as scripts
    for arg in args:
        if os.path.exists(arg):
            draw.play(arg)
    GD.app_started = True
    GD.app.exec_()

    # store the main window size/pos
    GD.cfg.update({'size':GD.gui.size(),'pos':GD.gui.pos()},name='gui')

#### End
