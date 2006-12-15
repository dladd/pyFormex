#!/usr/bin/env python
# $Id$
"""Graphical User Interface for pyformex."""

import globaldata as GD

import sys,time,os.path,string,re

from PyQt4 import QtCore, QtGui, QtOpenGL
import menu
import cameraMenu
import fileMenu
import scriptsMenu
import prefMenu
import toolbar
import canvas
import views
import script
import utils
import draw


_start_message = GD.Version + ', by B. Verhegghe'

# Find interesting supporting software

# import (ImageMagick)
m = re.match("Version: ImageMagick (\S+) .*",script.system('import -version'))
if m:
    GD.magick_version = m.group(1)
    script.message("Congratulations! You have ImageMagick version %s" % GD.magick_version)


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

    def write(self,s):
        """Write a string to the message board."""
        s = s.rstrip('\n')
        if len(s) > 0:
            self.append(s)
            self.cursor.movePosition(QtGui.QTextCursor.End)
            self.setTextCursor(self.cursor)


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

def printsize(w,t=None):
    print "%s %s x %s" % (t,w.width(),w.height())


################# GUI ###############
class GUI(QtGui.QMainWindow):
    """Implements a GUI for pyformex."""

    def __init__(self,windowname,size=(800,600),pos=(0,0)):
        """Constructs the GUI.

        The GUI has a central canvas for drawing, a menubar and a toolbar
        on top, and a statusbar at the bottom.
        """
        QtGui.QMainWindow.__init__(self)
        self.setWindowTitle(windowname)
        # add widgets to the main window
        self.statusbar = self.statusBar()
        self.curfile = QtGui.QLabel('No File')
        self.curfile.setLineWidth(0)
        self.statusbar.addWidget(self.curfile)
        self.smiley = QtGui.QLabel()
        self.statusbar.addWidget(self.smiley)
        self.menu = self.menuBar()
        self.toolbar = self.addToolBar('Top ToolBar')
        self.editor = None
        # Create a box for the central widget
        self.box = QtGui.QWidget()
        self.boxlayout = QtGui.QVBoxLayout()
        self.box.setLayout(self.boxlayout)
        self.setCentralWidget(self.box)
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
        # ... and the toolbar
        self.actions = toolbar.addActionButtons(self.toolbar)
        self.toolbar.addSeparator()
        if GD.cfg['gui/camerabuttons']:
            toolbar.addCameraButtons(self.toolbar)
        self.menu.show()
        # Create a menu with standard views
        # and insert it before the help menu
        self.viewsMenu = None
        if GD.cfg['gui/viewsmenu']:
            self.viewsMenu = views.ViewsMenu()
            self.insertMenu(self.viewsMenu)
        # Install the default canvas views
        # defviews = self.canvas.views.keys()
        # NO, these are not sorted, better:
        defviews = [ 'front', 'back', 'top', 'bottom', 'left', 'right', 'iso' ]
        self.views = views.Views(defviews,self.viewsMenu,self.toolbar)
        # Display the main menubar
        #self.menu.show()
        self.resize(*size)
        self.move(*pos)
        if GD.options.redirect:
            sys.stderr = self.board
            sys.stdout = self.board
        if GD.options.debug:
            printsize(self,'Main:')
            printsize(self.canvas,'Canvas:')
            printsize(self.board,'Board:')


    def insertMenu(self,menu,before='&Help'):
        """Insert a menu in the menubar before the specified menu.

        The new menu can be inserted BEFORE any of the existing menus.
        By default the new menu will be inserted before the Help menu.

        Also, the menu's title should be unique.
        """
        if not self.menus.has_key(str(menu.title())):
            self.menu.insertMenu(self.menus[before],menu)
            self.menus = dict([[str(a.text()),a] for a in self.menu.actions()])


    def removeMenu(self,menu):
        """Remove a menu from the main menubar.

        menu is either a menu title or a menu action.
        """
        if type(menu) == str:
            if self.menus.has_key(menu):
                menu = self.menus[menu]
            else:
                menu = None
        else:
            menu = menu.menuAction()
            if menu not in self.menu.actions():
                menu = None
        if menu is not None:
            self.menu.removeAction(menu)
            self.menus = dict([[str(a.text()),a] for a in self.menu.actions()])
        
    
    def resizeCanvas(self,wd,ht):
        """Resize the canvas."""
        self.canvas.resize(wd,ht)
        self.box.resize(wd,ht+self.board.height())
        self.adjustSize()
    
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


    def Size(self):
        """Return the size of the main window() as a tuple."""
        s = self.size()
        return s.width(),s.height()
    

    def Pos(self):
        """Return the position of the main window() as a tuple."""
        p = self.pos()
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


def setStyle(style):
    """Set the main application style."""
    GD.app.setStyle(style)
    if GD.gui:
        GD.gui.update()


def setFont(font):
    """Set the main application font."""
    GD.app.setFont(font)
    if GD.gui:
        GD.gui.update()


def setFontSize(s=None):
    """Set the main application font size to the given point size."""
    if s:
        GD.cfg['gui/fontsize'] = s
    else:
        s = GD.cfg.get('gui/fontsize',12)
    font = GD.app.font()
    font.setPointSize(int(s))
    setFont(font)


def windowExists(windowname):
    """Check if a GUI window with the given name exists.

    On X-Window systems, we can use the xwininfo cammand to find out whether
    a window with the specified name exists.
    """
    return not os.system('xwininfo -name "%s" > /dev/null 2>&1' % windowname)


def runApp(args):
    """Create and run the qt application."""
    GD.app = QtGui.QApplication(args)
    QtCore.QObject.connect(GD.app,QtCore.SIGNAL("lastWindowClosed()"),GD.app,QtCore.SLOT("quit()"))

    # Set some globals
    GD.image_formats_qt = map(str,QtGui.QImageWriter.supportedImageFormats())
    GD.image_formats_qtr = map(str,QtGui.QImageReader.supportedImageFormats())
    if GD.cfg.get('imagesfromeps',False):
        GD.image_formats_qt = []
    if GD.options.debug:
        print "Qt image types for saving: ",GD.image_formats_qt
        print "Qt image types for input: ",GD.image_formats_qtr
        print "gl2ps image types:",GD.image_formats_gl2ps
        print "image types converted from EPS:",GD.image_formats_fromeps
        
    # create GUI, show it, run it
    windowname = GD.Version
    count = 0
    while windowExists(windowname):
        if count > 255:
            print "Can not open the main window --- bailing out"
            return 1
        count += 1
        windowname = '%s (%s)' % (GD.Version,count)
    if GD.cfg.has_key('gui/fontsize'):
        setFontSize()
    GD.gui = GUI(windowname,GD.cfg['gui/size'],GD.cfg['gui/pos'])
    GD.gui.setcurfile()
    GD.board = GD.gui.board
    GD.canvas = GD.gui.canvas
    GD.gui.show()
    # Create additional menus (put them in a list to save)
    menus = []
    # History
    history = GD.cfg['history']
    if type(history) == list:
        m = scriptsMenu.ScriptsMenu('History',files=history,max=10)
        GD.gui.insertMenu(m)
        menus.append(m)
        GD.gui.history = m
    # Create a menu with pyFormex examples
    # and insert it before the help menu
    for title,dir in GD.cfg['scriptdirs']:
        m = scriptsMenu.ScriptsMenu(title,dir,autoplay=True)
        GD.gui.insertMenu(m)
        menus.append(m)
    GD.board.write(GD.Version+"   (C) B. Verhegghe")
    # remaining args are interpreted as scripts
    for arg in args:
        if os.path.exists(arg):
            draw.play(arg)
    GD.app_started = True
    GD.debug("Using window name %s" % GD.gui.windowTitle())
    GD.app.exec_()

    # store the main window size/pos
    GD.cfg['history'] = GD.gui.history.files
    GD.cfg.update({'size':GD.gui.Size(),'pos':GD.gui.Pos()},name='gui')
    return 0

#### End
