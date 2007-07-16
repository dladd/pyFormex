#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Graphical User Interface for pyFormex."""

import globaldata as GD

import sys,time,os.path,string,re

from PyQt4 import QtCore, QtGui

print "Congratulations! You have Qt version %s" % QtCore.QT_VERSION_STR

import menu
import cameraMenu
import fileMenu
import scriptsMenu
import prefMenu
import toolbar
import viewport

import script
import utils
import draw
import widgets


############### General Qt utility functions #######

## might go to a qtutils module

def Size(widget):
    """Return the size of a widget as a tuple."""
    s = widget.size()
    return s.width(),s.height()


def Pos(widget):
    """Return the position of a widget as a tuple."""
    p = widget.pos()
    return p.x(),p.y()

def printsize(w,t=None):
    print "%s %s x %s" % (t,w.width(),w.height())

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
        self.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.cursor = self.textCursor()

    def write(self,s):
        """Write a string to the message board."""
        s = s.rstrip('\n')
        if len(s) > 0:
            self.append(s)
            self.cursor.movePosition(QtGui.QTextCursor.End)
            self.setTextCursor(self.cursor)


## # WHERE SHOULD THIS GO?
## def initViewActions(parent,viewlist):
##     """Create the initial set of view actions."""
##     global views
##     views = []
##     for name in viewlist:
##         icon = name+"view"+GD.cfg['gui/icontype']
##         Name = string.capitalize(name)
##         tooltip = Name+" View"
##         menutext = "&"+Name
##         createViewAction(parent,name,icon,tooltip,menutext)


#####################################
################# GUI ###############
#####################################
            
class GUI(QtGui.QMainWindow):
    """Implements a GUI for pyformex."""

    def __init__(self,windowname,size=(800,600),pos=(0,0),bdsize=(0,0)):
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
        self.menu = widgets.MenuBar()
        self.setMenuBar(self.menu)
        
        self.toolbar = self.addToolBar('Top ToolBar')
        self.editor = None
        # Create a box for the central widget
        self.box = QtGui.QWidget()
        self.setCentralWidget(self.box)
        self.boxlayout = QtGui.QVBoxLayout()
        self.box.setLayout(self.boxlayout)
        #self.box.setFrameStyle(qt.QFrame.Sunken | qt.QFrame.Panel)
        #self.box.setLineWidth(2)
        # Create a splitter
        self.splitter = QtGui.QSplitter()
        self.boxlayout.addWidget(self.splitter)
        self.splitter.setOrientation(QtCore.Qt.Vertical)
        self.splitter.show()
        #s.moveSplitter(300,0)
        #s.moveSplitter(300,1)
        #s.setLineWidth(0)
        
        # Create an OpenGL canvas with a nice frame around it
        viewport.setOpenglFormat()
        self.viewports = viewport.MultiCanvas()
        # and update the default settings
        self.viewports.setDefaults(GD.cfg['canvas'])

        # self.canvas is the complete central widget of the main window
        self.canvas = QtGui.QWidget()
        #self.canvas.setFrameStyle(QtGui.QFrame.StyledPanel | QtGui.QFrame.Sunken)
        self.canvas.setSizePolicy(QtGui.QSizePolicy.MinimumExpanding,QtGui.QSizePolicy.MinimumExpanding)
        self.canvas.resize(*GD.cfg['gui/size'])
        self.canvas.setLayout(self.viewports)

        # Create the message board
        self.board = Board()
        #self.board.setPlainText(GD.Version+' started')
        # Put everything together
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.board)
        #self.splitter.setSizes([(800,200),(800,600)])
        self.box.setLayout(self.boxlayout)
        # Create the top menu
        menu.createMenuData()
        self.menu.addItems(menu.MenuData)
        # ... and the toolbar
        self.actions = toolbar.addActionButtons(self.toolbar)
        if GD.cfg.get('gui/camerabuttons','True'):
            self.toolbar.addSeparator()
            toolbar.addCameraButtons(self.toolbar)
        self.menu.show()

        ##  RENDER MODE menu and toolbar ##
##         if GD.cfg.get('gui/renderbuttons','True'):
##             self.render = toolbar.addRenderButtons(self.toolbar)
        modes = [ 'wireframe', 'smooth', 'smoothwire', 'flat', 'flatwire' ]
        if GD.cfg['gui/modemenu']:
            mmenu = QtGui.QMenu('Render Mode')
        else:
            mmenu = None
        if GD.cfg['gui/modebar']:
            tbar = self.toolbar
            tbar.addSeparator()
        else:
            tbar = None
        self.modebtns = widgets.ActionList(
            modes,draw.renderMode,menu=mmenu,toolbar=tbar)
        if mmenu:
            # insert the menu in the viewport menu
            pmenu = widgets.menuItem(self.menu,'viewport').menu()
            before = widgets.menuItem(pmenu,'background color')
            GD.debug("BEFORE = %s" % before)
            pmenu.insertMenu(before,mmenu)
        ##  VIEWS menu and toolbar
        self.viewsMenu = None
        if GD.cfg.get('gui/viewsmenu',True):
            self.viewsMenu = QtGui.QMenu('&Views')
            self.menu.insertMenu(self.viewsMenu)
        views = GD.cfg['gui/builtinviews']
        if GD.cfg['gui/viewsbar']:
            tbar = self.toolbar
            tbar.addSeparator()
        else:
            tbar = None
        self.viewbtns = widgets.ActionList(
            views,draw.view,
            menu=self.viewsMenu,
            toolbar=tbar,
            icons=['%sview' % t for t in views]
            )
        
        # Display the main menubar
        #self.menu.show()
        self.resize(*size)
        self.move(*pos)
        self.board.resize(*bdsize)
        self.setcurfile()
        if GD.options.redirect:
            sys.stderr = self.board
            sys.stdout = self.board
        if GD.options.debug:
            printsize(self,'Main:')
            printsize(self.canvas,'Canvas:')
            printsize(self.board,'Board:')


    def setStyle(self,style):
        """Set the main application style."""
        GD.debug('Setting new style: %s' % style)
        GD.app.setStyle(style)
        self.update()


    def setFont(self,font):
        """Set the main application font."""
        if type(font) == str:
            f = QtGui.QFont()
            f.fromString(font)
            font = f
        GD.app.setFont(font)
        self.update()


    def setFontFamily(self,family):
        """Set the main application font size to the given point size."""
        font = GD.app.font()
        font.setFamily(family)
        self.setFont(font)


    def setFontSize(self,size):
        """Set the main application font size to the given point size."""
        font = GD.app.font()
        font.setPointSize(int(size))
        self.setFont(font)
         
    
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
            self.actions['Step'].setEnabled(GD.canPlay)
            if GD.canPlay:
                icon = 'happy'
            else:
                icon = 'unhappy'
            self.smiley.setPixmap(QtGui.QPixmap(os.path.join(GD.cfg['icondir'],icon)+GD.cfg['gui/icontype']))


    def setViewAngles(self,name,angles):
        """Create a new view and add it to the list of predefined views.

        This creates a named view with specified angles for the canvas.
        If the name already exists in the canvas views, it is overwritten
        by the new angles.
        It adds the view to the views Menu and Toolbar, if these exist and
        do not have the name yet.
        """
        if name not in self.viewbtns.names():
            iconpath = os.path.join(GD.cfg['icondir'],'userview')+GD.cfg['gui/icontype']
            self.viewbtns.add(name,iconpath)
        GD.canvas.view_angles[name] = angles


    def setBusy(self,busy=True):
        if busy:
            GD.app.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        else:
            GD.app.restoreOverrideCursor()
        GD.app.processEvents()


    def keyPressEvent (self,e):
        """Top level key press event handler.

        Events get here if they are not handled by a lower level handler.
        """
        self.emit(QtCore.SIGNAL("Wakeup"),())
        if e.key() == QtCore.Qt.Key_F2:
            GD.debug('F2 pressed!')
            self.emit(QtCore.SIGNAL("Save"),())
        e.ignore()



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


def windowExists(windowname):
    """Check if a GUI window with the given name exists.

    On X-Window systems, we can use the xwininfo cammand to find out whether
    a window with the specified name exists.
    """
    return not os.system('xwininfo -name "%s" > /dev/null 2>&1' % windowname)


def quit():
    """Quit the GUI"""
    sys.stderr = sys.__stderr__
    sys.stdout = sys.__stdout__
    print "Quitting!!"
    draw.drawrelease()
    if GD.app:
        GD.app.exit()


def runApp(args):
    """Create and run the qt application."""
    GD.app = QtGui.QApplication(args)
    QtCore.QObject.connect(GD.app,QtCore.SIGNAL("lastWindowClosed()"),GD.app,QtCore.SLOT("quit()"))
    QtCore.QObject.connect(GD.app,QtCore.SIGNAL("aboutToQuit()"),quit)
        
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

    GD.gui = GUI(windowname,
                 GD.cfg.get('gui/size',(800,600)),
                 GD.cfg.get('gui/pos',(0,0)),
                 GD.cfg.get('gui/bdsize',(800,600))
                 )

    # set the appearence
    GD.gui.setStyle(GD.cfg.get('gui/style','Plastique'))
    font = GD.cfg.get('gui/font',None)
    if font:
        GD.gui.setFont(font)
    else:
        fontfamily = GD.cfg.get('gui/fontfamily',None)
        if fontfamily:
            GD.gui.setFontFamily(fontfamily)
        fontsize =  GD.cfg.get('gui/fontsize',None)
        if fontsize:
            GD.gui.setFontSize(fontsize)
    
    GD.gui.viewports.addView(0,0)
    GD.board = GD.gui.board
    GD.board.write(GD.Version+"  (C) B. Verhegghe")
    GD.gui.show()
    # Create additional menus (put them in a list to save)
    menus = []
    # History
    history = GD.cfg.get('history',None)
    if type(history) == list:
        m = scriptsMenu.ScriptsMenu('History',files=history,max=10)
        GD.gui.menu.insertMenu(m)
        #menus.append(m)
        GD.gui.history = m
    # Create a menu with pyFormex examples
    # and insert it before the help menu
    for title,dir in GD.cfg['scriptdirs']:
        GD.debug("Loading script dir %s" % dir)
        if os.path.exists(dir):
            m = scriptsMenu.ScriptsMenu(title,dir,autoplay=True)
            GD.gui.menu.insertMenu(m)
            menus.append(m)   # Needed to keep m linked to a name,
                              # else the menu is destroyed!
    GD.message = draw.message
    draw.reset()
    # Load plugins
    # This should be replaced with a plugin registering function
    # in the plugin __init__ ?
    for p in GD.cfg.get('gui/plugins',[]):
        print "loading plugin: %s" % p
        if p == 'stl':
            from plugins import stl_menu
            stl_menu.show_menu()
        elif p == 'formex':
            from plugins import formex_menu
            formex_menu.show_menu()
    GD.gui.update()
    # remaining args are interpreted as scripts
    for arg in args:
        if os.path.exists(arg):
            draw.play(arg)
    GD.app_started = True
    GD.debug("Using window name %s" % GD.gui.windowTitle())
    GD.app.exec_()

    # Cleanup
    draw.drawrelease()

    # store the main window size/pos
    GD.cfg['history'] = GD.gui.history.files
    GD.cfg.update({'size':Size(GD.gui),
                   'pos':Pos(GD.gui),
                   'bdsize':Size(GD.gui.board),
                   },name='gui')
    return 0

#### End
