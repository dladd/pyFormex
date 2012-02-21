# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##  Distributed under the GNU General Public License version 3 or later.
##
##
##  This program is free software: you can redistribute it and/or modify
##  it under the terms of the GNU General Public License as published by
##  the Free Software Foundation, either version 3 of the License, or
##  (at your option) any later version.
##
##  This program is distributed in the hope that it will be useful,
##  but WITHOUT ANY WARRANTY; without even the implied warranty of
##  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##  GNU General Public License for more details.
##
##  You should have received a copy of the GNU General Public License
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Menus for the pyFormex GUI.

This modules implements specialized classes and functions for building
the pyFormex GUI menu system.
"""

import pyformex as pf
from pyformex.gui import *
from PyQt4 import QtGui,QtCore # For Sphinx
import odict
import utils

import fileMenu
import cameraMenu
import prefMenu
import viewportMenu
import scriptMenu
import toolbar
import helpMenu
import image
import draw
import script

import os
from gettext import gettext as _

############################# Menu ##############################


class BaseMenu(object):
    """A general menu class.

    A hierarchical menu that keeps a list of its item names and actions.
    The item names are normalized by removing all '&' characters and
    converting the result to lower case.
    It thus becomes easy to search for an existing item in a menu.
    
    This class is not intended for direct use, but through subclasses.
    Subclasses should implement at least the following methods:
    
    - addSeparator()
    - insertSeperator(before)
    - addAction(text,action)
    - insertAction(before,text,action)
    - addMenu(text,menu)
    - insertMenu(before,text,menu)
      
    QtGui.Menu and QtGui.MenuBar provide these methods.
    """

    def __init__(self,title='AMenu',parent=None,before=None,items=None):
        """Create a menu."""
        self._title = title
        pf.debug("Creating menu %s" % title,pf.DEBUG.MENU)
        self.parent = parent
        self.separators = odict.ODict()
        self._actions_ = []
        if items:
            self.insertItems(items)
        if parent and isinstance(parent,BaseMenu):
            before = parent.action(before)
            parent.insert_menu(self,before)


    def actionList(self):
        """Return a list with the current actions."""
        return [ utils.strNorm(str(c.text())) for c in self.actions() ]
    

    def index(self,text):
        """Return the index of the specified item in the actionlist.

        If the requested item is not in the actionlist, -1 is returned.
        """
        try:
            return self.actionList().index(utils.strNorm(text))
        except ValueError:
            return -1
    

    def action(self,text):
        """Return the action with specified text.

        First, a normal action is tried. If none is found,
        a separator is tried.
        """
        if text is None:
            return None
        if text in self.actions():
            return text
        i = self.index(text)
        if i >= 0:
            return self.actions()[i]
        else:
            return self.separators.get(utils.strNorm(text),None)


    def item(self,text):
        """Return the item with specified text.

        For a normal action or a separator, an qction is returned.
        For a menu action, a menu is returned.
        """
        i = self.index(text)
        if i >= 0:
            a = self.actions()[i]
            m = a.menu()
            if m:
                return m
            else:
                return a
        else:
            return self.separators.get(utils.strNorm(text),None)
        


    def nextitem(self,text):
        """Returns the name of the next item.

        This can be used to replace the current item with another menu.
        If the item is the last, None is returned.
        """
        itemlist = self.actionList()
        i = itemlist.index(utils.strNorm(text))
        #print "POS = %s/%s" % (i,len(itemlist))
        if i >= 0 and i < len(itemlist)-1:
            return itemlist[i+1]
        else:
            return None


    def removeItem(self,item):
        """Remove an item from this menu."""
        #print self.actionList()
        action = self.action(item)
        if action:
            #print "INDEED REMOVING %s = %s" % (item,action)
            self.removeAction(action)
            if isinstance(action,QtGui.QMenu):
                action.close()
                del action
        #print self.actionList()
    
    # The need for the following functions demonstrates how much more
    # powerful a dynamically typed language as Python is as compared to
    # the C++ language used by Qt
    def insert_sep(self,before=None):
        """Create and insert a separator"""
        if before:
            return self.insertSeparator(before)
        else:
            return self.addSeparator()

    def insert_menu(self,menu,before=None):
        """Insert an existing menu."""
        if before:
            return self.insertMenu(before,menu)
        else:
            return self.addMenu(menu)

    def insert_action(self,action,before=None):
        """Insert an action.""" 
        if before:
            return self.insertAction(before,action)
        else:
            #print ("ADDING SAVED ACTION %s" % action)
            self._actions_.append(action)
            return self.addAction(action)

    def create_insert_action(self,str,val,before=None):
        """Create and insert an action.""" 
        if before:
            raise RuntimeError,"THIS CAN NOT WORK"
            return self.insertAction(before,str,val)
        else:
            return self.addAction(str,val)
    

    def insertItems(self,items,before=None):
        """Insert a list of items in the menu.

        Parameters:

        - `items`: a list of menuitem tuples. Each item is a tuple of two
          or three elements: (text, action, options):

          - `text`: the text that will be displayed in the menu item.
            It is stored in a normalized way: all lower case and with
            '&' removed.

          - `action`: can be any of the following:

            - a Python function or instance method : it will be called when the
              item is selected,
            - a string with the name of a function/method,
            - a list of Menu Items: a popup Menu will be created that will
              appear when the item is selected,
            - an existing Menu,
            - None : this will create a separator item with no action.

          - `options`: optional dictionary with following honoured fields:

            - `icon`: the name of an icon to be displayed with the item text.
              This name should be that of one of the icons in the pyFormex
              icondir.
            - `shortcut`: is an optional key combination to select the item.
            - `tooltip`: a text that is displayed as popup help.

        - `before`: if specified, should be the text *or* the action of one
          of the items in the Menu (not the items list!): the new list of
          items will be inserted before the specified item.
        """
        before = self.action(before)
        for item in items:
            txt,val = item[:2]
            if len(item) > 2:
                options = item[2]
            else:
                options = {}
            if  val is None:
                a = self.insert_sep(before)
                self.separators[txt] = a
            elif isinstance(val, list):
                a = Menu(txt,parent=self,before=before)
                a.insertItems(val)
            else:
                if type(val) == str:
                    val = eval(val)
                if 'data' in options:
                    # DActions should be saved to keep them alive !!!
                    a = DAction(txt,data = options['data'])
                    QtCore.QObject.connect(a,QtCore.SIGNAL(a.signal),val)
                    self.insert_action(a,before)
                else:
                    if before is not None:
                        raise RuntimeError,"I can not insert a QAction menu item before an existing one."
                    #print ("CREATE %s,%s,%s" % (txt,val,before))
                    a = self.create_insert_action(txt,val,before)
                for k,v in options.items():                        
                    if k == 'icon':
                        a.setIcon(QtGui.QIcon(QtGui.QPixmap(utils.findIcon(v))))
                    elif k == 'shortcut':
                        a.setShortcut(v)
                    elif k == 'tooltip':
                        a.setToolTip(v)
                    elif k == 'checkable':
                        a.setCheckable(v)
                    elif k == 'checked':
                        a.setCheckable(True)
                        a.setChecked(v)
                    elif k == 'disabled':
                        a.setDisabled(True)
                

class Menu(BaseMenu,QtGui.QMenu):
    """A popup/pulldown menu."""

    def __init__(self,title='UserMenu',parent=None,before=None,tearoff=False,items=None):
        """Create a popup/pulldown menu.

        If parent==None, the menu is a standalone popup menu.
        If parent is given, the menu will be inserted in the parent menu.
        If parent==pf.GUI, the menu is inserted in the main menu bar.
        If a parent is given, and tearoff==True, the menu can be teared-off.
        
        If insert == True, the menu will be inserted in the main menubar
        before the item specified by before.
        If before is None or not the normalized text of an item of the
        main menu, the new menu will be inserted at the end.
        Calling the close() function of an inserted menu will remove it
        from the main menu.

        If insert == False, the created menu will be an independent dialog
        and the user will have to process it explicitely.
        """
        QtGui.QMenu.__init__(self,title,parent)
        BaseMenu.__init__(self,title,parent,before,items)
        if parent is None:
            self.setWindowFlags(QtCore.Qt.Dialog)
            self.setWindowTitle(title)
        else:
            if tearoff:
                print "TEAR OFF menus currently not implemented"
                tearoff = False
            self.setTearOffEnabled(tearoff)
        self.done = False
            

    def process(self):
        if not self.done:
            if not self.insert:
                self.show()
            pf.app.processEvents()


    def remove(self):
        self.close()
        self.parent.removeItem(self.title())


class MenuBar(BaseMenu,QtGui.QMenuBar):
    """A menu bar allowing easy menu creation."""

    def __init__(self,title='TopMenuBar'):
        """Create the menubar."""
        QtGui.QMenuBar.__init__(self)
        BaseMenu.__init__(self,title)


    def title(self):
        return self._title


###################### Action List ############################################

class DAction(QtGui.QAction):
    """A DAction is a QAction that emits a signal with a string parameter.

    When triggered, this action sends a signal (default 'Clicked') with a
    custom string as parameter. The connected slot can then act depending
    on this parameter.
    """

    signal = "Clicked"
    
    def __init__(self,name,icon=None,data=None,signal=None):
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
        if signal is None:
            signal = DAction.signal
        self.signal = signal
        if data is None:
            data = name
        self.setData(QtCore.QVariant(data))
        self.connect(self,QtCore.SIGNAL("triggered()"),self.activated)
        
    def activated(self):
        self.emit(QtCore.SIGNAL(self.signal), str(self.data().toString()))


class ActionList(object):
    """Menu and toolbar with named actions.

    An action list is a list of strings, each connected to some action.
    The actions can be presented in a menu and/or a toolbar.
    On activating one of the menu or toolbar buttons, a given signal is
    emitted with the button string as parameter. A fixed function can be
    connected to this signal to act dependent on the string value.
    """

    def __init__(self,actions=[],function=None,menu=None,toolbar=None,icons=None,text=None):
        """Create an new action list, empty by default.

        A list of strings can be passed to initialize the actions.
        If a menu and/or toolbar are passed, a button is added to them
        for each string in the action list.
        If a function is passed, it will be called with the string as
        parameter when the item is triggered.

        If no icon names are specified, they are taken equal to the
        action names. Icons will be taken from the installed icon directory.
        If you want to specify other icons, use the add() method.
        """
        self.actions = []
        self.function = function
        self.menu = menu
        self.toolbar = toolbar
        if icons is None:
            icons = actions
        icons = map(utils.findIcon,icons)
        if text is None:
            text = actions
        for name,icon,txt in zip(actions,icons,text):
            self.add(name,icon,txt)


    def add(self,name,icon=None,text=None):
        """Add a new name to the actions list and create a matching DAction.

        If the actions list has an associated menu or toolbar,
        a matching button will be inserted in each of these.
        If an icon is specified, it will be used on the menu and toolbar.
        The icon is either a filename or a QIcon object.
        If text is specified, it is displayed instead of the action's name.
        """
        if type(icon) == str:
            if os.path.exists(icon):
                icon = QtGui.QIcon(QtGui.QPixmap(icon))
            else:
                raise RuntimeError,'Icons not installed properly'
        if text is None:
            text = name
        a = DAction(text,icon,name)
        if self.function:
            QtCore.QObject.connect(a,QtCore.SIGNAL(a.signal),self.function)
        self.actions.append([name,a])
        if self.menu:
            self.menu.addAction(a)
        if self.toolbar:
            self.toolbar.addAction(a)


    def names(self):
        """Return an ordered list of names of the action items."""
        return [ i[0] for i in self.actions ]


## ###########################################################################

# pyFormex main menus


save = NotImplemented
saveAs = NotImplemented

## def editor():
##     if pf.GUI.editor:
##         print("Close editor")
##         pf.GUI.closeEditor()
##     else:
##         print("Open editor")
##         pf.GUI.showEditor()

 
def resetGUI():
    """Reset the GUI to its default operating mode.

    When an exception is raised during the execution of a script, the GUI
    may be left in a non-consistent state.
    This function may be called to reset most of the GUI components
    to their default operating mode. 
    """
    ## resetPick()
    pf.GUI.resetCursor()
    pf.GUI.actions['Play'].setEnabled(True)
    pf.GUI.actions['Step'].setEnabled(True)
    pf.GUI.actions['Continue'].setEnabled(False)
    pf.GUI.actions['Stop'].setEnabled(False)


def resetWarnings():
    """Reset the warning filters to the default."""
    del pf.prefcfg['warnings/filters']
    print("This will only become effective in your future sessions!")
    print "FILTERS:",pf.prefcfg['warnings/filters']

            
# The menu actions can be simply function names instead of strings, if the
# functions have already been defined here.

def printwindow():
    pf.app.syncX()
    r = pf.GUI.frameGeometry()
    print("Qt4 geom(w,h,x,y): %s,%s,%s,%s" % (r.width(),r.height(),r.x(),r.y()))
    print("According to xwininfo, (x,y) is %s,%s" % pf.GUI.XPos())


_geometry=None

def saveGeometry():
    global _geometry
    _geometry = pf.GUI.saveGeometry()

def restoreGeometry():
    pf.GUI.restoreGeometry(_geometry)


def moveCorrect():
    pf.GUI.move(*pf.GUI.XPos())

def closeLogFile():
    if draw.logfile:
        draw.logfile.close()
        draw.logfile = None
        
def openLogFile():
    fn = draw.askFilename(filter=['*.log','*'],multi=False)
    if fn:
        closeLogFile()
        draw.logfile = open(fn,'w')


def saveBoard():
    fn = draw.askFilename(filter=['*.txt','*'],multi=False,exist=False)
    if fn:
        pf.GUI.board.save(fn)
    

def unloadCurrentApp():
    appname = pf.cfg['curfile']
    import apps
    apps.unload(appname)
    

def createMenuData():
    """Returns the default pyFormex GUI menu data."""

    ActionMenuData = [
        (_('&Step'),draw.step),
        (_('&Continue'),draw.fforward), 
        ## (_('&Reset Picking Mode'),resetPick),
        (_('&Reset GUI'),resetGUI),
        (_('&Reset Warning Filters'),resetWarnings),
        (_('&Force Finish Script'),script.force_finish),
        (_('&Unload Current App'),unloadCurrentApp),
        ## (_('&Execute single statement'),command),
        (_('&Save Message Board'),saveBoard),
        (_('&Open Log File'),openLogFile),
        (_('&Close Log File'),closeLogFile),
        (_('&ListFormices'),script.printall),
        (_('&PrintGlobalNames'),script.printglobalnames),
        (_('&PrintGlobals'),script.printglobals),
        (_('&PrintConfig'),script.printconfig),
        (_('&Print Detected Software'),script.printdetected),
        (_('&Print Loaded Apps'),script.printLoadedApps),
        (_('&Print Used Memory'),script.printVMem),
        (_('&PrintBbox'),draw.printbbox),
        (_('&Print Viewport Settings'),draw.printviewportsettings),
        (_('&Print Window Geometry'),printwindow),
        (_('&Correct the Qt4 Geometry'),moveCorrect),
        (_('&Save Geometry'),saveGeometry),
        (_('&Restore Geometry'),restoreGeometry),
        (_('&Toggle Input Timeout'),toolbar.timeout),
        ]
             

    MenuData = [
        (_('&File'),fileMenu.MenuData),
        (_('&Actions'),ActionMenuData),
        (_('&Help'),helpMenu.createMenuData())
        ]
    
    # Insert configurable menus
    if pf.cfg.get('gui/prefsmenu','True'):
        MenuData[1:1] = prefMenu.MenuData
    if pf.cfg.get('gui/viewportmenu','True'):
        MenuData[2:2] = viewportMenu.MenuData
    if pf.cfg.get('gui/cameramenu','True'):
        MenuData[3:3] = [(_('&Camera'),cameraMenu.MenuData)]

    return MenuData

 
# End
