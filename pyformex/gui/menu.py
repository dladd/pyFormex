# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Menus for the pyFormex GUI."""

import pyformex
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
import help
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
        pyformex.debug("Creating menu %s" % title)
        self.parent = parent
        self.menuitems = odict.ODict()
        if items:
            self.insertItems(items)
        if parent and isinstance(parent,BaseMenu):
            if before:
                before = parent.itemAction(before)
            parent.insert_menu(self,before)
            title = utils.strNorm(title)
            if not title in parent.menuitems:
                parent.menuitems[title] = self


    def item(self,text):
        """Get the menu item with text.

        The text will be normalized before searching it.
        If an item with the resulting name exists, it is returned.
        Else None is returned.
        """
        return self.menuitems.get(utils.strNorm(text),None)


    def nextitem(self,text):
        """Returns the name of the next item.

        This can be used to replace the current item with another menu.
        If the item is the last, None is returned.
        """
        i = self.menuitems.pos(utils.strNorm(text))
        #print "POS = %s" % i
        #print self.menuitems._order
        if i is not None and i < len(self.menuitems._order)-1:
            i = self.menuitems._order[i+1]
            #print "FINAL %s" % i
        return i


    def itemAction(self,item):
        """Return the action corresponding to item.

        item is either one of the menu's item texts, or one of its
        values. This method guarantees that the return value is either the
        corresponding Action, or None.
        """
        if item not in self.menuitems.values():
            item = self.item(item)
        if isinstance(item,QtGui.QMenu):
            item = item.menuAction()
        return item
    
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
            return self.addAction(action)

    def create_insert_action(self,str,val,before=None):
        """Create and insert an action.""" 
        if before:
            return self.insertAction(before,str,val)
        else:
            return self.addAction(str,val)
    

    def insertItems(self,items,before=None):
        """Insert a list of items in the menu.
        
        Each item is a tuple of two to five elements:
        Text, Action, [ Icon,  ShortCut, ToolTip ].

        Item text is the text that will be displayed in the menu.
        It will be stored in a normalized way: all lower case and with
        '&' removed.

        Action can be any of the following:
        
        - a Python function or instance method : it will be called when the
          item is selected,
        - a string with the name of a function/method,
        - a list of Menu Items: a popup Menu will be created that will appear
          when the item is selected,
        - an existing Menu,
        - None : this will create a separator item with no action.

        Icon is the name of one of the icons in the installed icondir.
        ShortCut is an optional key combination to select the item.
        Tooltip is a popup help string.

        If before is given, it specifies the text OR the action of one of the
        items in the menu: the new items will be inserted before that one.
        """
        if before:
            before = self.itemAction(before)
        for item in items:
            txt,val = item[:2]
            if  val is None:
                a = self.insert_sep(before)
            elif isinstance(val, list):
                a = Menu(txt,self)
                a.insertItems(val)
            else:
                if type(val) == str:
                    val = eval(val)
                if len(item) > 2 and item[2].has_key('data'):
                    a = DAction(txt,data=item[2]['data'])
                    QtCore.QObject.connect(a,QtCore.SIGNAL(a.signal),val)
                    self.insert_action(a,before)
                else:
                    a = self.create_insert_action(txt,val,before)
                if len(item) > 2:
                    for k,v in item[2].items():                        
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
            txt = utils.strNorm(txt)
            if not txt in self.menuitems:
                self.menuitems[txt] = a
                

class Menu(BaseMenu,QtGui.QMenu):
    """A popup/pulldown menu."""

    def __init__(self,title='UserMenu',parent=None,before=None,tearoff=False,items=None):
        """Create a popup/pulldown menu.

        If parent==None, the menu is a standalone popup menu.
        If parent is given, the menu will be inserted in the parent menu.
        If parent==pyformex.GUI, the menu is inserted in the main menu bar.
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
            self.setTearOffEnabled(tearoff)
        self.done = False
            

    def process(self):
        if not self.done:
            if not self.insert:
                self.show()
            pyformex.app.processEvents()


    def remove(self):
        """Remove this menu from its parent."""
        self.done=True
        if self.parent:
            self.parent.removeAction(self.menuAction())
            for k,v in self.parent.menuitems.items():
                if v == self:
                    del self.parent.menuitems[k]


    def replace(self,menu):
        """Replace this menu with the specified one."""
        self.done=True
        if self.parent:
            self.parent.removeAction(self.menuAction())
            for k,v in self.parent.menuitems.items():
                if v == self:
                    self.parent.menuitems[k] = menu


class MenuBar(BaseMenu,QtGui.QMenuBar):
    """A menu bar allowing easy menu creation."""

    def __init__(self,title='TopMenuBar'):
        """Create the menubar."""
        QtGui.QMenuBar.__init__(self)
        BaseMenu.__init__(self,title)


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

    def __init__(self,actions=[],function=None,menu=None,toolbar=None,icons=None):
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
        for name,icon in zip(actions,icons):
            self.add(name,icon)


    def add(self,name,icon=None):
        """Add a new name to the actions list and create a matching DAction.

        If the actions list has an associated menu or toolbar,
        a matching button will be inserted in each of these.
        If an icon is specified, it will be used on the menu and toolbar.
        The icon is either a filename or a QIcon object. 
        """
        if type(icon) == str:
            if os.path.exists(icon):
                icon = QtGui.QIcon(QtGui.QPixmap(icon))
            else:
                raise RuntimeError,'Icons not installed properly'
        menutext = '&' + name.capitalize()
        a = DAction(menutext,icon,name)
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


###########################################################################

# pyFormex main menus


save = NotImplemented
saveAs = NotImplemented

def editor():
    if pyformex.GUI.editor:
        print("Close editor")
        pyformex.GUI.closeEditor()
    else:
        print("Open editor")
        pyformex.GUI.showEditor()

 
def resetGUI():
    """Reset the GUI to its default operating mode.

    When an exception is raised during the execution of a script, the GUI
    may be left in a non-consistent state.
    This function may be called to reset most of the GUI components
    to their default operating mode. 
    """
    ## resetPick()
    pyformex.GUI.setBusy(False)
    pyformex.GUI.actions['Play'].setEnabled(True)
    pyformex.GUI.actions['Step'].setEnabled(True)
    pyformex.GUI.actions['Continue'].setEnabled(False)
    pyformex.GUI.actions['Stop'].setEnabled(False)


## def resetPick():
##     """This function can be called to reset the GUI picking state.

##     It might be useful if an exception was raised during a picking operation.
##     Calling this function will restore the GUI components to non-picking mode.
##     """
##     if pyformex.canvas.selection_mode is not None:
##         pyformex.canvas.finish_selection()
##     pyformex.GUI.statusbar.removeWidget(pyformex.GUI.pick_buttons)
##     pyformex.GUI.statusbar.removeWidget(pyformex.GUI.filter_combo)

  

def addViewport():
    """Add a new viewport."""
    n = len(pyformex.GUI.viewports.all)
    if n < 4:
        pyformex.GUI.viewports.addView(n/2,n%2)

def removeViewport():
    """Remove a new viewport."""
    n = len(pyformex.GUI.viewports.all)
    if n > 1:
        pyformex.GUI.viewports.removeView()


def viewportSettings():
    """Interactively set the viewport settings."""
    pass

            
# The menu actions can be simply function names instead of strings, if the
# functions have already been defined here.

def printwindow():
    pyformex.app.syncX()
    r = pyformex.GUI.frameGeometry()
    print("Qt4 geom(w,h,x,y): %s,%s,%s,%s" % (r.width(),r.height(),r.x(),r.y()))
    print("According to xwininfo, (x,y) is %s,%s" % pyformex.GUI.XPos())


_geometry=None

def saveGeometry():
    global _geometry
    _geometry = pyformex.GUI.saveGeometry()

def restoreGeometry():
    pyformex.GUI.restoreGeometry(_geometry)


def moveCorrect():
    pyformex.GUI.move(*pyformex.GUI.XPos())

def closeLogFile():
    if draw.logfile:
        draw.logfile.close()
        draw.logfile = None
        
def openLogFile():
    fn = draw.askFilename(filter=['*.log','*'],multi=False)
    if fn:
        closeLogFile()
        draw.logfile = file(fn,'w')
        

ActionMenuData = [
    (_('&Step'),draw.step),
    (_('&Continue'),draw.fforward), 
    ## (_('&Reset Picking Mode'),resetPick),
    (_('&Reset GUI'),resetGUI),
    (_('&Force Finish Script'),draw.force_finish),
    ## (_('&Execute single statement'),command),
    (_('&Open Log File'),openLogFile),
    (_('&Close Log File'),closeLogFile),
    (_('&ListFormices'),script.printall),
    (_('&PrintGlobalNames'),script.printglobalnames),
    (_('&PrintGlobals'),script.printglobals),
    (_('&PrintConfig'),script.printconfig),
    (_('&Print Detected Software'),script.printdetected),
    (_('&PrintBbox'),draw.printbbox),
    (_('&Print Viewport Settings'),draw.printviewportsettings),
    (_('&Print Window Geometry'),printwindow),
    (_('&Correct the Qt4 Geometry'),moveCorrect),
    (_('&Save Geometry'),saveGeometry),
    (_('&Restore Geometry'),restoreGeometry),
#    (_('&Add Project to Status Bar'),gui.addProject),
    ]
             

MenuData = [
    (_('&File'),fileMenu.MenuData),
    (_('&Actions'),ActionMenuData),
    (_('&Help'),help.MenuData)
    ]


def createMenuData():
    """Returns the full data menu."""
    # Insert configurable menus
    if pyformex.cfg.get('gui/prefsmenu','True'):
        MenuData[1:1] = prefMenu.MenuData
    if pyformex.cfg.get('gui/viewportmenu','True'):
        MenuData[2:2] = viewportMenu.MenuData
    if pyformex.cfg.get('gui/cameramenu','True'):
        MenuData[3:3] = [(_('&Camera'),cameraMenu.MenuData)]
    
# End
