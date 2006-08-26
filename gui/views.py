#!/usr/bin/env python
# $Id$
"""Menu with available views."""

import globaldata as GD
import os
from PyQt4 import QtCore, QtGui
import menu
import draw

class Views:
    """A list of named views"""

    def __init__(self,views=[],menu=None,toolbar=None):
        """Create an new views list and add the given names.

        If a menu or toolbar is passed, a button will be added for each name.
        """
        self.views = []
        self.menu = menu
        self.toolbar = toolbar
        for name in views:
            self.add(name)


    def add(self,name):
        """Add a new name to the views list and create a maching MyQaction.

        If the views list has an associated menu or toolbar,
        a matching button will be inserted in these.
        """
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(GD.cfg.icondir)+GD.iconType))
        a = menu.MyQAction(name,icon)
        QtCore.QObject.connect(a,QtCore.SIGNAL("Clicked"),draw.view)
        self.views.append([name,a])
        if self.menu:
            self.menu.addAction(a)
        if self.toolbar:
            self.tool.addAction(a)

        

class ViewsMenu(QtGui.QMenu):
    """A menu of views (camera settings)."""
    
    def __init__(self,views=[]):
        """Create a menu with views in list. 
        
        """
        QtGui.QMenu.__init__(self,'&Views')
        
    def add(view):
        """Add the named view to the views menu.

        When the menubutton is clicked, a PYSIGNAL 'Clicked' is sent with
        the view name as parameter.
        """
        


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
        icon = name+"view"+GD.iconType
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
    a = MyQAction(name,QtGui.QIconSet(QtGui.QPixmap(os.path.join(dir,icon))),menutext,0,parent)
    QtCore.QObject.connect(a,QtCore.PYSIGNAL("Clicked"),draw.view)
    views.append(name)
    if viewsMenu:
        a.addTo(viewsMenu)
    if viewsBar:
        a.addTo(viewsBar)
 
def addView(name,angles,icon=None,tooltip=None,menutext=None):
    """Add a new view to the list of predefined views.

    This creates a new named view with specified angles for the canvas.
    It also creates a MyQAction which sends the name when activated, and
    adds the MyQAction to the viewsMenu and/or the viewsBar if they exist.
    """
    global views,viewsMenu,viewsBar
    if not icon:
        icon = 'userview'+GD.iconType
    if tooltip == None:
        tooltip = name
    if menutext == None:
        menutext = name
    dir = GD.cfg['icondir']
    if not GD.canvas.views.has_key(name):
        createViewAction(GD.gui.main,name,icon,tooltip,menutext)
    GD.canvas.createView(name,angles)
