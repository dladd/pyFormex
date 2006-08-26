#!/usr/bin/env python
# $Id$
"""Menu with available views."""

import globaldata as GD
import os
from PyQt4 import QtCore, QtGui
import menu
import draw

###################### Views #############################################
# Views are different camera postitions from where to view the structure.
# They can be activated from menus, or from the  view toolbox
# A number of views are predefined in the canvas class
# Any number of new views can be created, deleted, changed.
# Each view is identified by a string
 
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


    def add(self,name,icon=None):
        """Add a new name to the views list and create a matching DAction.

        If the views list has an associated menu or toolbar,
        a matching button will be inserted in each of these.
        """
        if not icon:
            iconpath = os.path.join(GD.cfg.icondir,name+'view')+GD.iconType
            if not os.path.exists(iconpath):
                iconpath = os.path.join(GD.cfg.icondir,'userview')+GD.iconType
            if os.path.exists(iconpath):
                icon = QtGui.QIcon(QtGui.QPixmap(iconpath))
        menutext = '&' + name.capitalize()
        a = menu.DAction(menutext,icon,name)
        QtCore.QObject.connect(a,QtCore.SIGNAL("Clicked"),draw.view)
        self.views.append([name,a])
        if self.menu:
            self.menu.addAction(a)
        if self.toolbar:
            self.toolbar.addAction(a)


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
        pass

        
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
