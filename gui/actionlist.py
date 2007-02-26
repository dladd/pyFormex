#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Menu and toolbar with named actions."""

import os
from PyQt4 import QtCore, QtGui

class DAction(QtGui.QAction):
    """A DAction is a QAction that emits a signal with a string parameter.

    When triggered, this action sends a signal (default 'Clicked') with a
    custom string as parameter. The connected slot can then act depending
    on this parameter.
    """

    signal = "Clicked"
    
    def __init__(self,name,icon=None,data=None):
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
        if not data:
            data = name
        self.setData(QtCore.QVariant(data))
        self.connect(self,QtCore.SIGNAL("triggered()"),self.activated)
        
    def activated(self):
        self.emit(QtCore.SIGNAL(DAction.signal), str(self.data().toString()))


###################### Action List ############################################

class ActionList(object):
    """A list of named actions.

    An action list is a list of strings, each connected to some action.
    The actions can be presented in a menu and/or a toolbar.
    On activating one of the menu or toolbar buttons, a given signal is
    emitted with the button string as parameter. A fixed function can be
    connected to this signal to act dependent on the string value.
    """

    def __init__(self,actions=[],function=None,menu=None,toolbar=None,iconpath=None):
        """Create an new action list, empty by default.

        A list of strings can be passed to initialize the actions.
        If a menu and/or toolbar are passed, a button is added to them
        for each string in the action list.
        If a function is passed, it will be called with the string as
        parameter when the item is triggered.

        If an iconpath is specified, it will be used on all menu and toolbar
        . All the toolbar buttons
        will get this icon If it includes a '%s' it will
        """
        self.actions = []
        self.function = function
        self.menu = menu
        self.toolbar = toolbar
        self.iconpath = iconpath
        self.iconpathreplace = iconpath is not None and iconpath.find('%s') >= 0
        for name in actions:
            self.add(name)


    def add(self,name,icon=None):
        """Add a new name to the actions list and create a matching DAction.

        If the actions list has an associated menu or toolbar,
        a matching button will be inserted in each of these.
        If an icon file is specified, it will be used on the menu and toolbar.
        If None is specified the ActionList's iconpath will be used if one
        was specified. If the latter contains a substring '%s', it will be
        replaced with the button string.
        """
        if icon is None and self.iconpath is not None:
            if self.iconpathreplace:
                icon = self.iconpath % name
            else:
                icon = iconpath

        if os.path.exists(icon):
            icon = QtGui.QIcon(QtGui.QPixmap(icon))
        menutext = '&' + name.capitalize()
        a = DAction(menutext,icon,name)
        if self.function:
            QtCore.QObject.connect(a,QtCore.SIGNAL(a.signal),self.function)
        self.actions.append([name,a])
        if self.menu:
            self.menu.addAction(a)
        if self.toolbar:
            self.toolbar.addAction(a)

# End
