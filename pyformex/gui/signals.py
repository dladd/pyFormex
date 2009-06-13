# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 09:32:38 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""signals.py: Definition of our own signals used in the GUI communication.

Signals are treated by the normal QT4 machine. They can be emitted from
anywhere, causing attached functions to be executed.
"""
import pyformex as GD

from PyQt4 import QtCore

# signals
CANCEL = QtCore.SIGNAL("Cancel")   # cancel the operation, undoing it
DONE   = QtCore.SIGNAL("Done")     # accept and finish te operation
WAKEUP = QtCore.SIGNAL("Wakeup")   # wake up from a sleep state
TIMEOUT = QtCore.SIGNAL("Timeout") # terminate what was going on
SAVE = QtCore.SIGNAL("Save")       # 

def onSignal(signal,function,widget=None):
    if widget is None:
        widget = GD.GUI
    QtCore.QObject.connect(widget,signal,function)

def offSignal(signal,function,widget=None):
    if widget is None:
        widget = GD.GUI
    QtCore.QObject.disconnect(widget,signal,function)


    
# End

