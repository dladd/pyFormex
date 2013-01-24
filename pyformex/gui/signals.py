# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
"""signals.py: Definition of our own signals used in the GUI communication.

Signals are treated by the normal QT4 machine. They can be emitted from
anywhere, causing attached functions to be executed.
"""
from __future__ import print_function
import pyformex as pf

from gui import QtCore
Qt = QtCore.Qt
QObject = QtCore.QObject
SIGNAL = QtCore.SIGNAL
QThread = QtCore.QThread

# signals
CANCEL = SIGNAL("Cancel")   # cancel the operation, undoing it
DONE   = SIGNAL("Done")     # accept and finish the operation
REDRAW = SIGNAL("Redraw")   # redraw a preview state
WAKEUP = SIGNAL("Wakeup")   # wake up from a sleep state
TIMEOUT = SIGNAL("Timeout") # terminate what was going on
SAVE = SIGNAL("Save")       #
FULLSCREEN = SIGNAL("Fullscreen")    #

keypress_signal = {
    Qt.Key_F2: SAVE,
    Qt.Key_F5: FULLSCREEN,
    }


def onSignal(signal,function,widget=None):
    """Connect a function to a signal"""
    if widget is None:
        widget = pf.GUI
    QObject.connect(widget,signal,function)

def offSignal(signal,function,widget=None):
    """Disconnect a function from a signal"""
    if widget is None:
        widget = pf.GUI
    QObject.disconnect(widget,signal,function)

# End

