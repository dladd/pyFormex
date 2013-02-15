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
"""pyFormex GUI module initialization.

This module is intended to form a single access point to the Python
wrappers for the QT libraries, which form the base of the pyFormex GUI.
By using a single access point, we can better deal with API changes
in the wrappers.

All pyFormex modules accessing QT libraries should do this by importing
from this module.

This module also detects the underlying windowing system.
Currently, the pyFormex GUI is only guaranteed on X11.
For other systems, a warning will be printed that some things may not work.
"""
from __future__ import print_function

import pyformex as pf

import sip
try:
    sip.setapi('QDate', 2)
    sip.setapi('QDateTime', 2)
    sip.setapi('QString', 2)
    sip.setapi('QTextStream', 2)
    sip.setapi('QTime', 2)
    sip.setapi('QUrl', 2)
    sip.setapi('QVariant', 2)
except ValueError, e:
    raise RuntimeError('Could not set PyQt4 API version (%s)' % e)


if pf.options.pyside:
    from PySide import QtCore, QtGui, QtOpenGL
    from PySide.QtCore import Signal
    from PySide.QtCore import Slot

else:
    from PyQt4 import QtCore, QtGui, QtOpenGL
    from PyQt4.QtCore import pyqtSignal as Signal
    from PyQt4.QtCore import pyqtSlot as Slot


try:
    QtGui.QColor.setAllowX11ColorNames(True)
    pf.X11 = True
except:
    print("WARNING: THIS IS NOT AN X11 WINDOW SYSTEM!")
    print("SOME THINGS MAY NOT WORK PROPERLY!")
    pf.X11 = False

# End
