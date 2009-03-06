# $Id$
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

