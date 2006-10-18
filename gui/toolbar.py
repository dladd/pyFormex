#!/usr/bin/env python
# $Id$
"""Toolbars for pyformex GUI."""

import globaldata as GD
import os
from PyQt4 import QtCore, QtGui

import fileMenu
import cameraMenu
import draw

################### Script action toolbar ###########
def addActionButtons(toolbar):
    """Add the script action buttons to the toolbar."""
    action = {}
    dir = GD.cfg['icondir']
    buttons = [ [ "Play", "next", fileMenu.play, False ],
                [ "Step", "nextstop", draw.step, False ],
                [ "Continue", "ff", draw.fforward, False ],
              ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,b[1])+GD.cfg['gui/icontype']))
        a = toolbar.addAction(icon,b[0],b[2])
        a.setEnabled(b[3])
        action[b[0]] = a
    return action

################# Camera action toolbar ###############

def addCameraButtons(toolbar):
    """Add the camera buttons to a toolbar."""
    dir = GD.cfg['icondir']
    buttons = [ [ "Rotate left", "rotleft", cameraMenu.rotLeft ],
                [ "Rotate right", "rotright", cameraMenu.rotRight ],
                [ "Rotate up", "rotup", cameraMenu.rotUp ],
                [ "Rotate down", "rotdown", cameraMenu.rotDown ],
                [ "Twist left", "twistleft", cameraMenu.twistLeft ],
                [ "Twist right", "twistright", cameraMenu.twistRight ],
                [ "Translate left", "left", cameraMenu.transLeft ],
                [ "Translate right", "right", cameraMenu.transRight ],
                [ "Translate down", "down", cameraMenu.transDown ],
                [ "Translate up", "up", cameraMenu.transUp ],
                [ "Zoom In", "zoomin", cameraMenu.zoomIn ],
                [ "Zoom Out", "zoomout", cameraMenu.zoomOut ],  ]
    for but in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,but[1])+GD.cfg['gui/icontype']))
        a = toolbar.addAction(icon,but[0],but[2])
        b =  toolbar.children()[-1]
        b.setAutoRepeat(True)
        a.connect(b,QtCore.SIGNAL("clicked()"),QtCore.SLOT("trigger()"))
        b.setToolTip(but[0])
        

# End
