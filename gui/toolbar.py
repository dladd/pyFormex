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

toggle_perspective = None # the toggle perspective button

def togglePerspective():
    global toggle_perspective
    mode = not GD.canvas.camera.perspective
    print "Set mode ",mode
    toggle_perspective.setChecked(mode)
    cameraMenu.setPerspective(not GD.canvas.camera.perspective)

def addCameraButtons(toolbar):
    """Add the camera buttons to a toolbar."""
    global toggle_perspective
    dir = GD.cfg['icondir']
    # The buttons have the following fields:
    #  0 : tooltip
    #  1 : icon
    #  2 : function
    # optional:
    #  3 : REPEAT  (default True)
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
                [ "Zoom Out", "zoomout", cameraMenu.zoomOut ],
                [ "Zoom In", "zoomin", cameraMenu.zoomIn ],
                [ "Zoom All", "zoomall", draw.zoomAll, False ],
                ]
    for but in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(os.path.join(dir,but[1])+GD.cfg['gui/icontype']))
        a = toolbar.addAction(icon,but[0],but[2])
        b =  toolbar.children()[-1] # Get the QToolButton for the last action
        if len(but) < 4 or but[3]:
            b.setAutoRepeat(True)
            QtCore.QObject.connect(b,QtCore.SIGNAL("clicked()"),a,QtCore.SLOT("trigger()"))
        if len(but) >= 5:
            b.setCheckable(but[4])
            b.connect(b,QtCore.SIGNAL("clicked()"),QtCore.SLOT("toggle()"))
            
        b.setToolTip(but[0])

    # Add the toggle_perspective button
    icon_on = QtGui.QPixmap(os.path.join(dir,'perspect')+GD.cfg['gui/icontype'])
    icon_off = QtGui.QPixmap(os.path.join(dir,'project')+GD.cfg['gui/icontype'])
    icon = QtGui.QIcon()
    icon.addPixmap(icon_on,QtGui.QIcon.Normal,QtGui.QIcon.On)
    icon.addPixmap(icon_off,QtGui.QIcon.Normal,QtGui.QIcon.Off)
    a = toolbar.addAction(icon,'Toggle Perspective/Projective Mode', togglePerspective)
    b = toolbar.children()[-1]
    b.setCheckable(True)
    #b.connect(b,QtCore.SIGNAL("clicked()"),QtCore.SLOT("toggle()"))
    b.setChecked(True)
    #toggle()
    toggle_perspective = b     

# End
