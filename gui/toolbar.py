#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Mon Jul 30 13:38:48 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Toolbars for pyformex GUI."""

import globaldata as GD
import os
from PyQt4 import QtCore, QtGui

import fileMenu, scriptsMenu
import cameraMenu
import draw
import utils


################### Script action toolbar ###########
def addActionButtons(toolbar):
    """Add the script action buttons to the toolbar."""
    action = {}
    buttons = [ [ "Play", "next", fileMenu.play, False ],
                [ "Step", "nextstop", draw.step, False ],
                [ "Continue", "ff", draw.fforward, False ],
                [ "Stop", "stop", draw.stopatbreakpt, False ],
              ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(utils.findIcon(b[1])))
        a = toolbar.addAction(icon,b[0],b[2])
        a.setEnabled(b[3])
        action[b[0]] = a
    return action

################### Script action toolbar ###########
## def addRenderButtons(toolbar):
##     """Add the rendermode buttons to the toolbar."""
##     action = {}
##     buttons = [ [ "Wireframe", "wireframe", draw.wireframe, True ],
##                 [ "Smooth", "smooth", draw.smooth, True ],
##                 [ "Flat", "flat", draw.flat, True ],
##               ]
##     for b in buttons:
##         icon = QtGui.QIcon(QtGui.QPixmap(utils.findIcon(b[1])))
##         a = toolbar.addAction(icon,b[0],b[2])
##         a.setEnabled(b[3])
##         action[b[0]] = a
##     return action
    

################# Transparency Button ###############

toggle_transparency = None # the toggle transparency button

def toggleTransparency():
    global toggle_transparency
    mode = not GD.canvas.alphablend
    draw.transparency(mode)
    toggle_transparency.setChecked(mode)

################# Camera action toolbar ###############

toggle_perspective = None # the toggle perspective button

def togglePerspective():
    global toggle_perspective
    mode = not GD.canvas.camera.perspective
    cameraMenu.setPerspective(mode)
    toggle_perspective.setChecked(mode)


def addCameraButtons(toolbar):
    """Add the camera buttons to a toolbar."""
    global toggle_perspective, toggle_transparency
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
        icon = QtGui.QIcon(QtGui.QPixmap(utils.findIcon(but[1])))
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
    icon_on = QtGui.QPixmap(utils.findIcon('perspect'))
    icon_off = QtGui.QPixmap(utils.findIcon('project'))
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

    # Add the transparency button
    icon = QtGui.QIcon(QtGui.QPixmap(utils.findIcon('transparent')))    
    a = toolbar.addAction(icon,'Toggle Transparent Mode', toggleTransparency)
    b = toolbar.children()[-1]
    b.setCheckable(True)
    b.setChecked(False)
    toggle_transparency = b

# End
