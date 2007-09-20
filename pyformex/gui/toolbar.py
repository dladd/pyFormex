#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
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

import fileMenu
import scriptsMenu
import cameraMenu
import widgets
import draw
import utils


################### Script action toolbar ###########
def addActionButtons(toolbar):
    """Add the script action buttons to the toolbar."""
    action = {}
    buttons = [ [ "Play", "next", draw.play, False ],
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

################### General Button Functions ###########

def addButton(toolbar,text,icon,func,repeat=False,toggle=False,checked=False,icon0=None):
    """Add a button to a toolbar.

    toolbar is where the button will be added, 
    text appears as tooltip,
    icon is the name of the icon to be displayed on the button,
    func is called when button is pressed,

    repeat == True: func will repeatedly be called if button is held down.
    toggle == True: button is a toggle (stays in pressed state).
    If the button is a toggle, checked is the initial state and icon1 may
    specify an icon that will be displayed when button is not checked.
    """
    iconset = QtGui.QIcon()
    icon_on = QtGui.QPixmap(utils.findIcon(icon))
    iconset.addPixmap(icon_on,QtGui.QIcon.Normal,QtGui.QIcon.On)
    if toggle and icon0:
        icon_off = QtGui.QPixmap(utils.findIcon(icon0))
        iconset.addPixmap(icon_off,QtGui.QIcon.Normal,QtGui.QIcon.Off)
                                 
    a = toolbar.addAction(iconset,text,func)
    b =  toolbar.children()[-1] # Get the QToolButton for the last action

    if repeat:
        b.setAutoRepeat(True)
        QtCore.QObject.connect(b,QtCore.SIGNAL("clicked()"),a,QtCore.SLOT("trigger()"))

    if toggle:
        b.setCheckable(True)
        b.connect(b,QtCore.SIGNAL("clicked()"),QtCore.SLOT("toggle()"))
        b.setChecked(checked)

    b.setToolTip(text)

    return b
    


################# Camera action toolbar ###############

def addCameraButtons(toolbar):
    """Add the camera buttons to a toolbar."""
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

# We should probably make a general framework for toggle buttons ?

################# Transparency Button ###############

transparency_button = None # the toggle transparency button

def toggleTransparency(): # Called by the button
    mode = not GD.canvas.alphablend
    GD.canvas.setTransparency(mode)
    GD.canvas.update()
    GD.app.processEvents()

def addTransparencyButton(toolbar):
    global transparency_button
    transparency_button = addButton(toolbar,'Toggle Transparent Mode',
                                    'transparent',toggleTransparency,
                                    toggle=True)    

def setTransparency(mode):
    GD.canvas.setTransparency(mode)
    GD.canvas.update()
    if transparency_button:
        transparency_button.setChecked(mode)
    GD.app.processEvents()
  

################# Perspective Button ###############

perspective_button = None # the toggle perspective button

def togglePerspective(): # Called by the button
    mode = not GD.canvas.camera.perspective
    #setPerspective(mode)
    GD.canvas.camera.setPerspective(mode)
    GD.canvas.display()
    GD.canvas.update()
    GD.app.processEvents()

def addPerspectiveButton(toolbar):
    global perspective_button
    perspective_button = addButton(toolbar,'Toggle Perspective/Projective Mode',
                                   'perspect',togglePerspective,
                                   toggle=True,icon0='project',checked=True)    
def setPerspective(mode=True):
    GD.canvas.camera.setPerspective(mode)
    GD.canvas.display()
    GD.canvas.update()
    if perspective_button:
        perspective_button.setChecked(mode)
    GD.app.processEvents()

def setProjection():
    setPerspective(False)

################# Timeout Button ###############

timeout_button = None # the toggle timeout button

def toggleTimeout(onoff=None):
    #print "TOGGLE TIMEOUT"
    #print widgets.input_timeout
    if onoff is None:
        onoff = widgets.input_timeout < 0
    if onoff:
        widgets.input_timeout = GD.cfg.get('input/timeout',-1)
    else:
        widgets.input_timeout = -1
    #print widgets.input_timeout
    onoff = widgets.input_timeout > 0
    #print "Button should be checked: %s" % onoff 
    return onoff


def addTimeoutButton(toolbar):
    global timeout_button
    timeout_button = addButton(toolbar,'Toggle Timeout','clock',toggleTimeout,
                               toggle=True,checked=False)

def timeout(onoff=None):
    timeout_button.setChecked( toggleTimeout(onoff) )

# End
