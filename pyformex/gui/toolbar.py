# $Id$
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Toolbars for the pyFormex GUI.

This module defines the functions for creating the pyFormex window toolbars.
"""

import pyformex as GD
import os
from PyQt4 import QtCore, QtGui

import widgets
import draw
import utils

 
################### Script action toolbar ###########
def addActionButtons(toolbar):
    """Add the script action buttons to the toolbar."""
    action = {}
    buttons = [ ( "Play", "next", draw.play, False ),
                ( "Step", "nextstop", draw.step, False ),
                ( "Continue", "ff", draw.fforward, False ),
                ( "Stop", "stop", draw.stopatbreakpt, False ),
#                ( "Stop", "stop", draw.raiseExit, False ),
                ]
    for b in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(utils.findIcon(b[1])))
        a = toolbar.addAction(icon,b[0],b[2])
        a.setEnabled(b[3])
        action[b[0]] = a
    return action

################### General Button Functions ###########

def addButton(toolbar,tooltip,icon,func,repeat=False,toggle=False,checked=False,icon0=None):
    """Add a button to a toolbar.

    - `toolbar`:  the toolbar where the button will be added
    - `tooltip`: the text to appears as tooltip
    - `icon`: name of the icon to be displayed on the button,
    - `func`: function to be called when the button is pressed,
    - `repeat`: if True, the `func` will repeatedly be called if button is
      held down.
    - `toggle`: if True, the button is a toggle and stays in depressed state
      until pressed again.
    - `checked`: initial state for a toggle buton.
    - `icon1`: for a toggle button, icon to display when button is not checked.
    """
    iconset = QtGui.QIcon()
    icon_on = QtGui.QPixmap(utils.findIcon(icon))
    iconset.addPixmap(icon_on,QtGui.QIcon.Normal,QtGui.QIcon.On)
    if toggle and icon0:
        icon_off = QtGui.QPixmap(utils.findIcon(icon0))
        iconset.addPixmap(icon_off,QtGui.QIcon.Normal,QtGui.QIcon.Off)
                                 
    a = toolbar.addAction(iconset,tooltip,func)
    b = toolbar.widgetForAction(a)
    
    if repeat:
        b.setAutoRepeat(True)
        b.setAutoRepeatDelay(500)
        QtCore.QObject.connect(b,QtCore.SIGNAL("clicked()"),a,QtCore.SLOT("trigger()"))

    if toggle:
        b.setCheckable(True)
        b.connect(b,QtCore.SIGNAL("clicked()"),QtCore.SLOT("toggle()"))
        b.setChecked(checked)

    b.setToolTip(tooltip)

    return b


def removeButton(toolbar,button):
    """Remove a button from a toolbar."""
    toolbar.removeAction(button)


################# Camera action toolbar ###############


def addCameraButtons(toolbar):
    """Add the camera buttons to a toolbar."""
    # The buttons have the following fields:
    #  0 : tooltip
    #  1 : icon
    #  2 : function
    # optional:
    #  3 : REPEAT  (default True)
    import cameraMenu
    buttons = [ [ "Rotate left", "rotleft", cameraMenu.rotLeft ],
                [ "Rotate right", "rotright", cameraMenu.rotRight ],
                [ "Rotate up", "rotup", cameraMenu.rotUp ],
                [ "Rotate down", "rotdown", cameraMenu.rotDown ],
                [ "Twist left", "twistleft", cameraMenu.twistLeft ],
                [ "Twist right", "twistright", cameraMenu.twistRight ],
                [ "Translate left", "left", cameraMenu.panLeft ],
                [ "Translate right", "right", cameraMenu.panRight ],
                [ "Translate down", "down", cameraMenu.panDown ],
                [ "Translate up", "up", cameraMenu.panUp ],
                [ "Zoom Out", "zoomout", cameraMenu.dollyOut ],
                [ "Zoom In", "zoomin", cameraMenu.dollyIn ],
                [ "Zoom Rectangle", "zoomrect", draw.zoomRectangle, False ],
                [ "Zoom All", "zoomall", draw.zoomAll, False ],
                ]
    for but in buttons:
        icon = QtGui.QIcon(QtGui.QPixmap(utils.findIcon(but[1])))
        a = toolbar.addAction(icon,but[0],but[2])
        b = toolbar.children()[-1] # Get the QToolButton for the last action
        if len(but) < 4 or but[3]:
            b.setAutoRepeat(True)
            b.setAutoRepeatDelay(500)
            QtCore.QObject.connect(b,QtCore.SIGNAL("released()"),a,QtCore.SLOT("trigger()"))
        if len(but) >= 5:
            b.setCheckable(but[4])
            b.connect(b,QtCore.SIGNAL("released()"),QtCore.SLOT("toggle()"))
            
        b.setToolTip(but[0])


# We should probably make a general framework for toggle buttons ?

################# Transparency Button ###############

transparency_button = None # the toggle transparency button

def toggleTransparency(): # Called by the button, not by user
    mode = not GD.canvas.alphablend
    GD.canvas.setTransparency(mode)
    GD.canvas.update()
    GD.app.processEvents()

def addTransparencyButton(toolbar):
    global transparency_button
    transparency_button = addButton(toolbar,'Toggle Transparent Mode',
                                    'transparent',toggleTransparency,
                                    toggle=True)    

def setTransparency(mode=True):
    """Set the transparency mode on or off."""
    GD.canvas.setTransparency(mode)
    GD.canvas.update()
    if transparency_button:
        transparency_button.setChecked(mode)
    GD.app.processEvents()
  

################# Lights Button ###############

light_button = None

def toggleLight(): 
    mode = not GD.canvas.lighting
    GD.canvas.setLighting(mode)
    #GD.canvas.display()
    GD.canvas.update()
    GD.app.processEvents()

def addLightButton(toolbar):
    global light_button
    light_button = addButton(toolbar,'Toggle Lights',
                             'lamp-on',toggleLight,icon0='lamp',
                             toggle=True,checked=True)    

def setLight(mode=True):
    """Set the lights mode on or off."""
    GD.canvas.setLighting(mode)
    GD.canvas.update()
    if light_button:
        light_button.setChecked(mode)
    GD.app.processEvents()
  

################# Normals Button ###############

normals_button = None

def toggleNormals():
    state = not GD.canvas.avgnormals
    GD.canvas.setAveragedNormals(state)
    GD.canvas.update()
    GD.app.processEvents()

def addNormalsButton(toolbar):
    global normals_button
    normals_button = addButton(toolbar,'Toggle Normals',
                               'normals-avg',toggleNormals,icon0='normals-ind',
                               toggle=True,checked=False)    

def setNormals(state=True):
    """Set the normals mode to averaged or not."""
    GD.canvas.setAveragedNormals(state)
    GD.canvas.update()
    if normals_button:
        normals_button.setChecked(state)
    GD.app.processEvents()


################# Perspective Button ###############

perspective_button = None # the toggle perspective button

def togglePerspective(): # Called by the button, not by user
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
    """Set the perspective mode on or off."""
    GD.canvas.camera.setPerspective(mode)
    GD.canvas.display()
    GD.canvas.update()
    if perspective_button:
        perspective_button.setChecked(mode)
    GD.app.processEvents()

def setProjection():
    setPerspective(False)

################# Shrink Button ###############

#
# The shrink button currently does not redraw, it only sets the default
# shrink factor and clears the viewport
#
shrink_button = None # the toggle shrink button

def toggleShrink(): # Called by the button
    mode = draw.DrawOptions.get('shrink',None)
    if mode is None:
        mode = 0.8
    else:
        mode = None
    draw.shrink(mode)

def addShrinkButton(toolbar):
    global shrink_button
    shrink_button = addButton(toolbar,'Toggle Shrink Mode',
                              'shrink',toggleShrink,
                              toggle=True)    

def setShrink(mode):
    draw.shrink(mode)
    if shrink_button:
        shrink_button.setChecked(mode != None)

################# Timeout Button ###############

timeout_button = None # the timeout toggle button

def toggleTimeout(onoff=None):
    if onoff is None:
        onoff = widgets.input_timeout < 0
    if onoff:
        timeout = GD.cfg.get('gui/timeoutvalue',-1)
    else:
        timeout = -1

    widgets.setInputTimeout(timeout)
    onoff = widgets.input_timeout > 0
    if onoff:
        # THIS SUSPENDS ALL WAITING! WE SHOULD IMPLEMENT A TIMEOUT!
        # BY FORCING ALL INDEFINITE PAUSES TO A WAIT TIME EQUAL TO
        # WIDGET INPUT TIMEOUT
        GD.debug("FREEING the draw lock")
        GD.GUI.drawlock.free()
    else:
        GD.debug("ALLOWING the draw lock")
        GD.GUI.drawlock.allow()
    return onoff


def addTimeoutButton(toolbar):
    """Add or remove the timeout button,depending on cfg."""
    global timeout_button
    if GD.cfg['gui/timeoutbutton']:
        if timeout_button is None:
            timeout_button = addButton(toolbar,'Toggle Timeout','clock',toggleTimeout,toggle=True,checked=False)
    else:
        if timeout_button is not None:
            removeButton(toolbar,timeout_button)
            timeout_button = None
            

def timeout(onoff=None):
    """Programmatically toggle the timeout button"""
    if timeout_button is not None:
        timeout_button.setChecked(toggleTimeout(onoff))



# End
