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
"""Viewport Menu."""

import globaldata as GD
import canvas
import widgets
import draw
from gettext import gettext as _


def addViewport():
    """Add a new viewport."""
    n = len(GD.gui.viewports.all)
    if n < 4:
        GD.gui.viewports.addView(n/2,n%2)

def removeViewport():
    """Remove a new viewport."""
    n = len(GD.gui.viewports.all)
    if n > 1:
        GD.gui.viewports.removeView()


def setRenderMode():
    """Change the rendering mode."""
    mode = GD.canvas.rendermode
    modes = canvas.Canvas.rendermodes
    itemlist = [('Render Mode', modes, 'select', mode)]
    res,accept = widgets.InputDialog(itemlist,'Select Render Mode').getResult()
    if accept:
        GD.canvas.setRenderMode(res['Render Mode'])
        
def setBgColor():
    """Change the background color."""
    color = GD.canvas.settings.bgcolor
    color = widgets.getColor(color)
    if color:
        GD.canvas.setBgColor(color)
        
def setFgColor():
    """Change the default drawing color."""
    color = GD.canvas.settings.fgcolor
    color = widgets.getColor(color)
    if color:
        GD.canvas.setFgColor(color)
        
def setLineWidth():
    """Change the default line width."""
    lw = GD.canvas.settings.linewidth
    itemlist = [('Line Width', lw, 'float')]
    res,accept = widgets.InputDialog(itemlist,'Choose default line width').getResult()
    if accept:
        GD.canvas.setLineWidth(res['Line Width'])
    

def viewportSettings():
    """Interactively set the viewport settings."""
    mode = GD.canvas.rendermode
    modes = canvas.Canvas.rendermodes
    print modes
    s = GD.canvas.settings
    itemlist = [('rendermode', modes, 'select', mode),
                ('bgcolor', s.bgcolor, 'color'),
                ('fgcolor', s.fgcolor, 'color'),
                ('slcolor', s.slcolor, 'color'),
                ('linewidth', s.linewidth, 'float'),
                ('Store these settings as defaults', False),
                ]
    res,accept = widgets.InputDialog(itemlist,'Config Dialog').getResult()
    if accept:
        GD.debug(res)
        s.reset(res)
        GD.canvas.setRenderMode(res['rendermode'])
        GD.canvas.redrawAll()
        if res['Store these settings as defaults']:
            GD.cfg.update(GD.canvas.settings.__dict__,name='canvas')
    

MenuData = [
    (_('&Viewport'),[
        (_('&Clear'),draw.clear),
        (_('Toggle &Triade'),draw.setTriade), 
#        (_('&Transparency'),setOpacity), 
        (_('&Background Color'),setBgColor), 
        (_('&Foreground Color'),setFgColor), 
        (_('Line&Width'),setLineWidth), 
        (_('&All Viewport Settings'),viewportSettings), 
        (_('&Redraw'),draw.redraw),
        (_('&Reset'),draw.reset),
        (_('&Add new viewport'),addViewport), 
        (_('&Remove last viewport'),removeViewport), 
        ]),
    ]

    
# End
