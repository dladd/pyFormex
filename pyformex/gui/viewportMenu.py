# $Id$
##
## This file is part of pyFormex 0.6 Release Fri Nov 16 22:39:28 2007
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


def setRenderMode():
    """Change the rendering mode."""
    mode = GD.canvas.rendermode
    modes = canvas.Canvas.rendermodes
    itemlist = [('Render Mode', mode, 'select', modes)]
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
    
def setCanvasSize():
    """Save the current viewport size"""
    itemlist = [('w',GD.canvas.width()),('h',GD.canvas.height())]
    res,accept = widgets.InputDialog(itemlist,'Set Canvas Size').getResult()
    if accept:
        GD.canvas.resize(int(res['w']),int(res['h']))





def viewportSettings():
    """Interactively set the viewport settings."""
    mode = GD.canvas.rendermode
    modes = canvas.Canvas.rendermodes
    print modes
    s = GD.canvas.settings
    itemlist = [('rendermode', mode, 'select', modes),
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


def viewportLayout():
    """Set the viewport layout."""
    directions = [ 'rowwise','columnwise' ]
    if GD.gui.viewports.rowwise:
        current = directions[0]
    else:
        current = directions[1]
    itemlist = [('Number of viewports',len(GD.gui.viewports.all)),
                ('Viewport layout direction',current,'select',directions),
                ('Number of viewports per row/column',GD.gui.viewports.ncols),
                ]
    res,accept = widgets.InputDialog(itemlist,'Config Dialog').getResult()
    if accept:
        GD.debug(res)
        nvps = res['Number of viewports']
        rowwise = res['Viewport layout direction'] == 'rowwise'
        ncols = res['Number of viewports per row/column']
        if rowwise:
            nrows = None
        else:
            nrows = ncols
            ncols = None
        GD.gui.viewports.changeLayout(nvps,ncols,nrows)
#        if res['Store these settings as defaults']:
#            GD.cfg.update()

MenuData = [
    (_('&Viewport'),[
        (_('&Clear'),draw.clear),
        (_('Toggle &Triade'),draw.setTriade), 
#        (_('&Transparency'),setOpacity), 
        (_('&Background Color'),setBgColor), 
        (_('&Foreground Color'),setFgColor), 
        (_('Line&Width'),setLineWidth), 
        (_('&Canvas Size'),setCanvasSize), 
        (_('&All Viewport Settings'),viewportSettings),
        ('&OpenGL Settings',
         [('&Flat',canvas.glFlat),
          ('&Smooth',canvas.glSmooth),
          ('&Culling',canvas.glCulling),
          ('&No Culling',canvas.glNoCulling),
          ('&Polygon Line',canvas.glLine),
          ('&Polygon Fill',canvas.glFill),
          ('&Polygon Front Fill',canvas.glFrontFill),
          ('&Polygon Back Fill',canvas.glBackFill),
          ('&Polygon Front and Back Fill',canvas.glBothFill),
          ]),
        (_('&Redraw'),draw.redraw),
        (_('&Reset'),draw.reset),
        (_('&Change viewport layout'),viewportLayout), 
        (_('&Add new viewport'),draw.addViewport), 
        (_('&Remove last viewport'),draw.removeViewport), 
        ]),
    ]

    
# End
