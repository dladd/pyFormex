# $Id$
##
##  This file is part of pyFormex 0.8 Release Sat Jun 13 10:22:42 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##
"""Viewport Menu."""

import pyformex as GD
import canvas
import widgets
import draw
from gettext import gettext as _


def setTriadeParams():
    try:
        size = GD.canvas.triade.size
        pos = GD.canvas.triade.pos.tolist()
    except:
        size = 1.0
        pos = [0.,0.,0.]
    res = draw.askItems([('size',size),('pos',pos)])
    if res:
        draw.setTriade(True,res['size'],res['pos'])
        

def setRenderMode():
    """Change the rendering mode."""
    mode = GD.canvas.rendermode
    modes = canvas.Canvas.rendermodes
    itemlist = [('Render Mode', mode, 'select', modes)]
    res = widgets.InputDialog(itemlist,'Select Render Mode').getResult()
    if res:
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

        
def setSlColor():
    """Change the highlighting color."""
    color = GD.canvas.settings.slcolor
    color = widgets.getColor(color)
    if color:
        GD.canvas.setSlColor(color)


        
def setLineWidth():
    """Change the default line width."""
    lw = GD.canvas.settings.linewidth
    itemlist = [('Line Width', lw, 'float')]
    res = widgets.InputDialog(itemlist,'Choose default line width').getResult()
    if res:
        GD.canvas.setLineWidth(res['Line Width'])
    
def setCanvasSize():
    """Save the current viewport size"""
    itemlist = [('w',GD.canvas.width()),('h',GD.canvas.height())]
    res = widgets.InputDialog(itemlist,'Set Canvas Size').getResult()
    if res:
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
    res = widgets.InputDialog(itemlist,'Config Dialog').getResult()
    if res:
        GD.debug(res)
        s.reset(res)
        GD.canvas.setRenderMode(res['rendermode'])
        GD.canvas.redrawAll()
        if res['Store these settings as defaults']:
            GD.cfg.update(GD.canvas.settings.__dict__,name='canvas')


def viewportLayout():
    """Set the viewport layout."""
    directions = [ 'rowwise','columnwise' ]
    if GD.GUI.viewports.rowwise:
        current = directions[0]
    else:
        current = directions[1]
    itemlist = [('Number of viewports',len(GD.GUI.viewports.all)),
                ('Viewport layout direction',current,'select',directions),
                ('Number of viewports per row/column',GD.GUI.viewports.ncols),
                ]
    res = widgets.InputDialog(itemlist,'Config Dialog').getResult()
    if res:
        GD.debug(res)
        nvps = res['Number of viewports']
        rowwise = res['Viewport layout direction'] == 'rowwise'
        ncols = res['Number of viewports per row/column']
        if rowwise:
            nrows = None
        else:
            nrows = ncols
            ncols = None
        GD.GUI.viewports.changeLayout(nvps,ncols,nrows)
#        if res['Store these settings as defaults']:
#            GD.cfg.update()

MenuData = [
    (_('&Viewport'),[
        (_('&Clear'),draw.clear),
        (_('Toggle &Axes Triade'),draw.setTriade), 
        (_('Set &Axes Triade Properties'),setTriadeParams), 
#        (_('&Transparency'),setOpacity), 
        (_('&Background Color'),setBgColor), 
        (_('&Foreground Color'),setFgColor), 
        (_('&Highlight Color'),setSlColor), 
        (_('Line&Width'),setLineWidth), 
        (_('&Canvas Size'),setCanvasSize), 
        (_('&All Viewport Settings'),viewportSettings),
        (_('&Global Draw Options'),draw.askDrawOptions),
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
