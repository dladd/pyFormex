# $Id$
##
##  This file is part of pyFormex 0.8.1 Release Wed Dec  9 11:27:53 2009
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
"""Viewport Menu."""

import pyformex as GD
import canvas
import widgets
import draw
from gettext import gettext as _

        

def set_near_clip(v):
    dist = GD.canvas.camera.getDist()
    GD.canvas.camera.setClip(10**v*dist,100.*dist)
    GD.canvas.update()
    
    
def setClip():
    items = [
        ('near',-2.0,'slider',{'min':-100,'max':100,'scale':0.01,'func': set_near_clip}),
        ]
    res = draw.askItems(items)
    ## if res:
    ##     updateSettings(res,GD.cfg)


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


_the_dialog = None

def set_the_color(field='top'):
    """Set the_color from a dialog"""
    _the_dialog.acceptData()
    color = _the_dialog.results[field]
    color = widgets.getColor(color)
    #print(color)
    if color:
        _the_dialog.updateData({field:color})


def set_the_color_top():
    set_the_color('top')
def set_the_color_bottom():
    set_the_color('bottom')
                

def setBgColor():
    """Change the background color."""
    color = GD.canvas.settings.bgcolor
    color = widgets.getColor(color)
    if color:
        GD.canvas.setBgColor(color)


def setBgColor2():
    """Interactively set the viewport background colors."""
    global _the_dialog
    color = GD.canvas.settings.bgcolor
    color2 = GD.canvas.settings.bgcolor2
    if color2 is None:
        color2 = color
    itemlist = [('top',color,'color',{'text':'Top background color'}),
                ('bottom',color2,'color',{'text':'Bottom background color'}),
                ]
    _the_dialog = widgets.InputDialog(itemlist,'Config Dialog')
    res = _the_dialog.getResult()
    GD.debug(res)
    if res:
        GD.canvas.setBgColor(res['top'],res['bottom'])
        GD.canvas.update()
    _the_dialog = None

        
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
    s = GD.canvas.settings
    if s.bgcolor2 is None:
        s.bgcolor2 = s.bgcolor
    itemlist = [('rendermode', mode, 'select', modes),
                ('linewidth', s.linewidth, 'float'),
                ('bgcolor', s.bgcolor, 'color'),
                ('bgcolor2', s.bgcolor2, 'color'),
                ('fgcolor', s.fgcolor, 'color'),
                ('slcolor', s.slcolor, 'color'),
                ('Store these settings as defaults', False),
                ]
    res = widgets.InputDialog(itemlist,'Config Dialog').getResult()
    if res:
        GD.debug(res)
        GD.canvas.updateSettings(res)
        GD.canvas.setRenderMode(res['rendermode'])
        #GD.canvas.clear()
        GD.canvas.redrawAll()
        GD.canvas.update()
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


def openglSettings():
    dia = None
    def apply():
        dia.acceptData()
        canvas.glSettings(dia.results)
    def close():
        dia.close()
        
    dia = widgets.InputDialog(
        caption='OpenGL Settings',
        items=[
            ('Line Smoothing','Off','radio',{'choices':['On','Off']}),
            ('Polygon Mode',None,'radio',{'choices':['Fill','Line']}),
            ('Polygon Fill',None,'radio',{'choices':['Front and Back','Front','Back']}),
            ('Culling','Off','radio',{'choices':['On','Off']}),
# These are currently set by the render mode
#            ('Shading',None,'radio',{'choices':['Smooth','Flat']}),
#            ('Lighting',None,'radio',{'choices':['On','Off']}),
            ],
        actions=[('Done',close),('Apply',apply)]
        )
    dia.show()

def lineSmoothOn():
    canvas.glLineSmooth(True)

def lineSmoothOff():
    canvas.glLineSmooth(False)

MenuData = [
    (_('&Viewport'),[
        (_('&Clear'),draw.clear),
        (_('Toggle &Axes Triade'),draw.setTriade), 
        (_('Set &Axes Triade Properties'),setTriadeParams), 
        (_('Set Near and Far Clipping Planes'),setClip), 
#        (_('&Transparency'),setOpacity), 
        (_('&Background Color'),setBgColor), 
        (_('&Background 2Color'),setBgColor2), 
        (_('&Foreground Color'),setFgColor), 
        (_('&Highlight Color'),setSlColor), 
        (_('Line&Width'),setLineWidth), 
        (_('&Canvas Size'),setCanvasSize), 
        (_('&All Viewport Settings'),viewportSettings),
        (_('&Global Draw Options'),draw.askDrawOptions),
        (_('&OpenGL Settings'),openglSettings),
        ## ('&OpenGL Settings',
        ##  [('&Flat',canvas.glFlat),
        ##   ('&Smooth',canvas.glSmooth),
        ##   ('&Culling',canvas.glCulling),
        ##   ('&No Culling',canvas.glNoCulling),
        ##   ('&Line Smoothing On',lineSmoothOn),
        ##   ('&Line Smoothing Off',lineSmoothOff),
        ##   ('&Polygon Line',canvas.glLine),
        ##   ('&Polygon Fill',canvas.glFill),
        ##   ('&Polygon Front Fill',canvas.glFrontFill),
        ##   ('&Polygon Back Fill',canvas.glBackFill),
        ##   ('&Polygon Front and Back Fill',canvas.glBothFill),
        ##   ]),
        (_('&Redraw'),draw.redraw),
        (_('&Reset'),draw.reset),
        (_('&Change viewport layout'),viewportLayout), 
        (_('&Add new viewport'),draw.addViewport), 
        (_('&Remove last viewport'),draw.removeViewport), 
        ]),
    ]

    
# End
