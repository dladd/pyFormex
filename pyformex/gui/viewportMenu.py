# $Id$
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

import pyformex as pf
import canvas
import widgets
import draw
from gettext import gettext as _

from widgets import simpleInputItem as _I


def setTriade():
    try:
        pos = pf.canvas.triade.pos
        siz = pf.canvas.triade.siz
    except:
        pos = 'lb'
        siz = 100
    res = draw.askItems([
        ('triade',True),
        ('pos',pos,'select',{'choices':['lt','lc','lb','ct','cc','cb','rt','rc','rb']}),
        ('size',siz),
        ])
    if res:
        draw.setTriade(res['triade'],res['pos'],res['size'])


def setBgColor():
    """Interactively set the viewport background colors."""
    from gui.drawable import saneColorArray
    from numpy import resize
    import os
    bgmodes = [ 'solid', 'vertical', 'horizontal', 'full' ]
    color = saneColorArray(pf.canvas.settings.bgcolor,(4,))
    color = resize(color,(4,3))
    cur = pf.canvas.settings.bgimage
    showimage = os.path.exists(cur)
    if not showimage:
        cur = pf.cfg['gui/splash']
    viewer = widgets.ImageView(cur,maxheight=200)
    def changeImage(fn):
        fn = draw.askImageFile(fn)
        if fn:
            viewer.showImage(fn)
        return fn
    dialog = widgets.InputDialog(
        [
            _I('mode',choices=bgmodes),
            _I('color1',color[0],itemtype='color',text='Background color 1 (Bottom Left)'),
            _I('color2',color[1],itemtype='color',text='Background color 2 (Bottom Right)'),
            _I('color3',color[2],itemtype='color',text='Background color 3 (Top Right)'),
            _I('color4',color[3],itemtype='color',text='Background color 4 (Top Left'),
            _I('showimage',showimage,text='Show background image'),
            _I('image',cur,text='Background image',itemtype='button',func=changeImage),
            viewer,
            ],
        caption='Config Dialog',
        enablers=[
            ('mode','vertical','color4'),
            ('mode','horizontal','color2'),
            ('mode','full','color2','color3','color4'),
            ('showimage',True,'image'),
            ]
        )
    res = dialog.getResult()
    pf.debug(res)
    if res:
        if res['mode'] == 'solid':
            color = res['color1']
        elif res['mode'] == 'vertical':
            c1,c4 = res['color1'],res['color4']
            color = [c1,c1,c4,c4]
        elif res['mode'] == 'horizontal':
            c1,c2 = res['color1'],res['color2']
            color = [c1,c2,c2,c1]
        elif res['mode'] == 'full':
            color = [res['color1'],res['color2'],res['color3'],res['color4']]
        if res['showimage']:
            image = res['image']
        else:
            image = None
        pf.canvas.setBackground(color=color,image=image)
        pf.canvas.update()

        
def setFgColor():
    """Change the default drawing color."""
    color = pf.canvas.settings.fgcolor
    color = widgets.getColor(color)
    if color:
        pf.canvas.setFgColor(color)

        
def setSlColor():
    """Change the highlighting color."""
    color = pf.canvas.settings.slcolor
    color = widgets.getColor(color)
    if color:
        pf.canvas.setSlColor(color)


        
def setLineWidth():
    """Change the default line width."""
    res = draw.askItems(
        [_I('Line Width',pf.canvas.settings.linewidth)],
        'Choose default line width'
        )
    if res:
        pf.canvas.setLineWidth(res['Line Width'])

    
def setCanvasSize():
    """Save the current viewport size"""
    res = draw.askItems(
        [_I('w',pf.canvas.width()),_I('h',pf.canvas.height())],
        'Set Canvas Size'
        )
    if res:
        pf.canvas.resize(int(res['w']),int(res['h']))


def viewportSettings():
    """Interactively set the viewport settings."""
    mode = pf.canvas.rendermode
    modes = canvas.Canvas.rendermodes
    s = pf.canvas.settings
    if s.bgcolor2 is None:
        s.bgcolor2 = s.bgcolor
    res = draw.askItems(
        [_I('rendermode', mode, choices=modes),
         _I('linewidth', s.linewidth, itemtype='float'),
         _I('bgcolor', s.bgcolor, itemtype='color'),
         _I('bgcolor2', s.bgcolor2, itemtype='color'),
         _I('fgcolor', s.fgcolor, itemtype='color'),
         _I('slcolor', s.slcolor, itemtype='color'),
         _I('Store these settings as defaults', False),
         ],
        'Config Dialog')
    if res:
        pf.debug(res)
        pf.canvas.setRenderMode(res['rendermode'])
        pf.canvas.settings.update(res,strict=False)
        #pf.canvas.clear()
        pf.canvas.redrawAll()
        pf.canvas.update()
        if res['Store these settings as defaults']:
            pf.cfg.update(pf.canvas.settings.__dict__,name='canvas')
        

def viewportLayout():
    """Set the viewport layout."""
    directions = [ 'rowwise','columnwise' ]
    if pf.GUI.viewports.rowwise:
        current = directions[0]
    else:
        current = directions[1]
    res = draw.askItems(
        [_I('Number of viewports',len(pf.GUI.viewports.all)),
         _I('Viewport layout direction',current,choices=directions),
         _I('Number of viewports per row/column',pf.GUI.viewports.ncols),
         ],
        'Config Dialog')
    if res:
        pf.debug(res)
        nvps = res['Number of viewports']
        rowwise = res['Viewport layout direction'] == 'rowwise'
        ncols = res['Number of viewports per row/column']
        if rowwise:
            nrows = None
        else:
            nrows = ncols
            ncols = None
        pf.GUI.viewports.changeLayout(nvps,ncols,nrows)
#        if res['Store these settings as defaults']:
#            pf.cfg.update()



def cameraSettings():
    from plugins import cameratools
    cameratools.showCameraTool()


def openglSettings():
    dia = None
    def apply_():
        dia.acceptData()
        canvas.glSettings(dia.results)
    def close():
        dia.close()
        
    dia = widgets.InputDialog(
        caption='OpenGL Settings',
        items=[
            _I('Line Smoothing','Off',itemtype='radio',choices=['On','Off']),
            _I('Polygon Mode',None,itemtype='radio',choices=['Fill','Line']),
            _I('Polygon Fill',None,itemtype='radio',choices=['Front and Back','Front','Back']),
            _I('Culling','Off',itemtype='radio',choices=['On','Off']),
# These are currently set by the render mode
#            ('Shading',None,'radio',{'choices':['Smooth','Flat']}),
#            ('Lighting',None,'radio',{'choices':['On','Off']}),
            ],
        actions=[('Done',close),('Apply',apply_)]
        )
    dia.show()

def lineSmoothOn():
    canvas.glLineSmooth(True)

def lineSmoothOff():
    canvas.glLineSmooth(False)

def singleViewport():
    draw.layout(1)

def clearAll():
    for vp in pf.GUI.viewports.all:
        vp.removeAny()
        vp.clear()
        vp.update()
    pf.GUI.processEvents()


MenuData = [
    (_('&Viewport'),[
        (_('&Clear'),draw.clear),
        (_('&Clear All'),clearAll),
        (_('&Axes Triade'),setTriade), 
#        (_('&Transparency'),setOpacity), 
        (_('&Background Color'),setBgColor), 
        (_('&Foreground Color'),setFgColor), 
        (_('&Highlight Color'),setSlColor), 
        (_('Line&Width'),setLineWidth), 
        (_('&Canvas Size'),setCanvasSize), 
        (_('&All Viewport Settings'),viewportSettings),
        (_('&Global Draw Options'),draw.askDrawOptions),
        (_('&Camera Settings'),cameraSettings),
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
        (_('&Reset viewport'),draw.reset),
        (_('&Reset layout'),singleViewport),
        (_('&Change viewport layout'),viewportLayout), 
        (_('&Add new viewport'),draw.addViewport), 
        (_('&Remove last viewport'),draw.removeViewport), 
        ]),
    ]

    
# End
