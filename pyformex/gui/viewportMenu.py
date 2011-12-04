# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 15:52:41 CET 2011)
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

from widgets import simpleInputItem as _I, compatInputItem as _C


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
    mode = pf.canvas.settings.bgmode
    color = pf.canvas.settings.bgcolor
    color2 = pf.canvas.settings.bgcolor2
    dialog = widgets.InputDialog([
        _I('mode',mode,choices=pf.canvas.bgmodes),
        _I('bgcolor',color,itemtype='color',text='Solid/Top/Left background color'),
        _I('bgcolor2',color2,itemtype='color',text='Bottom/Right background color'),
        ],caption='Config Dialog',enablers=[('mode','vertical gradient','bgcolor2'),('mode','horizontal gradient','bgcolor2')])
    res = dialog.getResult()
    pf.debug(res)
    if res:
        pf.canvas.setBgColor(res['bgcolor'],res['bgcolor2'],res['mode'])
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
    lw = pf.canvas.settings.linewidth
    itemlist = [_C('Line Width', lw, 'float')]
    res = widgets.InputDialog(itemlist,'Choose default line width').getResult()
    if res:
        pf.canvas.setLineWidth(res['Line Width'])

    
def setCanvasSize():
    """Save the current viewport size"""
    itemlist = [_I('w',pf.canvas.width()),_I('h',pf.canvas.height())]
    res = widgets.InputDialog(itemlist,'Set Canvas Size').getResult()
    if res:
        pf.canvas.resize(int(res['w']),int(res['h']))


def viewportSettings():
    """Interactively set the viewport settings."""
    mode = pf.canvas.rendermode
    modes = canvas.Canvas.rendermodes
    s = pf.canvas.settings
    if s.bgcolor2 is None:
        s.bgcolor2 = s.bgcolor
    itemlist = [_I('rendermode', mode, choices=modes),
                _I('linewidth', s.linewidth, itemtype='float'),
                _I('bgcolor', s.bgcolor, itemtype='color'),
                _I('bgcolor2', s.bgcolor2, itemtype='color'),
                _I('fgcolor', s.fgcolor, itemtype='color'),
                _I('slcolor', s.slcolor, itemtype='color'),
                _I('Store these settings as defaults', False),
                ]
    res = widgets.InputDialog(itemlist,'Config Dialog').getResult()
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
    itemlist = [_C('Number of viewports',len(pf.GUI.viewports.all)),
                _C('Viewport layout direction',current,'select',{'choices':directions}),
                _C('Number of viewports per row/column',pf.GUI.viewports.ncols),
                ]
    res = widgets.InputDialog(itemlist,'Config Dialog').getResult()
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


def canvasSettings():
    dia = None
    def apply_():
        dia.acceptData()
        set_near_clip(dia.results['near'])
    def close():
        dia.close()
        
    def set_near_clip(v):
        dist = pf.canvas.camera.getDist()
        pf.canvas.camera.setClip(10**v*dist,10.*dist)
        pf.canvas.update()
        
    dia = widgets.InputDialog(
        caption='Canvas Settings',
        items=[
            _C('near',-1.0,'slider',{'min':-100,'max':100,'scale':0.01,'func': set_near_clip,'text':'Near clipping plane'}),
            ],
        actions=[('Done',close),('Apply',apply_)]
        )
    dia.show()


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
            _C('Line Smoothing','Off','radio',{'choices':['On','Off']}),
            _C('Polygon Mode',None,'radio',{'choices':['Fill','Line']}),
            _C('Polygon Fill',None,'radio',{'choices':['Front and Back','Front','Back']}),
            _C('Culling','Off','radio',{'choices':['On','Off']}),
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
        vp.removeAll()
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
        (_('&Canvas Settings'),canvasSettings),
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
