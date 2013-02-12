# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be)
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
"""Viewport Menu.

This module defines the functions of the Viewport menu.
"""
from __future__ import print_function

import pyformex as pf
import canvas
import widgets
import draw
from gettext import gettext as _
import prefMenu
from widgets import simpleInputItem as _I
import utils


def setTriade():
    try:
        pos = pf.canvas.triade.pos
        siz = pf.canvas.triade.siz
    except:
        pos = 'lb'
        siz = 100
    res = draw.askItems([
        _I('triade',True),
        _I('pos',pos,choices=['lt','lc','lb','ct','cc','cb','rt','rc','rb']),
        _I('size',siz),
        ])
    if res:
        draw.setTriade(res['triade'],res['pos'],res['size'])


def setBgColor():
    """Interactively set the viewport background colors."""
    from gui.drawable import saneColorArray
    from numpy import resize
    import os
    bgmodes = pf.canvas.settings.bgcolormodes
    mode = pf.canvas.settings.bgmode
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
            _I('mode',mode,choices=bgmodes),
            _I('color1',color[0],itemtype='color',text='Background color 1 (Bottom Left)'),
            _I('color2',color[1],itemtype='color',text='Background color 2 (Bottom Right)'),
            _I('color3',color[2],itemtype='color',text='Background color 3 (Top Right)'),
            _I('color4',color[3],itemtype='color',text='Background color 4 (Top Left'),
            _I('showimage',showimage,text='Show background image'),
            _I('image',cur,text='Background image',itemtype='button',func=changeImage),
            viewer,
            _I('_save_',True,text='Save as default'),
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
        setBackground(**res)


def setBackground(mode,color1,color2,color3,color4,showimage,image,_save_):
    if mode == 'solid':
        color = color1
    elif mode == 'vertical':
        color = [color1,color1,color4,color4]
    elif mode == 'horizontal':
        color = [color1,color2,color2,color1]
    else:
        color = [color1,color2,color3,color4]
    if not showimage:
        image = None
    pf.canvas.setBackground(color=color,image=image)
    pf.canvas.update()
    if _save_:
        prefMenu.updateSettings({
            'canvas/bgmode':mode,
            'canvas/bgcolor':color,
            'canvas/bgimage':image,
            '_save_':_save_})


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
    res = draw.askItems([
        _I('w',pf.canvas.width()),
        _I('h',pf.canvas.height())
        ],'Set Canvas Size'
        )
    if res:
        draw.canvasSize(res['w'],res['h'])


def canvasSettings():
    """Interactively change the canvas settings.

    Creates a dialog to change the canvasSettings of the current or any other
    viewport
    """

    dia = None

    def close():
        dia.close()


    def getVp(vp):
        """Return the vp corresponding with a vp choice string"""
        if vp == 'current':
            vp = pf.GUI.viewports.current
        elif vp == 'focus':
            vp = pf.canvas
        else:
            vp = pf.GUI.viewports.all[int(vp)]
        return vp


    def accept(save=False):
        dia.acceptData()
        res = dia.results
        vp = getVp(res['viewport'])
        pf.debug("Changing Canvas settings for viewport %s to:\n%s"%(pf.GUI.viewports.viewIndex(vp),res),pf.DEBUG.CANVAS)
        pf.canvas.settings.update(res,strict=False)
        pf.canvas.redrawAll()
        pf.canvas.update()
        if save:
            res = utils.prefixDict(res,'canvas/')
            print(res)
            res['_save_'] = save
            prefMenu.updateSettings(res)

    def acceptAndSave():
        accept(save=True)

    def changeViewport(vp):
        if vp == 'current':
            vp = pf.GUI.viewports.current
        elif vp == 'focus':
            vp = pf.canvas
        else:
            vp = pf.GUI.viewports.all[int(vp)]
        dia.updateData(vp.settings)

    canv = pf.canvas
    vp = pf.GUI.viewports
    pf.debug("Focus: %s; Current: %s" % (canv,vp),pf.DEBUG.CANVAS)
    s = canv.settings

    dia = widgets.InputDialog(
        caption='Canvas Settings',
        store=canv.settings,
        items=[
            _I('viewport',choices=['focus','current']+[str(i) for i in range(len(pf.GUI.viewports.all))],onselect=changeViewport),
            _I('pointsize',),
            _I('linewidth',),
            _I('linestipple',),
            _I('fgcolor',itemtype='color'),
            _I('slcolor',itemtype='color'),
            _I('smooth'),
            _I('fill'),
            _I('lighting'),
            _I('culling'),
            _I('alphablend'),
            _I('transparency',min=0.0,max=1.0),
            _I('avgnormals',),
            ],
        enablers =[
            ('alphablend',('transparency')),
            ],
        actions=[
            ('Close',close),
            ('Apply and Save',acceptAndSave),
            ('Apply',accept),
            ],
        )
    #dia.resize(800,400)
    dia.show()


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



def drawOptions(d={}):
    """Set the Drawing options.

    A dictionary may be specified to override the current defaults.
    """
    draw.setDrawOptions(d)
    print(pf.canvas.options)
    res = draw.askItems(store=pf.canvas.options,items=[
        _I('view',choices=['None']+pf.canvas.view_angles.keys(),tooltip="Camera viewing direction"),
        _I('bbox',choices=['auto','last'],tooltip="Automatically focus/zoom on the last drawn object(s)"),
        _I('clear',tooltip="Clear the canvas on each drawing action"),
        _I('shrink',tooltip="Shrink all elements to make their borders better visible"),
        _I('shrink_factor'),
        _I('marksize'),
        ],enablers=[('shrink',True,'shrink_factor')]
    )
    if not res:
        return
    if res['view'] == 'None':
        res['view'] = None
    draw.setDrawOptions(res)


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
        (_('&Canvas Settings'),canvasSettings),
        (_('&Global Draw Options'),drawOptions),
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
