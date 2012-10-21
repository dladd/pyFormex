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
"""Viewport Menu.

This module defines the functions of the Viewport menu.
"""
from __future__ import print_function

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
            # "SHOULD ADD canvas/
            res['_save_'] = save
            #prefMenu.updateSettings(res)
            #pf.cfg.update(pf.canvas.settings.__dict__,name='canvas')

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
#            _I('shading'),
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


def setRendering():
    import canvas

    vp = pf.GUI.viewports.current
    dia = None

    def enableLightParams(mode):
        if dia is None:
            return
        mode = str(mode)
        on = mode.startswith('smooth')
        for f in ['ambient','material']:
            dia['render/'+f].setEnabled(on)
        dia['material'].setEnabled(on)

    def updateLightParams(matname):
        matname=str(matname)
        mat = pf.GUI.materials[matname]
        val = utils.prefixDict(mat.dict(),'material/')
        print("UPDATE",val)
        dia.updateData(val)
    
    def close():
        dia.close()
        
    def accept(save=False):
        dia.acceptData()
        print("RESULTS",dia.results)
        if dia.results['render/mode'].startswith('smooth'):
            res = utils.subDict(dia.results,'render/',strip=False)
            matname = dia.results['render/material']
            matdata = utils.subDict(dia.results,'material/')
            # Currently, set both in cfg and Material db
            pf.cfg['material/%s' % matname] = matdata
            pf.GUI.materials[matname] = canvas.Material(matname,**matdata)
        else:
            res = utils.selectDict(dia.results,['render/mode','render/lighting'])
        res['_save_'] = save
        print("RES",res)
        updateSettings(res)
        print(pf.cfg)
        vp = pf.GUI.viewports.current
        vp.resetLighting()
        #if pf.cfg['render/mode'] != vp.rendermode:
        print("SETMODE %s %s" % (pf.cfg['render/mode'],pf.cfg['render/lighting']))
        vp.setRenderMode(pf.cfg['render/mode'],pf.cfg['render/lighting'])
        print(vp.rendermode,vp.lighting)
        vp.update()
        toolbar.updateLightButton()
        

    def acceptAndSave():
        accept(save=True)
        
    def createDialog():  
        matnames = pf.GUI.materials.keys()
        mat = vp.material
        mat_items = [
            _I(a,text=a,value=getattr(mat,a),itemtype='slider',min=0,max=100,scale=0.01,func=set_mat_value) for a in [ 'ambient', 'diffuse', 'specular', 'emission']
            ] + [
            _I(a,text=a,value=getattr(mat,a),itemtype='slider',min=1,max=128,scale=1.,func=set_mat_value) for a in ['shininess']
            ]
        items = [
            _I('render/mode',vp.rendermode,text='Rendering Mode',itemtype='select',choices=canvas.Canvas.rendermodes),#,onselect=enableLightParams),
            _I('render/lighting',vp.lighting,text='Use Lighting'),
            _I('render/ambient',vp.lightprof.ambient,text='Global Ambient Lighting'),
            _I('render/material',vp.material.name,text='Material',choices=matnames,onselect=updateLightParams),
            _G('material',text='Material Parameters',items=mat_items),
            ]

        enablers = [
            ('render/lighting',True,'render/ambient','render/material','material'),
            ]
        dia = widgets.InputDialog(
            caption='pyFormex Settings',
            enablers = enablers,
            #store=pf.cfg,
            items=items,
            #prefix='render/',
            autoprefix=True,
            actions=[
                ('Close',close),
                ('Apply and Save',acceptAndSave),
                ('Apply',accept),
                ]
            )
        enableLightParams(vp.rendermode)
        return dia

    dia = createDialog()
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
    res = draw.askItems(pf.canvas.options.items())
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
