#!/usr/bin/env pyformex
# $Id$
##
##  This file is part of pyFormex 0.8.3 Release Sun Dec  5 18:01:17 2010
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

"""ColorScale

Examples showing the use of the 'colorscale' plugin

level = 'normal'
topics = ['FEA']
techniques = ['dialog', 'color']

"""

from gui.colorscale import *
from gui.gluttext import GLUTFONTS
from gui.widgets import simpleInputItem as I, groupInputItem as G


input_data = [
    I('valrange',text='Value range type',itemtype='select',choices=['Minimum-Medium-Maximum','Minimum-Maximum']),
    I('maxval',12.0,text='Maximum value'),
    I('medval',0.0,text='Medium value'),
    I('minval',-6.0,text='Minimum value'),
    I('predef',True,text='Use a predefined color palette'),
    I('palet',text='Predefined color palette',choices=Palette.keys(),enabled=True),
    G('Custom Color palette',[
        I('maxcol',[1.,0.,0.],text='Maximum color',itemtype='color'),
        I('medcol',[1.,1.,0.],text='Medium color',itemtype='color'),
        I('mincol',[1.,1.,1.],text='Minimum color',itemtype='color'),
        ],enabled=False),
    I('maxexp',1.0,text='High exponent'),
    I('minexp',1.0,text='Low exponent'),
    I('ncolors',12,text='Number of colors'),
    I('showgrid',False,text='Show grid'),
    I('linewidth',1.5,text='Line width',enabled=False),
    I('dec',2,text='Decimals'),
    I('scale',0,text='Scaling exponent'),
    I('lefttext',True,text='Text left of colorscale'),
    I('font','hv18',text='Font',choices=GLUTFONTS.keys()),
    I('header','Currently not displayed',text='Header',enabled=False),
    I('gravity','Notused',text='Gravity',enabled=False),
    I('autosize',True,text='Autosize'),
    I('size',(100,600),text='Size'),
    I('position',[400,50],text='Position'),
    ]

input_enablers = [
    ('valrange','Minimum-Medium-Maximum','medval','medcol'),
    ('predef',True,'palet'),
    ('predef',False,'Custom Color palette'),
    ('showgrid',True,'linewidth'),
    ('autosize',False,'size'),
    ]


def show():
    """Accept the data and draw according to them"""
    global medval,medcol,palet,minexp,grid
    
    clear()
    lights(False)
    dialog.acceptData()
    res = dialog.results
    globals().update(res)

    
    if valrange == 'Minimum-Maximum':
        medval = None
#        if not predef:
#            medcol = None
        minexp = None

    if not predef:
        palet = map(GLColor,[mincol,medcol,maxcol])

    if showgrid:
        if ncolors <= 50:
            grid = ncolors
        else:
            grid = 1
    else:
        grid = 0

    x,y = position

    if autosize:
        mw,mh = pf.canvas.Size()
        h = int(0.9*(mh-y))
        w = min(0.1*mw,100)
    else:
        w,h = size

    # ok, now draw it
    drawColorScale(palet,minval,maxval,medval,maxexp,minexp,ncolors,dec,scale,grid,linewidth,lefttext,font,x,y,w,h)     


def drawColorScale(palet,minval,maxval,medval,maxexp,minexp,ncolors,dec,scale,grid,linewidth,lefttext,font,x,y,w,h):
    """Draw a color scale with the specified parameters"""
    CS = ColorScale(palet,minval,maxval,midval=medval,exp=maxexp,exp2=minexp)
    CL = ColorLegend(CS,ncolors)
    CLA = decors.ColorLegend(CL,x,y,w,h,grid=grid,font=font,dec=dec,scale=scale,linewidth=linewidth,lefttext=lefttext) 
    decorate(CLA)


def close():
    global dialog
    pf.PF['ColorScale_data'] = dialog.results
    if dialog:
        dialog.close()
        dialog = None


def timeOut():
    """What to do on a InputDialog timeout event.

    As a policy, all pyFormex examples should behave well on a
    dialog timeout.
    Most users can simply ignore this.
    """
    show()
    close()


if __name__ == 'draw':

    # Update the data items from saved values
    try:
        saved_data = named('ColorScale_data')
        widgets.updateDialogItems(input_data,save_data)
    except:
        pass

    # Create the modeless dialog widget
    dialog = widgets.InputDialog(input_data,enablers=input_enablers,caption='ColorScale Dialog',actions = [('Close',close),('Show',show)],default='Show')

    # Examples style requires a timeout action
    dialog.timeout = timeOut

    # Show the dialog and let the user have fun
    dialog.show()

# End

