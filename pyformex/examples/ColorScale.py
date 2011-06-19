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

input_data = [
    _I('valrange',text='Value range type',itemtype='select',choices=['Minimum-Medium-Maximum','Minimum-Maximum']),
    _I('maxval',12.0,text='Maximum value'),
    _I('medval',0.0,text='Medium value'),
    _I('minval',-6.0,text='Minimum value'),
    _I('palet',text='Predefined color palette',choices=Palette.keys()),
    _G('custom',text='Custom Color palette',items=[
        _I('maxcol',[1.,0.,0.],text='Maximum color',itemtype='color'),
        _I('medcol',[1.,1.,0.],text='Medium color',itemtype='color'),
        _I('mincol',[1.,1.,1.],text='Minimum color',itemtype='color'),
        ],checked=False),
    _I('maxexp',1.0,text='High exponent'),
    _I('minexp',1.0,text='Low exponent'),
    _I('ncolors',200,text='Number of colors'),
    _T('Grid/Label',[
        _I('ngrid',-1,text='Number of grid intervals'),
        _I('linewidth',1.5,text='Line width'),
        _I('nlabel',-1, text='Number of label intervals'), 
        _I('dec',2,text='Decimals'),
        _I('scale',0,text='Scaling exponent'),
        _I('lefttext',True,text='Text left of colorscale'),
        _I('font','hv18',text='Font',choices=GLUTFONTS.keys()),
        _I('header','Currently not displayed',text='Header',enabled=False),
        _I('gravity','Notused',text='Gravity',enabled=False),
        ]),
    _T('Position/Size',[
        _I('autosize',True,text='Autosize'),
        _I('size',(100,600),text='Size'),
        _I('autopos',True,text='Autoposition'),
        _I('position',[400,50],text='Position'),
        ]),
    ]

input_enablers = [
    ('valrange','Minimum-Medium-Maximum','medval','medcol'),
    ('custom',False,'palet'),
    ('autosize',False,'size'),
    ('autopos',False,'position'),
    ]


def show():
    """Accept the data and draw according to them"""
    global medval,medcol,palet,minexp,grid,nlabels
    
    clear()
    lights(False)
    dialog.acceptData()
    res = dialog.results
    print res
    globals().update(res)

    
    if valrange == 'Minimum-Maximum':
        medval = None
        minexp = None

    if custom:
        palet = map(GLColor,[mincol,medcol,maxcol])

    mw,mh = pf.canvas.Size()
    x,y = position
    if autopos:
        x = mw / 2
    if autosize:
        h = int(0.9*(mh-y))
        w = min(0.1*mw,100)
    else:
        w,h = size
        

    # ok, now draw it
    drawColorScale(palet,minval,maxval,medval,maxexp,minexp,ncolors,dec,scale,ngrid,linewidth,nlabel,lefttext,font,x,y,w,h)     


def drawColorScale(palet,minval,maxval,medval,maxexp,minexp,ncolors,dec,scale,ngrid,linewidth,nlabel,lefttext,font,x,y,w,h):
    """Draw a color scale with the specified parameters"""
    CS = ColorScale(palet,minval,maxval,midval=medval,exp=maxexp,exp2=minexp)
    CL = ColorLegend(CS,ncolors)
    CLA = decors.ColorLegend(CL,x,y,w,h,ngrid=ngrid,linewidth=linewidth,nlabel=nlabel,font=font,dec=dec,scale=scale,lefttext=lefttext) 
    decorate(CLA)


def close():
    global dialog
    pf.PF['ColorScale_data'] = dialog.results
    if dialog:
        dialog.close()
        dialog = None
    # Release scriptlock
    scriptRelease(__file__)


def timeOut():
    """What to do on a InputDialog timeout event.

    As a policy, all pyFormex examples should behave well on a
    dialog timeout.
    Most users can simply ignore this.
    """
    show()
    wait()
    close()


if __name__ == 'draw':

    # Update the data items from saved values
    try:
        saved_data = named('ColorScale_data')
        widgets.updateDialogItems(input_data,saved_data)
    except:
        pass

    # Create the modeless dialog widget
    dialog = widgets.InputDialog(input_data,enablers=input_enablers,caption='ColorScale Dialog',actions = [('Close',close),('Show',show)],default='Show')

    # Examples style requires a timeout action
    dialog.timeout = timeOut

    # Show the dialog and let the user have fun
    dialog.show()

    # Block other scripts 
    scriptLock(__file__)

# End

