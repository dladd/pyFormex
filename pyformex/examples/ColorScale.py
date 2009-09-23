#!/usr/bin/env pyformex
# $Id$

"""ColorScale

Examples showing the use of the 'colorscale' plugin

level = 'normal'
topics = ['FEA']
techniques = ['dialog', 'colors']

"""

from gui.colorscale import *
from gui.gluttext import GLUTFONTS


def getData():
    res = askItems([
        ('Value range type',None,'radio',['Minimum-Medium-Maximum','Minimum-Maximum']),
        ('Maximum value',12.0),
        ('Medium value',0.0),
        ('Minimum value',-6.0),
        ('Use a predefined color palet',True),
        ('Predefined color palet',None,'select',Palette.keys()),
        ('Maximum color',[1.,0.,0.]),
        ('Medium color',[1.,1.,0.]),
        ('Minimum color',[1.,1.,1.]),
        ('High exponent',1.0),
        ('Low exponent',1.0),
        ('Number of colors',12),
        ('Decimals',2),
        ('Scaling exponent',0),
        ('Show grid',True),
        ('Line width',1.5),
        ('Text left of colorscale',True),
        ('Font','hv18','select',GLUTFONTS.keys()),
        ('Position',[400,50]),
        ('Size',(100,600)),
        ('Header','Currently not displayed'),
        ('Gravity','Notused'),
        ])
    
    if not res:
        return

    valrange = res['Value range type']
    maxval = res['Maximum value']
    minval = res['Minimum value']
    if valrange == 'Minimum-Medium-Maximum':
        medval = res['Medium value']
    else: 
        medval = None

    predef = res['Use a predefined color palet']
    if predef:
        palet = res['Predefined color palet']
    else:
        maxcol = res['Maximum color']
        mincol = res['Minimum color'] 
        if valrange == 'Minimum-Medium-Maximum':
            medcol = res['Medium color']
        else: 
            medcol = None
        palet = [mincol,medcol,maxcol]

    maxexp = res['High exponent']
    if valrange == 'Minimum-Medium-Maximum':
        minexp = res['Low exponent']
    else:
        minexp = None

    ncolors = res['Number of colors']
    dec = res['Decimals']
    scale = res['Scaling exponent']
    if res['Show grid']:
        if ncolors <= 50:
            grid = ncolors
        else:
            grid = 1
    else:
        grid = 0
    linewidth = res['Line width']
    lefttext = res['Text left of colorscale']
    font = res['Font']
    print res['Position']
    x,y = 200,50
    w,h = 50,400
    # ok, now draw it
    drawColorScale(palet,minval,maxval,medval,maxexp,minexp,ncolors,dec,scale,grid,linewidth,lefttext,font,x,y,w,h)     


def drawColorScale(palet,minval,maxval,medval,maxexp,minexp,ncolors,dec,scale,grid,linewidth,lefttext,font,x,y,w,h):
    """Draw a color scale with the specified parameters"""
    CS = ColorScale(palet,minval,maxval,midval=medval,exp=maxexp,exp2=minexp)
    CL = ColorLegend(CS,ncolors)
    CLA = decors.ColorLegend(CL,x,y,w,h,grid=grid,font=font,dec=dec,scale=scale,linewidth=linewidth,lefttext=lefttext) 
    decorate(CLA)
    

if __name__ == 'draw':
    flat()
    clear()
    view('front')
    bgcolor('white')
    getData()
