#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""Functions from the Pref menu."""

import globaldata as GD
import os

import widgets
import draw


def askConfigPreferences(items,prefix=None,store=None):
    """Ask preferences stored in config variables.

    Items in list should only be keys. store is usually a dictionary, but
    can be any class that allow the setdefault method for lookup while
    setting the default, and the store[key]=val syntax for setting the
    value.
    If a prefix is given, actual keys will be 'prefix/key'. 
    The current values are retrieved from the store, and the type returned
    will be in accordance.
    If no store is specified, the global config GD.cfg is used.
    """
    if not store:
        store = GD.cfg
    if prefix:
        items = [ '%s/%s' % (prefix,i) for i in items ]
    itemlist = [ [ i,store.setdefault(i,'') ] for i in items ]
    res,accept = widgets.InputDialog(itemlist,'Config Dialog').getResult()
    if accept:
        for i,r in zip(itemlist,res):
            GD.debug("IN : %s\nOUT: %s" % (i,r))
            if type(i[1]) == str:
                store[r[0]] = r[1]
            else:
                store[r[0]] = eval(r[1])
    return accept


def setHelp():
    askConfigPreferences(['viewer','help/manual','help/pydocs'])

def setCommands():
    askConfigPreferences(['editor','viewer','browser'])

def setDrawtimeout():
    askConfigPreferences(['draw/wait'])

def setBGcolor():
    col = GD.cfg['draw/bgcolor']
    col = widgets.getColor(col)
    if col:
        GD.cfg['draw/bgcolor'] = col
        draw.bgcolor(col)

def setLinewidth():
    askConfigPreferences(['draw/linewidth'])

def setSize():
    GD.gui.resize(800,600)
    
def setCanvasSize():
    res = draw.askItems([['w',GD.canvas.width()],['h',GD.canvas.height()]])
    GD.canvas.resize(int(res['w']),int(res['h']))

def setPickSize():
    w,h = GD.cfg['pick/size']
    res = draw.askItems([['w',w],['h',h]])
    GD.cfg['pick/size'] = (int(res['w']),int(res['h']))
        
    
def setRender():
    if askConfigPreferences(['ambient', 'specular', 'emission', 'shininess'],'render'):
        draw.smooth()

def setLight(light=0):
    keys = [ 'ambient', 'diffuse', 'specular', 'position' ]
    tgt = 'render/light%s'%light
    localcopy = {}
    localcopy.update(GD.cfg[tgt])
    if askConfigPreferences(keys,store=localcopy):
        GD.cfg[tgt] = localcopy
        draw.smooth()

def setLight0():
    setLight(0)

def setLight1():
    setLight(1)

def setRotFactor():
    askConfigPreferences(['gui/rotfactor'])
def setPanFactor():
    askConfigPreferences(['gui/panfactor'])
def setZoomFactor():
    askConfigPreferences(['gui/zoomfactor'])
 

def setFont(font=None):
    """Set the main application font.

    If no font is specified, a user dialog pops up to select it.
    """
    if font is None:
        font,ok = widgets.selectFont()
        if not ok:
            return
    GD.app.setFont(font)
    if GD.gui:
        GD.gui.update()


def setFontSize(s=None):
    """Set the main application font size to the given point size.

    If no size is specified, it is set from the configuration.
    """
    if s:
        GD.cfg['gui/fontsize'] = s
    else:
        s = GD.cfg.get('gui/fontsize',12)
    font = GD.app.font()
    font.setPointSize(int(s))
    setFont(font)


def setStyle(style):
    """Set the main application style."""
    GD.app.setStyle(style)
    if GD.gui:
        GD.gui.update()


def setAppearance():
    """Set the main application style and font from user dialog."""
    style,font = widgets.AppearenceDialog().getResult()
    if font:
        setFont(font)
    if style:
        setStyle(style)
    
# End
