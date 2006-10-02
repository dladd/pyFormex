#!/usr/bin/env python
# $Id$
"""Functions from the Pref menu."""

import globaldata as GD

import os

import draw
import widgets


def askConfigPreferences(items,section=None):
    """Ask preferences stored in config variables.

    Items in list should only be keys. The current values are retrieved
    from the config.
    A config section name should be specified if the items are not in the
    top config level.
    """
    if section:
        store = GD.cfg[section]
    else:
        store = GD.cfg
    # insert current values
    for it in items:
        it.insert(1,store.setdefault(it[0],''))
    res,accept = widgets.inputDialog(items,'Config Dialog').process()
    if accept:
        GD.prefsChanged = True
        for r in res:
            GD.debug("%s" % r)
            store[r[0]] = eval(r[1])


def newaskConfigPreferences(items,store=None):
    """Ask preferences stored in config variables.

    Items in list should only be keys. store is usually a dictionary, but
    can be any class that allow the setdefault method for lookup while
    setting the default, and the store[key]=val syntax for setting the
    value.
    The current values are retrieved from the store, and the type returned
    will be in accordance.
    If no store is specified, the global config GD.cfg is used.
    """
    if not store:
        store = GD.cfg
    itemlist = [ [ i,store.setdefault(i,'') ] for i in items ]
    print itemlist
    res,accept = widgets.inputDialog(itemlist,'Config Dialog').process()
    if accept:
        for i,r in zip(itemlist,res):
            GD.debug("IN : %s\nOUT: %s" % (i,r))
            if type(i[1]) == str:
                store[r[0]] = r[1]
            else:
                store[r[0]] = eval(r[1])


def setHelp():
    newaskConfigPreferences(['viewer','help/manual','help/pydocs'],GD.cfg)

def setDrawtimeout():
    newaskConfigPreferences(['wait'],GD.cfg['draw'])


def setBGcolor():
    col = GD.cfg['draw/bgcolor']
    col = widgets.getColor(col)
    if col:
        GD.debug("New background color %s" % col)
        GD.prefsChanged = True
        GD.cfg['draw/bgcolor'] = col
        draw.bgcolor(col)


def setLinewidth():
    newaskConfigPreferences(['draw/linewidth'])
    draw.linewidth(GD.cfg['draw/linewidth'])

def setSize():
    GD.gui.resize(800,600)
    
def setCanvasSize():
    res = draw.askItems([['w',GD.canvas.width()],['h',GD.canvas.height()]])
    GD.canvas.resize(int(res['w']),int(res['h']))
        
    
def setRender():
    newaskConfigPreferences(['specular', 'shininess'],GD.cfg['render'])

def setLight(light=0):
    store = GD.cfg['render/light%d' % light]
    keys = [ 'ambient', 'diffuse', 'specular', 'position' ]
    newaskConfigPreferences(keys,store)

def setLight0():
    setLight(0)
    draw.smooth()

def setLight1():
    setLight(1)
    draw.smooth()


def setLocalAxes():
    GD.cfg['draw/localaxes'] = True 
def setGlobalAxes():
    GD.cfg['draw/localaxes'] = False 

# End
