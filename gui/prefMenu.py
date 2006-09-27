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

def newaskConfigPreferences(items,store):
    """Ask preferences stored in config variables.

    Items in list should only be keys. The current values are retrieved
    from the config.
    A config section name should be specified if the items are not in the
    top config level.
    """
    if not store:
        store = GD.cfg
    itemlist = [ [ i,store.setdefault(i,'') ] for i in items ]
    res,accept = widgets.inputDialog(itemlist,'Config Dialog').process()
    if accept:
        #print "ACCEPTED following values:"
        for r in res:
            #print r
            store[r[0]] = eval(r[1])


def setHelp():
    askConfigPreferences([['viewer'],['homepage'],['history'],['bookmarks']],'help')

def setDrawtimeout():
    askConfigPreferences([['drawtimeout','int']])


def setBGcolor():
    col = GD.cfg['draw/bgcolor']
    col = widgets.getColor(col)
    if col:
        print "New background color %s" % col
        GD.prefsChanged = True
        GD.cfg['draw/bgcolor'] = col
        draw.bgcolor(col)


def setLinewidth():
    askConfigPreferences([['draw/linewidth']])
    draw.linewidth(GD.cfg['draw/linewidth'])

def setSize():
    GD.gui.resize(800,600)
    
def setCanvasSize():
    res = draw.askItems([['w',GD.canvas.width()],['h',GD.canvas.height()]])
    GD.canvas.resize(int(res['w']),int(res['h']))
        
    
def setRender():
    askConfigPreferences([['specular'], ['shininess']],'render')

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
