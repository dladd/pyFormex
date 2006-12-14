#!/usr/bin/env pyformex
# $Id$

"""Formex.py

Executing this script creates a Formex menu in the menubar.
"""

from PyQt4 import QtCore,QtGui
from gui import widgets
import commands, os


selection = None

def drawChanges(F,oldF):
    clear()
    draw(oldF,color='yellow',wait=False)
    draw(F)

def get_selection(mode=None):
    """Show the list of formices (and return a selection)"""
    return widgets.Selection(listAll(),'Known Formices',mode,sort=True,\
                             selected=selection).getResult()

def formex_list():
    """Show a list of known formices."""
    message(get_selection())
    
def make_selection():
    global selection
    selection = get_selection('multi')
    print selection

def draw_selection():
    global selection
    print selection
    if selection:
        map(draw,selection)


def translate_selection():
    """Translate the selected Formices."""
    global selection
    itemlist = [ [ 'direction',0], ['distance','1.0'] ] 
    res,accept = widgets.inputDialog(itemlist,'Translation Parameters').getResult()
    if accept:
        dir = int(res[0][1])
        dist = float(res[1][1])
        for F in byName(selection):
            oldF = F
            F = F.translate(dir,dist)
            drawChanges(F,oldF)
        

def center_selection():
    """Center the selection."""
    global selection
    for F in byName(selection):
        oldF = F
        F = F.translate(-array(F.center()))
        drawChanges(F,oldF)


def rotate_selection():
    """Rotate the selection."""
    global selection
    itemlist = [ [ 'axis',0], ['angle','0.0'] ] 
    res,accept = widgets.inputDialog(itemlist,'Rotation Parameters').process()
    if accept:
        axis = int(res[0][1])
        angle = float(res[1][1])
        for F in byName(selection):
            oldF = F
            F = F.rotate(angle,axis)
            drawChanges(F,oldF)

        
def clip_selection():
    """Clip the stl model."""
    global F
    itemlist = [['axis',0],['begin',0.0],['end',1.0]]
    res,accept = widgets.inputDialog(itemlist,'Clipping Parameters').process()
    if accept:
        Flist = byName(selection)
        bb = bbox(Flist)
        axis = int(res[0][1])
        xmi = bb[0][axis]
        xma = bb[1][axis]
        dx = xma-xmi
        xc1 = xmi + float(res[1][1]) * dx
        xc2 = xmi + float(res[2][1]) * dx
        for F in Flist:
            w = F.where(dir=axis,min=xc1,max=xc2)
            oldF = F.cclip(w)
            F = F.clip(w)
            drawChanges(F,oldF)


def create_menu():
    """Create the Formex menu."""
    menu = widgets.Menu('Formex')
    MenuData = [
#        ("&List Formices",formex_list),
        ("&Make Selection",make_selection),
        ("&Draw Selection",draw_selection),
        ("&Translate Selection",translate_selection),
        ("&Center Selection",center_selection),
        ("&Rotate Selection",rotate_selection),
        ("&Clip Selection",clip_selection),
        ("&Close",close_menu),
        ]
    menu.addItems(MenuData)
    return menu


def close_menu():
    """Close the Formex menu."""
    menu.close()
        
        
if __name__ == 'draw':
    message(__doc__)
    menu = create_menu()

# End
