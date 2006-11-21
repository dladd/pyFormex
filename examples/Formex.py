#!/usr/bin/env pyformex
# $Id$

"""Formex.py

Creates a Formex menu"""

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
            w = F.where(dir=axis,xmin=xc1,xmax=xc2)
            oldF = F.cclip(w)
            F = F.clip(w)
            drawChanges(F,oldF)

        
        
if __name__ == 'draw':

    Menu = widgets.Menu('Formex')
    MenuData = [
#        ("Action","&List Formices",formex_list),
        ("Action","&Make Selection",make_selection),
        ("Action","&Draw Selection",draw_selection),
        ("Action","&Translate Selection",translate_selection),
        ("Action","&Center Selection",center_selection),
        ("Action","&Rotate Selection",rotate_selection),
        ("Action","&Clip Selection",clip_selection),
        ("Action","&Close",Menu.close),
        ]
    for key,txt,val in MenuData:
        Menu.addItem(txt,val)

# End
