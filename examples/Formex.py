#!/usr/bin/env pyformex
# $Id$

"""Formex.py

Creates a Formex menu"""

from PyQt4 import QtCore,QtGui
from gui import widgets
import commands, os

clear()

selection = None

def get_selection(mode):
    """Show the list of formices (and return a selection)"""
    return widgets.Selection(listAll(),'Known Formices',mode).getResult()

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
        
        
if __name__ == 'draw':

    Menu = widgets.Menu('Formex')
    MenuData = [
        ("Action","&List Formices",formex_list),
        ("Action","&Make Selection",make_selection),
        ("Action","&Draw Selection",draw_selection),
        ("Action","&Close",Menu.close),
        ]
    for key,txt,val in MenuData:
        Menu.addItem(txt,val)

# End
