#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

from simple import rectangle
from gui.widgets import *


def showSuperEgg():
    """Draw a super Egg from set global parameters"""
    nx,ny = grid
    b,h = long_range[1]-long_range[0], lat_range[1]-lat_range[0]
    B = rectangle(nx,ny,b,h,diag).translate([long_range[0],lat_range[0],1.])
    F = B.superSpherical(n=north_south,e=east_west).scale(scale)
    clear()
    draw(F,color=color)
    export({'Egg':F})
    return

savefile = None

def show():
    w.acceptData()
    globals().update(w.result)
    showSuperEgg()

def close():
    w.close()
    if savefile:
        savefile.close()

def reset():
    print "RESET"

def save():
    global savefile
    show()
    if savefile is None:
        filename = askFilename(filter="Text files (*.txt)")
        if filename:
            savefile = file(filename,'a')
    if savefile:
        savefile.write('%s\n' % str(w.result))

def play():
    global savefile
    if savefile:
        filename = savefile.name
        savefile.close()
    else:
        filename = askFilename(filter="Text files (*.txt)",exist=True)
    if filename:
        savefile = file(filename,'r')
        for line in savefile:
            globals().update(eval(line))
            showSuperEgg()
        savefile = file(filename,'a')

    
if __name__ == "draw":

    reset()
    smoothwire()
    lights(True)
    transparent(False)
    setView('eggview',(0.,-30.,0.))
    view('eggview')

    # Initial values for global parameters
    scale = [1.,1.,1.]
    north_south = 1.0
    east_west = 1.0
    lat_range = (-90.,90.)
    long_range = (-180.,180.)
    grid = [24,16]
    diag = ''
    scale = [1.,1.,1.]
    color = 'red'

    items = [ [n,globals()[n]] for n in [
        'north_south','east_west','lat_range','long_range',
        'grid','diag','scale','color'] ]
    # turn 'diag' into a complex input widget
    items[5].extend(['radio',['','u','d']])

    actions = [('Close',close),('Reset',reset),('Replay',play),('Save',save),('Show',show)]
    
    w = InputDialog(items,caption='SuperEgg parameters',actions=actions,default='Show')

    w.show()
    
        

# End
