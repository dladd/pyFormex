#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
##

"""SuperEgg

level = 'advanced'
topics = ['geometry']
techniques = ['programming','modal dialog','persistence']

"""

from simple import rectangle
from gui.widgets import *
from gui.draw import *


dialog = None
savefile = None

def showSuperEgg():
    """Draw a super Egg from set global parameters"""
    nx,ny = grid
    b,h = long_range[1]-long_range[0], lat_range[1]-lat_range[0]
    if grid_base.startswith('tri'):
        diag = grid_base[-1]
    else:
        diag = ''
    B = rectangle(nx,ny,b,h,diag=diag,bias=grid_bias).translate([long_range[0],lat_range[0],1.])
    if grid_skewness != 0.0:
        B = B.shear(0,1,grid_skewness)
    F = B.superSpherical(n=north_south,e=east_west,k=eggness).scale(scale)
    clear()
    draw(F,color=color)
    export({name:F})


def show():
    dialog.acceptData()
    globals().update(dialog.result)
    showSuperEgg()

def close():
    dialog.close()
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
        savefile.write('%s\n' % str(dialog.result))

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


def createSuperEgg():
    global dialog
    
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
    eggness = 0.0
    lat_range = (-90.,90.)
    long_range = (-180.,180.)
    grid = [24,32]
    grid_base = 'quad'
    grid_bias = 0.0
    grid_skewness = 0.0
    scale = [1.,1.,1.]
    color = 'red'
    name = 'Egg-0'

    items = [ [n,locals()[n]] for n in [
        'north_south','east_west','eggness','lat_range','long_range',
        'grid','grid_base','grid_bias','grid_skewness','scale',
        'name','color'] ]
    # turn 'diag' into a complex input widget
    items[6].extend(['radio',['quad','tri-u','tri-d']])

    actions = [('Close',close),('Reset',reset),('Replay',play),('Save',save),('Show',show)]
    
    dialog = InputDialog(items,caption='SuperEgg parameters',actions=actions,default='Show')

    dialog.show()

    
if __name__ == "draw":

    createSuperEgg()


# End
