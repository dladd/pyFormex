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
from utils import NameSequence
from gui.widgets import *
from gui.draw import *


grid_dialog = None
dialog = None
savefile = None

tol = 1.e-4
    

def createGrid():
    """Create the grid from global parameters"""
    global B
    nx,ny = grid_size
    b,h = x_range[1]-x_range[0], y_range[1]-y_range[0]
    if grid_base.startswith('tri'):
        diag = grid_base[-1]
    else:
        diag = ''
    B = rectangle(nx,ny,b,h,diag=diag,bias=grid_bias).translate([x_range[0],y_range[0],1.])
    if grid_skewness != 0.0:
        B = B.shear(0,1,grid_skewness*b*ny/(h*nx))
    if x_clip:
        B = B.clip(B.test('any',dir=0,min=x_clip[0]-tol*b,max=x_clip[1]+tol*b))
    if y_clip:
        B = B.clip(B.test('any',dir=1,min=y_clip[0]-tol*h,max=y_clip[1]+tol*h))
    export({grid_name:B})
    

def createSuperShape():
    """Create a super shape from global parameters"""
    global F
    B = GD.PF[grid]
    F = B.superSpherical(n=north_south,e=east_west,k=eggness)
    if scale == [1.0,1.0,1.0]:
        print "No need to scale"
    else:
        print "Scaling"
    F = F.scale(scale)
    if post:
        print "Post transformation"
        F = eval(post)
    clear()
    draw(F,color=color)
    export({name:F})


def showGrid():
    """Show the last created grid"""
    clear()
    wireframe()
    draw(B,color=grid_color)
    

def showSuperShape():
    """Show the last created super shape"""
    clear()
    smoothwire()
    draw(F,color=color)


def grid_show():
    grid_dialog.acceptData()
    globals().update(grid_dialog.result)
    createGrid()
    showGrid()

def grid_showshape():
    """Show the shape from the grid dialog"""
    grid_dialog.acceptData()
    globals().update(grid_dialog.result)
    createGrid()
    show()
    
def grid_close():
    if grid_dialog:
        grid_dialog.close()

def grid_reset():
    print "RESET GRID"

def show():
    dialog.acceptData()
    globals().update(dialog.result)
    createSuperShape()
    showSuperShape()

def close():
    if dialog:
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


def openSuperShapeDialogs():
    global grid_dialog,dialog

    reset()
    smoothwire()
    lights(True)
    transparent(False)
    setView('eggview',(0.,-30.,0.))
    view('eggview')

    
    # Initial values for global grid parameters
    grid_size = [24,12]
    x_range = (-180.,180.)
    y_range = (-90.,90.)
    grid_base = 'quad'
    grid_bias = 0.0
    grid_skewness = 0.0
    x_clip = (-180.,180.)
    y_clip = (-90.,90.)
    grid_name = 'Grid-0'
    grid_color = 'blue'
    
    grid_items = [ [n,locals()[n]] for n in [
        'x_range','y_range','grid_size','grid_base','grid_bias','grid_skewness',
        'x_clip','y_clip','grid_name','grid_color'] ]
    # turn 'diag' into a complex input widget
    grid_items[3].extend(['radio',['quad','tri-u','tri-d']])

    grid_actions = [('Close',grid_close),('Reset',grid_reset),('Show Grid',grid_show),('Show',grid_showshape)]
    
    grid_dialog = InputDialog(grid_items,caption='SuperEgg parameters',actions=grid_actions,default='Show')

    grid_dialog.show()

    # Initial values for global grid parameters
    grid = grid_name
    scale = [1.,1.,1.]
    north_south = 1.0
    east_west = 1.0
    eggness = 0.0
    scale = [1.,1.,1.]
    post = ''
    name = 'Shape-0'
    color = 'red'

    items = [ [n,locals()[n]] for n in [
        'grid','north_south','east_west','eggness','scale', 'post',
        'name','color'] ]

    actions = [('Close',close),('Reset',reset),('Replay',play),('Save',save),('Show',show)]
    
    dialog = InputDialog(items,caption='SuperShape parameters',actions=actions,default='Show')

    dialog.show()

    
if __name__ == "draw":

    grid_close()
    close()
    openSuperShapeDialogs()


# End
