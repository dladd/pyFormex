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

def showSuperEgg():
    """Draw a super Egg from set global parameters"""
    nx,ny = grid
    b,h = long_range[1]-long_range[0], lat_range[1]-lat_range[0]
    B = rectangle(nx,ny,b,h,diag).translate([long_range[0],lat_range[0],1.])
    F = B.superSpherical(n=north_south,e=east_west).scale(scale)
    clear()
    draw(F,color=color)
    return

        
if __name__ == "draw":

    reset()
    smoothwire()
    lights(True)
    transparent(False)
    setView('eggview',(0.,-45.,0.))
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

    while True:
        items = [ [n,globals()[n]] for n in [
            'north_south','east_west','lat_range','long_range',
            'grid','diag','scale','color'] ]
        # turn 'diag' into a complex input widget
        items[5].extend(['radio',['','u','d']])
        res = askItems(items,caption="SuperEgg parameters")
        if not res:
            break;

        globals().update(res)

        clear()
        showSuperEgg()
            
        # Break from endless loop if an input timeout is active !
        if widgets.input_timeout >= 0:
            break

    exit()


# End
