# $Id$                  *** pyformex ***
##
##  This file is part of pyFormex
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
##  along with this program.  If not, see http://www.gnu.org/licenses/.
##
"""Manantiales

A building with a scallop dome roof, inspired by the shape of
Los Manantiales by Felix Candela.
"""
_status = 'checked'
_level = 'normal'
_topics = ['geometry','surface','domes']
_techniques = ['dialog', 'color', 'boolean']

from gui.draw import *

clear()
view('front')
import simple
from plugins.trisurface import *


def scallop(F,sx,sy,n,f,c,r):
    """Create a scallop dome from a sectorial layout.

    - `F`: a Formex with the layout of a sector of the projection
      of the dome. This layout is given in cylindrical coordinates,
      and normally covers a rectangular part over the range [0,R], [0,PHI],
      where R is the radius of the horizontal projection of the dome and
      PHI is the opening angle of the sector.
    - `sx`,`sy`: the range of the x, resp. y coordinates in F
    - `n`: number of modules along circumference. The layout will repeated
      this number of times in cirferential direction.
    - `f`: either 1 or 2. If 1, the arcs will meet under sharp angles. If 2,
      the arcs will meet with a smooth connection.
    - `c`: elevation at the center of the dome.
    - `r`: maximum elevation at the circumference.
    """
    message("Scallop Dome with n=%d, f=%d, c=%f, r=%f" % (n,f,c,r))
    F = F.scale([1./sx,1./sy,1.])
    F = F.map(lambda x,y,z: [x,y,c*(1.-x*x)+r*x*x*power(4*(1.-y)*y,f)])
    a = 360./n
    F = F.scale([sx,a,1.])
    F = F.cylindrical([0,1,2]).rosette(n,a)
    return F

	
def projectOnXY(F):
    """Project a structure on the xy-plane.

    Returns a copy of the structure with all z-coordinates set to zero.
    """
    G = F.copy()
    G.coords[:,:,2] = 0.
    return G


def Draw(F,G):
    clear()
    draw(F,wait=False,bkcolor=black)
    draw(G,marksize=10,nolight=True,bbox=None)
    


def do_manant(narcs,radius,ypowr,celev,relev,nmod,mmod):
    """Create the manantiales dome"""
    
    # Create a triangular pattern in the first quadrant, and circulize it
    F = Formex('3:012',1).replic2(nmod,mmod,1,1,0,1,1,-1) + Formex('3:021',3).reverse().replic2(nmod-1,mmod-1,1,1,0,1,1,-1).translate(0,1)
    # We also create points on the border of the circle sector, and give them
    # the same transformations
    G = Formex([nmod*1.,0.,0.]).replic(mmod+1,1.,dir=1)
    Draw(F,G)
    
    # Circulize it
    F = F.circulize1()
    G = G.circulize1()
    Draw(F,G)

    # Transform to cylindrical coordinates
    F = F.toCylindrical([0,1,2])
    G = G.toCylindrical([0,1,2])
    Draw(F,G)

    # Create the scallop dome
    sx,sy,sz = F.sizes()
    F = scallop(F,sx,sy,narcs,ypowr,celev,relev)
    G = scallop(G,sx,sy,narcs,ypowr,celev,relev)
    Draw(F,G)

    # Project on the xy plane to form the base
    dome = F
    base = projectOnXY(dome).reverse().setProp(8-dome.prop)
    draw(base,bkcolor=black)

    # Close dome and base
    F = projectOnXY(G)
    wall = connect([F,F,G],bias=[0,1,1])+connect([F,G,G],bias=[0,1,0])
    draw(wall,color=yellow,bkcolor=black)

    # Create a closed surface
    S = TriSurface(dome+base+wall)

    # Create a reversed cone
    
    # Cut surface with a cone

    # Create new cylindrical walls

    # Cut walls with dome

    # Combine floor, walls, roof

    # Compute surfaces and volumes


########################################################################
# User Interface #
##################

def run():
    reset()
    clear()
    
    # parameters
    defaults = dict(
        narcs = 8,
        radius = 16.0,
        ypowr = 2.0,
        celev = 5.0,
        relev = 8.0,
        nmod = 8,
        mmod = 8,
        )
    print defaults
    try:
        defaults.update(named('_Manantiales_data'))
    except:
        pass
    print defaults

    nmod = 8      # number of modules in radial direction
    mmod = 8      # number of modules over 1 arcade in tangential direction 

    res = askItems([
        _I('narcs',text='Number of arcades (>=6)'),
        _I('radius',text='Maximal radius of the dome'),
        _I('ypowr',text='Power of the elevation curves (1=sharp, 2=smooth)'),
        _I('celev',text='Elevation at center of dome'),
        _I('relev',text='Maximum elevation along circumference'),
        _I('nmod',text='Number of modules in radial direction'),
        _I('mmod',text='Number of modules over 1 arcade in tangential direction'),
        ], store=defaults)
    if not res:
        return

    # Save data for next execution
    export({'_Manantiales_data':res})
    do_manant(**res)


if __name__ == 'draw':
    run()

# End

