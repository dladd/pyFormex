#!/usr/bin/env pyformex --gui
# $Id$
##
##  This file is part of pyFormex 0.8.5     Sun Nov  6 17:27:05 CET 2011
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  https://savannah.nongnu.org/projects/pyformex/
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Wire Stent

level = 'normal'
topics = ['geometry']
techniques = ['dialog', 'persistence', 'color']
"""

# needed if we import this from another script
from formex import *

class DoubleHelixStent:
    """Constructs a double helix wire stent.

    A stent is a tubular shape such as used for opening obstructed
    blood vessels. This stent is made frome sets of wires spiraling
    in two directions.
    The geometry is defined by the following parameters:
      L  : approximate length of the stent
      De : external diameter of the stent 
      D  : average stent diameter
      d  : wire diameter
      be : pitch angle (degrees)
      p  : pitch  
      nx : number of wires in one spiral set
      ny : number of modules in axial direction
      ds : extra distance between the wires (default is 0.0 for
           touching wires)
      dz : maximal distance of wire center to average cilinder
      nb : number of elements in a strut (a part of a wire between two
           crossings), default 4
    The stent is created around the z-axis. 
    By default, there will be connectors between the wires at each
    crossing. They can be switched off in the constructor.
    The returned formex has one set of wires with property 1, the
    other with property 3. The connectors have property 2. The wire
    set with property 1 is winding positively around the z-axis.
    """
    def __init__(self,De,L,d,nx,be,ds=0.0,nb=4,connectors=True):
        """Create the Wire Stent."""
        D = De - 2*d - ds
        r = 0.5*D
        dz = 0.5*(ds+d)
        p = math.pi*D*tand(be)
        nx = int(nx)
        ny = int(round(nx*L/p))  # The actual length may differ a bit from L
        # a single bumped strut, oriented along the x-axis
        bump_z=lambda x: 1.-(x/nb)**2
        base = Formex('l:1').replic(nb,1.0).bump1(2,[0.,0.,dz],bump_z,0)
        # scale back to size 1.
        base = base.scale([1./nb,1./nb,1.])
        # NE and SE directed struts
        NE = base.shear(1,0,1.)
        SE = base.reflect(2).shear(1,0,-1.)
        NE.setProp(1)
        SE.setProp(3)
        # a unit cell of crossing struts
        cell1 = (NE+SE).rosette(2,180)
        # add a connector between first points of NE and SE
        if connectors:
            cell1 += Formex([[NE[0][0],SE[0][0]]],2)
        # create its mirror
        cell2 = cell1.reflect(2).translate([2.,2.,0.])
        base = cell1 + cell2
        # reposition to base to origin [0,0]
        base = base.translate(-base.bbox()[0])
        # Create the full pattern by replication
        dx,dy = base.bbox()[1][:2]
        F = base.replic2(nx,ny,dx,dy)
        # fold it into a cylinder
        self.F = F.translate([0.,0.,r]).cylindrical(dir=[2,0,1],scale=[1.,360./(nx*dx),p/nx/dy])
        self.ny = ny

    def getFormex(self):
        """Return the Formex with all bar elements.

        This includes the elements along all the wires, as well as the
        connectors between the wires.
        """
        return self.F

    def getWireAxes(self):
        """Return the wire axes as curves.

        The return value is two lists of curves (PolyLines), representing the
        individual wire axes for each wire direction.
        """
        import connectivity
        M = self.F.toMesh()
        ML = [ M.withProp(i) for i in [1,3] ]
        wires = [ connectivity.connectedLineElems(Mi.elems) for Mi in ML ]
        wireaxes = [ [ Formex(M.coords[wi]).toCurve() for wi in wiresi ] for wiresi in wires ]
        return wireaxes


if __name__ == "draw":

    # show an example

    wireframe()
    reset()

    res = askItems([
        _I('L',80.,text='Length of the stent'),
        _I('D',10.,text='Diameter of the stent'),
        _I('n',12 ,text='Total number of wires'),
        _I('b',30.,text='Pitch angle of the wires'),
        _I('d',0.2,text='Diameter of the wires'),
        _I('show',itemtype='radio',choices=['Formex','Curves']),
        ])

    if not res:
        exit()

    globals().update(res)
    if (n % 2) != 0:
        warning('Number of wires must be even!')
        exit()
    
    H = DoubleHelixStent(D,L,d,n,b)
    clear()
    
    if show=='Formex':
        draw(H.getFormex(),view='iso')

    else:
        view('iso')
        for w,c in zip(H.getWireAxes(),['black','magenta']):
            draw(w,color=c)


#End
