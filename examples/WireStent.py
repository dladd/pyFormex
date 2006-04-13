#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""Wire Stent"""

from formex import *
import globaldata
import math

class DoubleHelixStent:
    """Constructs a double helix wire stent.

    A stent is a tubular shape such as used for opening obstructed
    blood vessels. This stent is made frome sets of wires spiraling
    in two directions.
    The geometry is defined by the following parameters:
      L  : length of the stent
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
        print "Mean radius at creation = %s"% r
        dz = 0.5*(ds+d)
        p = math.pi*D*tand(be)
        ny = int(round(nx*L/p))  # The actual length may differ a bit from L
        #print "pitch",p
        #print "ny",ny
        # a single bumped strut, oriented along the x-axis
        bump_z=lambda x: 1.-(x/nb)**2
        base = Formex(pattern('1')).replic(nb,1.0).bump1(2,[0.,0.,dz],bump_z,0)
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
        # and create its mirror
        cell2 = cell1.reflect(2)
        # and move both to appropriate place
        self.cell1 = cell1.translate([1.,1.,0.])
        self.cell2 = cell2.translate([-1.,-1.,0.])
        # the base pattern cell1+cell2 now has size [-2,-2]..[2,2]
        # Create the full pattern by replication
        dx = 4.
        dy = 4.
        F = (self.cell1+self.cell2).replic2(nx,ny,dx,dy)
        # fold it into a cylinder
        self.F = F.translate([0.,0.,r]).cylindrical(dir=[2,0,1],scale=[1.,360./(nx*dx),p/nx/dy])
        self.ny = ny

    def all(self):
        """Return the Formex with all bar elements."""
        return self.F


if __name__ == "draw":


    # show an example
    L = 80.
    H = DoubleHelixStent(10.,L,0.2,12,30.,ds=-0.2).all()
    clear()
    draw(H,view='iso')
    # and save it in a lot of graphics formats
    if ack("Do you want to save this image (in lots of formats) ?"):
        for ext in [ 'bmp', 'jpg', 'pbm', 'png', 'ppm', 'xbm', 'xpm', 'eps', 'ps', 'pdf', 'tex' ]: 
            if ext == 'jpg':
                fmt = 'JPEG'
            else:
                fmt = ext.upper()
            save('WireStent.'+ext,fmt)
