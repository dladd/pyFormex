#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.5 Release Fri Aug 10 12:04:07 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Wire Stent Demo"""
##
##This Demo is intended for educational purposes by rewriting the WireStent.py 
##example and adding lots of drawing instructions and comments. More details
##regarding the used definitions can be found in the Pyformex reference manual.
###

# needed if we import this from another script
from formex import *

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
        dz = 0.5*(ds+d)
        p = math.pi*D*tand(be)
        nx = int(nx)
        ny = int(round(nx*L/p))  # The actual length may differ a bit from L
        # a single bumped strut, oriented along the x-axis
        bump_z=lambda x: 1.-(x/nb)**2
        A=Formex(pattern('1'),3)
        GD.message("\nThis Demo is intended for educational purposes by rewriting the WireStent.py \n example and adding lots of drawing instructions and comments. More details \n regarding the used definitions can be found in the Pyformex reference manual.")
        GD.message("\nStep 1: Create a Formex: a line of length 1 (with property 3) oriented along the X-axis\n               A = Formex(pattern('1'),3)")
        draw(A,view='bottom')
        pause()
        B=Formex(A.replic(nb,1.0),1)
        GD.message("Step 2: Copy the Formex nb times in the X(0)-direction\n               B = Formex(A.replic(nb,1.0),1)")
        draw(B,view='last')
        pause()
        clear() 
        base = Formex(B.bump1(2,[0.,0.,dz],bump_z,0),1)
        GD.message("Step 3: Create a bump in the Z(2)-direction\n               base = Formex(B.bump1(2,[0.,0.,dz],bump_z,0),1)")
        draw(base,view='last')
        pause()
        clear()
        # scale back to size 1.
        base = base.scale([1./nb,1./nb,1.])
        GD.message("Step 4: Rescale the base line to size 1\n               base = base.scale([1./nb,1./nb,1.])")
        draw(base,view='last')
        pause()
        clear()
        # NE and SE directed struts
        NE = base.shear(1,0,1.)
        NE.setProp(1)
        GD.message("Step 5: Skew the base line to NE-direction\n               NE = base.shear(1,0,1.)")
##      The nxt two lines serve to rotate the camera up over 30°, i.e. 6 times the rotUp definition from cameraMenu 
        GD.canvas.camera.rotate(30,1,0,0)
        GD.canvas.update()   
        draw(NE)
        pause()
        clear()
        SE = base.reflect(2).shear(1,0,-1.)
        SE.setProp(3)
        GD.message("Step 6: Create a mirrored base line and skew it to SE-direction\n               SE = base.reflect(2).shear(1,0,-1.)")
        draw(SE,view='last')
        pause()
        clear()
        cell=(NE+SE)
        GD.message("Step 7: Create the base cell by combining the NE and SE formices\n               cell = (NE+SE)")
        draw(cell,view='last')
        pause()
        clear()
        # a unit cell of crossing struts
        cell1 = (cell).rosette(2,180)
        GD.message("Step 8: Create the base module (cell1) of two crossing wires by replicating the base cell by an angular rotation\n               cell1 = (cell).rosette(2,180)")
        draw(cell1,view='last')
        pause()
        clear()
        # add a connector between first points of NE and SE
        if connectors:
            cell1 += Formex([[NE[0][0],SE[0][0]]],2)
        GD.message("Step 9: Add a connector between the first points of NE and SE of the base module\n               cell1 += Formex([[NE[0][0],SE[0][0]]],2)")
        draw(cell1,view='last')
        pause()
        clear()    
        # and create its mirror
        cell2 = cell1.reflect(2)
        GD.message("Step 10: Create a mirror in Z(2)-direction of the base module\n               cell2 = cell1.reflect(2)")
        draw(cell2,view='last')
        pause()
        clear()
        # and move both to appropriate place
        self.cell1 = cell1.translate([1.,1.,0.])
        self.cell2 = cell2.translate([-1.,-1.,0.])
        # the base pattern cell1+cell2 now has size [-2,-2]..[2,2]
        # Create the full pattern by replication
        dx = 4.
        dy = 4.
        module=(self.cell1+self.cell2)
        GD.message("Step 11: Extend the base module with its mirrored and translated copy\n               module = (self.cell1+self.cell2)")
        draw(module,view='last')
        pause()
        clear()
        F = module.replic2(nx,ny,dx,dy)
        GD.message("Step 12: Replicate the base module in both directions of the base plane\n               F = module.replic2(nx,ny,dx,dy)")
        draw(F,view='last')
        pause()
        clear()
        # fold it into a cylinder
        C=F.translate([0.,0.,r])
        GD.message("Step 13: Translate the full patern over the stent radius in Z-direction\n               C=F.translate([0.,0.,r])")
        draw(C,view='last')
        pause()
        clear()
        self.F = C.cylindrical(dir=[2,0,1],scale=[1.,360./(nx*dx),p/nx/dy])
        GD.message("Step 14: Roll the nearly planar grid into a cylinder\n               self.F = C.cylindrical(dir=[2,0,1],scale=[1.,360./(nx*dx),p/nx/dy])")
        draw(self.F,view='front')
        pause()
        clear()
        draw(self.F,view='left')
        pause()
        clear()
        self.ny = ny

    def all(self):
        """Return the Formex with all bar elements."""
        return self.F


if __name__ == "draw":

    # show an example
## The following default values are obtained from Jedwab and Clerc (except for L=87.5 and b-30.85)
    D = 16.71
    L = 40.
    d = 0.22
    n = 12
    b = 40
    res = askItems([['Diameter',D],['Length',L],['WireDiam',d],['NWires',n],
                    ['Pitch',b]])
    D = float(res['Diameter'])
    L = float(res['Length'])
    d = float(res['WireDiam'])
    n = int(res['NWires'])
####### The following 3 lines are commented by MDB as n seems to be the number of wires in one direction and thus, this number may be uneven!!! #######
##    if (n % 2) != 0:
##        warning('Number of wires must be even!')
##        exit()
    b = float(res['Pitch'])

    H = DoubleHelixStent(D,L,d,n,b).all()
 #   clear()
    draw(H,view='iso')
    
    # and save it in a lot of graphics formats
##    if ack("Do you want to save this image (in lots of formats) ?"):
##        for ext in [ 'bmp', 'jpg', 'pbm', 'png', 'ppm', 'xbm', 'xpm', 'eps', 'ps', 'pdf', 'tex' ]: 
##           saveImage('WireStent.'+ext)
