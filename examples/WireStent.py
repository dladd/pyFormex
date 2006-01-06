#!/usr/bin/env pyformex
# $Id$
#
"""Wire Stent"""

class WireStent:
    """Constructs a single or double layered tubular stent of crossed wires.

    A stent is a tubular shape such as used for opening obstructed blood
    vessels. This stent is made frome sets of wires spiraling in two
    directions. The geometry is defined by the following parameters:
      r  : radius of the stent
      l  : length of the stent
      nx : number of wires in one spiral set
      ny : number of modules in axial direction
      nb : number of elements in a strut (a part of a wire between two
           crossings)
      dz : distance between the wires axes at a crossing (this is equal to
           the wire thickness if they are just touching)
    By default, there will be connectors between the wires at echa crossing.
    They can be switched off in the constructor.
    The returned formex has one set of wires with property 1, the other with
    property 3. The connectors have property 2.
    """
    def __init__(self,r,l,nx,ny,dz,nb,connectors=True):
        """Create the Wire Stent."""
        dz = 0.5*dz
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
        cell1 = cell1.translate([1.,1.,0.])
        cell2 = cell2.translate([-1.,-1.,0.])
        # the base pattern cell1+cell2 now has size [-2,-2]..[2,2]
        # Create the full pattern by replication
        dx = 4.
        dy = 4.
        F = (cell1+cell2).replic2(nx,ny,dx,dy)
        # fold it into a cylinder
        self.F = F.translate([0.,0.,r]).cylindrical(dir=[2,0,1],scale=[1.,360./(nx*dx),l/ny/dy])

    def all(self):
        """Return the Formex with all bar elements."""
        return self.F

class DoubleHelixStent:
    """Constructs a double helix wire stent.

    A stent is a tubular shape such as used for opening obstructed blood
    vessels. This stent is made frome sets of wires spiraling in two
    directions. The geometry is defined by the following parameters:
      L  : length of the stent
      De : external diameter of the stent 
      D  : average stent diameter
      d  : wire diameter
      be : pitch angle (degrees)
      p  : pitch  
      nx : number of wires in one spiral set
      ny : number of modules in axial direction
      ds : extra distance between the wires (default is 0.0 for touching wires)
      dz : maximal distance of wire center to average cilinder
      nb : number of elements in a strut (a part of a wire between two
           crossings), default 4
    By default, there will be connectors between the wires at each crossing.
    They can be switched off in the constructor.
    The returned formex has one set of wires with property 1, the other with
    property 3. The connectors have property 2.
    """
    def __init__(self,De,L,d,nx,be,ds=0.0,nb=4,connectors=True):
        """Create the Wire Stent."""
        D = De - 2*d - ds
        r = 0.5*D
        dz = 0.5*(ds+d)
        p = math.pi*D*tand(be)
        ny = int(nx*L/p)  # The obtained length may be a bit shorter than L
        print "pitch",p
        print "ny",ny
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
        cell1 = cell1.translate([1.,1.,0.])
        cell2 = cell2.translate([-1.,-1.,0.])
        # the base pattern cell1+cell2 now has size [-2,-2]..[2,2]
        # Create the full pattern by replication
        dx = 4.
        dy = 4.
        F = (cell1+cell2).replic2(nx,ny,dx,dy)
        # fold it into a cylinder
        self.F = F.translate([0.,0.,r]).cylindrical(dir=[2,0,1],scale=[1.,360./(nx*dx),p/nx/dy])

    def all(self):
        """Return the Formex with all bar elements."""
        return self.F

clear()
F = WireStent(10.,200.,12,12,1.,4).all()
draw(F)
setview('myview1',(30.,0.,0.))
view('myview1',True)

# double
G = WireStent(8.,200.,12,12,1.,4).all()
draw(G)
draw(F.translate([0.,30.,0.]))

# Doublehelix
H = DoubleHelixStent(10.,150.,0.2,12,30.).all()
clear()
draw(H)
