#!/usr/bin/env pyformex
class WireStent:
    def __init__(self,r,l,nx,ny,nb,dz,connectors=True):
        dz = 0.5*dz
        bump_z=lambda x: 1.-(x/nb)**2
        base = Formex(pattern('1')).replic(nb,1.0).bump1(2,[0.,0.,dz],bump_z,0)
        base = base.scale([1./nb,1./nb,1.])
        NE = base.shear(1,0,1.)
        SE = base.reflect(2).shear(1,0,-1.)
        NE.setProp(1)
        SE.setProp(3)
        cell1 = (NE+SE).rosette(2,180)
        if connectors:
            cell1 += Formex([[NE[0][0],SE[0][0]]],2)
        cell2 = cell1.reflect(2)
        cell1 = cell1.translate([1.,1.,0.])
        cell2 = cell2.translate([-1.,-1.,0.])
        dx = 4.
        dy = 4.
        F = (cell1+cell2).replic2(nx,ny,dx,dy)
        F = F.translate([0.,0.,r])
        self.F = F.cylindrical(dir=[2,0,1],scale=[1.,360./(nx*dx),l/ny/dy])
    def all(self):
        return self.F

clear()
F = WireStent(10.,200.,12,12,4,1.).all()
G = WireStent(8.,200.,12,12,4,1.).all()
draw(F+G)
draw(F.translate([0.,30.,0.]))
