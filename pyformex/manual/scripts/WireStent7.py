        # fold it into a cylinder
        self.F = F.translate([0.,0.,r]).cylindrical(dir=
		[2,0,1],scale=[1.,360./(nx*dx),p/nx/dy])
	  self.ny = ny