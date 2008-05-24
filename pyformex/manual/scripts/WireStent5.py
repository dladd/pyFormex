        # and create its mirror
        cell2 = cell1.reflect(2)
        # and move both to appropriate place
        self.cell1 = cell1.translate([1.,1.,0.])
        self.cell2 = cell2.translate([-1.,-1.,0.])
        # the base pattern cell1+cell2 now has size [-2,-2]..[2,2]
