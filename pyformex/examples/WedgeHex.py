# *** pyformex ***
# $Id$
##
##  This file is part of pyFormex 0.8.5  (Sun Dec  4 21:24:46 CET 2011)
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

"""WedgeHex

level = 'normal'
topics = ['mesh']
techniques = ['revolve','degenerate'] 
"""

import simple

clear()
smoothwire()

# create a 2D xy mesh
nx,ny = 6,2
G = simple.rectangle(1,1,1.,1.).replic2(nx,ny)
M = G.toMesh()
draw(M, color='red')
view('iso')

# create a 3D axial-symmetric mesh by REVOLVING
n,a = 8,45.
R = M.revolve(n,angle=a,axis=1,around=[1.,0.,0.])
sleep(2)
draw(R,color='yellow')

# reduce the degenerate elements to WEDGE6
clear()
print R
ML = R.fuse().splitDegenerate()
print "AFTER SPLITTING: %s MESHES" % len(ML)
for m in ML:
    print m
ML = [ Mi.setProp(i) for i,Mi in enumerate(ML) ]
draw(ML)

# End
