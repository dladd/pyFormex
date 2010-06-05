#!/usr/bin/env pyformex --gui
# $Id: Isopar.py 921 2009-02-24 09:51:50Z bverheg $
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
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
sleep(2)
clear()
ML = R.splitDegenerate()
ML = [ Mi.setProp(i) for i,Mi in enumerate(ML) ]
draw(ML)

# End
