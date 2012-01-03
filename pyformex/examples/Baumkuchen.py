# $Id$ *** pyformex ***
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
"""Baumkuchen Vault

level = 'beginner'
topics = ['structure']
techniques = ['color','bump']

"""
clear()
m = 12 # number of cells in direction 0
n = 36 # number of cells in direction 1
k = 7  # number of vaults in direction 0
e1 = 30 # elevation of the major arcs
e2 = 5  # elevation of the minor arcs

# Create a grid of beam elements
a1 = Formex('l:2').replic2(m+1,n,1,1,0,1) + \
     Formex('l:1').replic2(m,n+1,1,1,0,1)
draw(a1,'front')
p = array(a1.center())
p[2] = e1
f = lambda x:1-(x/18)**2/2
a2 = a1.bump(2,p,f,1)
draw(a2,'bottom',color='red')
p[2] = e2
a3 = a2.bump(2,p,lambda x:1-(x/6)**2/2,0)
draw(a3,'bottom',color='green')
# Replicate the structure in x-direction
a4 = a3.replicate(k,dir=0,step=m)
draw(a4,'bottom',color='blue')
exit()
clear()
draw(a4,'bottom')

# End
