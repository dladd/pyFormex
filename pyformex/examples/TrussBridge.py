#!/usr/bin/env pyformex --gui
# $Id: TrussBridge.py 151 2006-11-02 18:18:49Z bverheg $
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
"""TrussBridge

level = 'normal'
topics = ['geometry']
techniques = ['colors']

"""

L = 12000 # Nominal length of the bridge
N = 12    # Number of modules (should be even)
Lo = 300  # Overshoot at the end of the bridge
B = 2000  # Nominal width of the bridge
Bo = 500  # Sideways overshoot for bracing
Bi = 200  # Offset of wind bracing system from girder
H = 1100  # Nominal height of the bridge
Hb = 600  # Height of the bracing

# First we half of one of the trusses. The origin is at the center of the
# bridge
Lm = L/N # modular length
n = N/2  # number of modules for half bridge
b = B/2
clear()

# We start with the bottom girder, and copy it to the top
nodes = Formex([[[0,0,b]]]).replic(n+1,Lm)
draw(nodes,view='iso')
bot_gird = connect([nodes,nodes],bias=[0,1])
top_gird = bot_gird.translate([0,H,0])

# Add the verticals and diagonals
verticals = connect([bot_gird,top_gird],nodid=[1,1])
diagonals = connect([bot_gird,top_gird],nodid=[0,1])
verticals.setProp(3)
diagonals.setProp(3)
# We missed the central vertical : we construct it by hand
central_vertical = Formex([[bot_gird[0,0],top_gird[0,0]]],3)

# Bridge deck and braces
nodes_out = nodes.translate([0,0,Bo])
nodes_up = nodes.translate([0,Hb,0])
deck_girders = connect([nodes_out,nodes_out.reflect(2)])
deck_girders.setProp(1)
braces = connect([nodes_out,nodes_up])
braces.setProp(2)

# Wind bracing
nodes1 = nodes.select([2*i for i in range(n/2+1)]).translate([0,0,-Bi])
nodes2 = nodes.select([2*i+1 for i in range(n/2)]).translate([0,0,-Bi]).reflect(2)
draw(nodes1+nodes2)
wind_bracing = connect([nodes1,nodes2]) + connect([nodes2,nodes1],bias=[0,1])
wind_bracing.setProp(5)


# Assemble half bridge
central = central_vertical + central_vertical.reflect(2)
half_truss = bot_gird + top_gird + verticals + diagonals
quarter = half_truss + braces
half_bridge = quarter + quarter.reflect(2) + deck_girders + wind_bracing
draw(half_bridge+central)


# And a full bridge
bridge = half_bridge + half_bridge.reflect(0) + central

clear()
draw(bridge)
