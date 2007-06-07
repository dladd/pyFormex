#!/usr/bin/env python pyformex.py
# $Id: TrussBridge.py 151 2006-11-02 18:18:49Z bverheg $
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
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
draw(nodes,'iso')
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
