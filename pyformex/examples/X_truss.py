#!/usr/bin/env pyformex --gui
# $Id: X_truss.py 131 2006-09-19 17:57:54Z bverheg $
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
"""X-shaped truss

level = 'normal'
topics = ['geometry']
techniques = ['colors']

"""

# This is needed if we want to import this module in another script
from formex import *

class X_truss:
    """An X-shaped truss girder.

    The truss has a constant height and consists of n modules of the same
    length. By default all modules are delimited by vertical bars and all
    crossing diagonals are connected to each other. There are options to
    not connect the diagonals and to not have interior verticals.
    This yields four possible layouts (bars are connected at o symbols):

                   with interior verticals      without interior verticals
                     o-----o-----o--               o-----o-----o--           
                     |\   /|\   /|\                |\   / \   / \   
        diagonals    | \ / | \ / | \               | \ /   \ /   \  
        connected    |  o  |  o  |                 |  o     o       
                     | / \ | / \ | /               | / \   / \   /  
                     |/   \|/   \|/                |/   \ /   \ /   
                     o-----o-----o--               o-----o-----o--  
   
                     o-----o-----o--               o-----o-----o--           
                     |\   /|\   /|\                |\   / \   / \   
        diagonals    | \ / | \ / | \               | \ /   \ /   \  
           not       |  X  |  X  |                 |  X     X       
        connected    | / \ | / \ | /               | / \   / \   /  
                     |/   \|/   \|/                |/   \ /   \ /   
                     o-----o-----o--               o-----o-----o--  
    """
    
    def __init__(self,n_mod,mod_length,height,diagonals_connected=True,interior_verticals=True):
        """Creates an X-shaped truss.

        The truss has n_mod modules, each of length mod_length and height.
        By default diagonals are connected and there are interior verticals.
        The instance will have the following attributes:
          bot_nodes: a Formex with the n_nod+1 bottom nodes
          top_nodes: a Formex with the n_mod+1 top nodes
          mid_nodes: a Formex with the n_mod central nodes (or None)
          bot : a Formex with the n_mod bottom bars
          top : a Formex with the n_mod top bars
          vert : a Formex with the (n_mod+1) or 2 vertical bars
          mid1 : a Formex with the n_mod climbing (SW-NE) diagonals
          mid2 : a Formex with the n_mod descending (NW-SE) diagonals
        All Formex instances have their members ordered from left to right.
        The bottom left node has coordinates [0.,0.,0.]
        All nodes have z-coordinate 0.
        """
        total_length = n_mod * mod_length
        # Nodes
        bot_nodes = Formex([[[0.0,0.0]]]).replic(n_mod+1,mod_length)
        top_nodes = bot_nodes.translate([0.0,height,0.0])
        if diagonals_connected:
            mid_nodes = Formex(bot_nodes[:n_mod]).translate([0.5*mod_length,0.5*height, 0.0])
        else:
            mid_nodes = None
        # Truss Members
        bot = connect([bot_nodes,bot_nodes],bias=[0,1])
        top = connect([top_nodes,top_nodes],bias=[0,1])
        if interior_verticals:
            vert = connect([bot_nodes,top_nodes])
        else:
            vert1 = connect([Formex(bot_nodes[:1]),Formex(top_nodes[:1])])
            vert2 = vert1.translate([total_length,0.,0.])
            vert = vert1+vert2
        if diagonals_connected:
            dia1 = connect([bot_nodes,mid_nodes]) + connect([mid_nodes,top_nodes],bias=[0,1])
            dia2 = connect([top_nodes,mid_nodes]) + connect([mid_nodes,bot_nodes],bias=[0,1])
        else:
            dia1 = connect([bot_nodes,top_nodes],bias=[0,1])
            dia2 = connect([top_nodes,bot_nodes],bias=[0,1])
        # save attributes
        self.n_mod = n_mod
        self.mod_length = mod_length
        self.height = height
        self.diagonals_connected = diagonals_connected
        self.interior_verticals = interior_verticals
        self.total_length = total_length
        self.bot_nodes = bot_nodes
        self.top_nodes = top_nodes
        self.mid_nodes = mid_nodes
        self.bot = bot
        self.top = top
        self.vert = vert
        self.dia1 = dia1
        self.dia2 = dia2


    def allNodes(self):
        """Return a Formex with all nodes."""
        all_nodes = self.top_nodes + self.bot_nodes
        if self.mid_nodes:
            all_nodes += self.mid_nodes
        return all_nodes

    def allBars(self):
        """Return a Formex with all nodes."""
        return self.bot+self.top+self.vert+self.dia1+self.dia2


if __name__ == 'draw':
    # This is executed when the example is launched from the GUI

    wireframe()
    reset()
    def example(diag=True,vert=True):
        truss = X_truss(12,2.35,2.65,diag,vert)

        truss.bot.setProp(3)
        truss.top.setProp(3)
        truss.vert.setProp(0)
        truss.dia1.setProp(1)
        truss.dia2.setProp(1)

        clear()
        draw(truss.allNodes(),wait=False)
        draw(truss.allBars())

    for diag in [True,False]:
        for vert in [True,False]:
            example(diag,vert)
    
# End
