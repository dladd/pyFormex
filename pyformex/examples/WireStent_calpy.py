#!/usr/bin/env pyformex
# $Id: WireStent_calpy.py 147 2006-10-13 09:30:49Z bverheg $
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
"""Wire stent analysis

level = 'advanced'
topics = ['FEA']
techniques = ['colors'] 
"""

chdir(__file__)

############################
# Load the needed calpy modules    

from plugins import calpy_itf
from calpy.fe_util import *
from calpy.beam3d import *

############################


############################################
# Create geometry
from pyformex.examples.WireStent import DoubleHelixStent
import datetime

# create a Doublehelix stent
stent_diameter = 10.
stent_length = 150.
wire_diameter = 0.2
number_wires = 6
pitch_angle = 30.

# during testing
stent_length = 10.
stent = DoubleHelixStent(stent_diameter,stent_length,
                         wire_diameter,number_wires,pitch_angle,nb=1).all()

if GD.options.gui:
    # draw it
    reset()
    clear()
    draw(stent,view='iso')


    
############################################
# Perform Analysis

# Create output file
if not checkWorkdir():
    print "Could not open a file for writing. I have to stop here"
    exit()

outfilename = 'WireStent_calpy.out'
outfile = file(outfilename,'w')
message("Output is written to file '%s' in %s" % (outfilename,os.getcwd()))
stdout_saved = sys.stdout
sys.stdout = outfile
print "# File created by pyFormex on %s" % time.ctime()
print "# Script name: %s" % GD.scriptName



nel = stent.nelems()
print "Number of elements: %s" % nel
print "Original number of nodes: %s" % stent.nnodes()
# Create FE model
message("Creating Finite Element model: this may take some time.")
nodes,elems = stent.fuse(nodesperbox=1)

nnod = nodes.shape[0]
print "Compressed number of nodes: %s" % nnod

# Create an extra node on the axis for beam orientations
extra_node = array([[0.0,0.0,-10.0]])
coords = concatenate([nodes,extra_node])
nnod = coords.shape[0]
print "After adding a node for orientation: %s" % nnod

# Create element definitions: i j k matnr, where k = nnod (the extra node)
# while incrementing node numbers with 1 (for calpy)
# (remember props are 1,2,3, so are OK)

thirdnode = nnod*ones(shape=(nel,1),dtype=int)
matnr = reshape(stent.prop,(nel,1))
elements = concatenate([elems+1,thirdnode,matnr],1)

# Create endnode sets (with calpy numbering)
bb = stent.bbox()
zlo = bb[0][2]
zhi = bb[1][2]
zmi = (zhi+zlo)/2.
count = zeros(nnod)
for n in elems.flat:
    count[n] += 1
unconnected = arange(nnod)[count==1]
zvals = nodes[unconnected][:,2]
#print zlo,zhi,zmi,zvals
end0 = unconnected[zvals<zmi]
end1 = unconnected[zvals>zmi]
print "Nodes at end 0:",end0
print "Nodes at end 1:",end1

# Create End Connectors to enforce radial boundary conditions
coords_end0 = coords[end0]
extra_nodes = coords_end0 * array([0.80,0.80,1.0])
nnod0 = nnod
coords = concatenate([coords,extra_nodes])
nnod = coords.shape[0]
print "Nodes added for boundary connectors: %s" % (nnod-nnod0)
print "Final number of nodes: %s" % nnod
extra_elems = zeros((nnod-nnod0,4),dtype=int)
end0_ext = arange(nnod0,nnod)
extra_elems[:,0] = end0_ext + 1
extra_elems[:,1] = end0 + 1
extra_elems[:,2] = nnod0
extra_elems[:,3] = 4  # Extra elements have matnr 4
print extra_elems
elements = concatenate([elements,extra_elems])

# Boundary conditions
s = ""
for n in end0_ext + 1:   # NOTICE THE +1 !
    s += "  %d  1  1  1  1  1  1\n" % n
# Also clamp the fake extra node
s += "  %d  1  1  1  1  1  1\n" % nnod0
print "Specified boundary conditions"
print s
bcon = ReadBoundary(nnod,6,s)
NumberEquations(bcon)
print bcon

# Materials (E, G, rho, A, Izz, Iyy, J)
mats = zeros((4,7),float)
A = math.pi * wire_diameter ** 2
Izz = Iyy = math.pi * wire_diameter ** 4 / 4
J = math.pi * wire_diameter ** 4 / 2
E = 207000.
nu = 0.3
G = E/2/(1+nu)
rho = 0.
mats[0] = mats[2] = [ E, G, rho, A, Izz, Iyy, J ]
mats[1] = [E, G, 0.0, A*10**3, Izz*10**6, Iyy*10**6, 0.0]
mats[3] = [E, G, 0.0, 0.0, Izz*10**6, Iyy*10**6, 1.0]
print mats

# Create loads
nlc = 1
ndof = bcon.max()
loads = zeros((ndof,nlc),float)
zforce = [ 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ]
for n in end1: # NO +1 HERE!
    loads[:,0] = AssembleVector(loads[:,0],zforce,bcon[n,:])

# Perform analysis
import calpy
calpy.options.optimize=True
print elements
displ,frc = static(coords,bcon,mats,elements,loads,Echo=True)

print "# Analysis finished on %s" % time.ctime()
sys.stdout = stdout_saved
outfile.close()



################################
#Using pyFormex as postprocessor
################################

if GD.options.gui:

    from gui.colorscale import *
    import gui.decors

    # Creating a formex for displaying results is fairly easy
    elems = elements[:,:2]-1
    results = Formex(coords[elems])
    clear()
    draw(results,color='black')

    # Now try to give the formex some meaningful colors.
    # The frc array returns element forces and has shape
    #  (nelems,nforcevalues,nloadcases)
    # In this case there is only one resultant force per element (the
    # normal force), and only load case; we still need to select the
    # scalar element result values from the array into a onedimensional
    # vector val. 
    val = frc[:,0,0]
    # create a colorscale
    CS = ColorScale([blue,yellow,red],val.min(),val.max(),0.,2.,2.)
    cval = array(map(CS.color,val))
    #aprint(cval,header=['Red','Green','Blue'])
    clear()
    draw(results,color=cval)

    bgcolor('lightgreen')
    linewidth(3)
    x = GD.canvas.width()//2
    TA = drawtext('Normal force in the members',x,100,font='tr32')
    CL = ColorLegend(CS,100)
    CLA = decors.ColorLegend(CL,10,10,30,200) 
    decorate(CLA)
    sleep(1)

    # and a deformed plot on multiple scales
    dscales = arange(1,6) * 1.0
    loadcase = 0
    for dscale in dscales:
        dcoords = coords + dscale * displ[:,0:3,loadcase]
        clear()
        decorate(CLA)
        decorate(TA)
        linewidth(1)
        draw(results,color='darkgreen',wait=False)
        linewidth(3)
        deformed = Formex(dcoords[elems])
        draw(deformed,color=cval)
        drawtext('Deformed geometry (scale %.2f)' % dscale,x,70)


    if ack("Show the output file?"):
        showFile(outfilename)

# End
