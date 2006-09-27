#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.3 Release Mon Feb 20 21:04:03 2006
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
#
"""X-shaped truss analysis"""

#######################################################################
# Setting this path correctly is required to import the analysis module
# You need calpy >= 0.3.3
# It can be downloaded from ftp://bumps.ugent.be/calpy/
calpy_path = '/usr/local/lib/calpy-0.3.3'
#######################################################################

linewidth(1.0)
clear()
from examples.X_truss import X_truss
bgcolor(lightgrey)

# create a truss (Vandepitte, Chapter 1, p.16)
n = 5
l = 800.
h = 800.
truss = X_truss(n,l,h)

# draw it
clear()
draw(truss.allNodes(),wait=False)
draw(truss.allBars())

# assign property numbers
truss.bot.setProp(0)
truss.top.setProp(0)
truss.vert.setProp(2)
truss.dia1.setProp(1)
truss.dia2.setProp(1)
for p in [ truss.bot.p, truss.top.p ]:
    p[0] = p[n-1] = 3 

# define member properties
materials={ 'steel' : { 'E' : 207000, 'nu' : 0.3 } }
sections={ 'hor' : 50, 'end' : 40, 'dia' : 40, 'vert': 30 }
properties = { '0' : [ 'steel', 'hor' ],
               '3' : [ 'steel', 'end' ],
               '2' : [ 'steel', 'vert' ],
               '1' : [ 'steel', 'dia' ] }

def getmat(key):
    """Return the 'truss material' with key (str or int)."""
    p = properties.get(str(key),[None,None])
    m = materials.get(p[0],{})
    E = m.get('E',0.)
    rho = m.get('rho',0.)
    A = sections.get(p[1],0.)
    return [ E, rho, A ]


# create model for structural analysis
model = truss.allBars()
coords,elems = model.nodesAndElements()
props = model.prop()
propset = model.propSet()

clear()
draw(Formex(reshape(coords,(coords.shape[0],1,coords.shape[1]))),wait=False)
draw(model)

############################################
##### NOW load the calpy analysis code #####

# Check if we have calpy:
import sys
sys.path.append(calpy_path)
try:
    from fe_util import *
    from truss3d import *
except ImportError:
    import globaldata as GD
    warning("You need calpy-0.3.3 or higher to perform the analysis.\nIt can be obtained from ftp://bumps.ugent.be/calpy/\nYou should also set the correct calpy installation path\n in this example's source file\n(%s).\nThe calpy_path variable is set near the top of that file.\nIts current value is: %s" % (GD.cfg['curfile'],calpy_path))
    exit()
    
############################################

nnod = coords.shape[0]
nelems = elems.shape[0]
# boundary conditions
# we use the knowledge that the elements are in the order
# bot,top,vert,mid1,mid2
# remember to add 1 to number starting from 1, as needed by calpy
nr_fixed_support = elems[0][0]
nr_moving_support = elems[n-1][1]
nr_loaded = elems[2][1] # right node of the 3-d element
bcon = ReadBoundary(nnod,3,"""
  all  0  0  1
  %d   1  1  1
  %d   0  1  1
""" % (nr_fixed_support + 1,nr_moving_support + 1))
NumberEquations(bcon)
mats=array([ getmat(i) for i in range(max(propset)+1) ])
matnod = concatenate([reshape(props+1,(nelems,1)),elems+1],1)
ndof=bcon.max()
nlc=1
loads=zeros((ndof,nlc),Float)
loads[:,0]=AssembleVector(loads[:,0],[ 0.0, -50.0, 0.0 ],bcon[nr_loaded,:])
message("Performing analysis: this may take some time")
displ,frc = static(coords,bcon,mats,matnod,loads,Echo=True)


################################
#Using pyFormex as postprocessor
################################

from gui.colorscale import *
import gui.decors

# Creating a formex for displaying results is fairly easy
results = Formex(coords[elems],range(nelems))
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
drawtext('Normal force in the truss members',400,100,'tr24')
CL = ColorLegend(CS,100)
CLA = decors.ColorLegend(CL,10,10,30,200) 
GD.canvas.addDecoration(CLA)
GD.canvas.update()

# and a deformed plot
dscale = 10000.
dcoords = coords + dscale * displ[:,:,0]
# first load case
deformed = Formex(dcoords[elems],range(nelems))
clear()
GD.canvas.addDecoration(CLA)
linewidth(1)
draw(results,color='darkgreen')
linewidth(3)
draw(deformed,color=cval)
drawtext('Normal force in the truss members',400,100,'tr24')
drawtext('Deformed geometry (scale %.2f)' % dscale,400,130,'tr24')


# End
