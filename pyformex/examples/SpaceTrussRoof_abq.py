#!/usr/bin/env pyformex
# $Id: SpaceTrussRoof_fe_abq.py 150 2006-11-01 11:13:34Z bverheg $
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
"""Double Layer Flat Space Truss Roof

level = 'advanced'
topics = ['FEA']
techniques = ['colors'] 
"""

from plugins.properties import *
from plugins.fe_abq import *

####
#Data
###################

dx = 1800 # Modular size [mm]
ht = 900  # Deck height [mm]
nx = 4     # number of bottom deck modules in x direction 
ny = 5   # number of bottom deck modules in y direction 

q = -0.005 #distributed load [N/mm^2]


#############
#Creating the model
###################

top = (Formex("1").replic2(nx-1,ny,1,1) + Formex("2").replic2(nx,ny-1,1,1)).scale(dx)
top.setProp(3)
bottom = (Formex("1").replic2(nx,ny+1,1,1) + Formex("2").replic2(nx+1,ny,1,1)).scale(dx).translate([-dx/2,-dx/2,-ht])
bottom.setProp(0)
T0 = Formex(4*[[[0,0,0]]]) 	   # 4 times the corner of the top deck
T4 = bottom.select([0,1,nx,nx+1])  # 4 nodes of corner module of bottom deck
dia = connect([T0,T4]).replic2(nx,ny,dx,dx)
dia.setProp(1)

F = (top+bottom+dia)

# Show upright
createView('myview1',(0.,-90.,0.))
clear();linewidth(1);draw(F,view='myview1')


############
#Creating FE-model
###################

M = F.toMesh()

###############
#Creating elemsets
###################
# Remember: elems are in the same order as elements in F
topbar = where(F.prop==3)[0]
bottombar = where(F.prop==0)[0]
diabar = where(F.prop==1)[0]

###############
#Creating nodesets
###################

nnod=M.ncoords()
nlist=arange(nnod)
count = zeros(nnod)
for n in M.elems.flat:
    count[n] += 1
field = nlist[count==8]
topedge = nlist[count==7]
topcorner = nlist[count==6]
bottomedge = nlist[count==5]
bottomcorner = nlist[count==3]
support = concatenate([bottomedge,bottomcorner])
edge =  concatenate([topedge,topcorner])

########################
#Defining and assigning the properties
#############################

Q = 0.5*q*dx*dx

P = PropertyDB()
P.nodeProp(set=field,cload = [0,0,Q,0,0,0])
P.nodeProp(set=edge,cload = [0,0,Q/2,0,0,0])
P.nodeProp(set=support,bound = [1,1,1,0,0,0])

circ20 = ElemSection(section={'name':'circ20','sectiontype':'Circ','radius':10, 'cross_section':314.159}, material={'name':'S500', 'young_modulus':210000, 'shear_modulus':81000, 'poisson_ratio':0.3, 'yield_stress' : 500,'density':0.000007850})

# example of how to set the element type by set
P.elemProp(set=topbar,section=circ20,eltype='T3D2')
P.elemProp(set=bottombar,section=circ20,eltype='T3D2')

# alternatively, we can specify the elements by an index value
# in an array that we will pass in the Abqdata 'eprop' argument
P.elemProp(prop=1,section=circ20,eltype='T3D2')

# Since all elements have same characteristics, we could just have used:
#   P.elemProp(section=circ20,elemtype='T3D2')
# But putting the elems in three sets allows for separate postprocessing 


# Print node and element property databases
for p in P.nprop:
    print p
for p in P.eprop:
    print p
    


#############
#Writing the inputfile
###################

step = Step()
out = Output(type='field',variable='preselect')
res = [ Result(kind='element',keys=['S']),
	Result(kind='node',keys=['U'])
	]
model = Model(M.coords,M.elems)

if not checkWorkdir():
    exit()

message("Writing the Abaqus file")
AbqData(model,P,[step],eprop=F.prop,out=[out],res=res).write('SpaceTruss')

# End
