#!/usr/bin/env pyformex
# $Id: SpaceTrussRoof_fe_abq.py 150 2006-11-01 11:13:34Z bverheg $
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
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

nodes,elems=F.feModel()

###############
#Creating elemsets
###################
# Remember: elems are in the same order as elements in F
topbar = where(F.p==3)[0]
bottombar = where(F.p==0)[0]
diabar = where(F.p==1)[0]

###############
#Creating nodesets
###################

nnod=nodes.shape[0]
nlist=arange(nnod)
count = zeros(nnod)
for n in elems.flat:
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
P.nodeProp(field,cload = [0,0,Q,0,0,0])
P.nodeProp(edge,cload = [0,0,Q/2,0,0,0])
P.nodeProp(support, bound = [1,1,1,0,0,0])

circ20 = ElemSection(section={'name':'circ20','sectiontype':'Circ','radius':10, 'cross_section':314.159}, material={'name':'S500', 'young_modulus':210000, 'shear_modulus':81000, 'poisson_ratio':0.3, 'yield_stress' : 500,'density':0.000007850})
P.elemProp(topbar,section=circ20,eltype='T3D2')
P.elemProp(bottombar,section=circ20,eltype='T3D2')
P.elemProp(diabar,section=circ20,eltype='T3D2')

# Since all elems have same characteristics, we could just have used:
#   P.elemProp(section=circ20,elemtype='T3D2')
# But putting the elems in three sets allows for separate postprocessing 

#############
#Writing the inputfile
###################

step = Step()
out = Output(type='field',variable='preselect')
res = [ Result(kind='element',keys=['S']),
	Result(kind='node',keys=['U'])
	]
model = Model(nodes,elems)
message("Writing the Abaqus file")
AbqData(model,P,[step],out=[out],res=res).write('SpaceTruss')
