#!/usr/bin/env python pyformex.py
# $Id: SpaceTrussRoof_f2abq.py 150 2006-11-01 11:13:34Z bverheg $
##
## This file is part of pyFormex 0.5 Release Mon Jul 30 13:38:48 2007
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""Double Layer Flat Space Truss Roof"""

####
#Data
###################

dx = 1800 # Modular size [mm]
ht = 900  # Deck height [mm]
nx = 4     # number of bottom deck modules in x direction 
ny = 5   # number of bottom deck modules in y direction 

q = -0.005 #distributed load [N/mm^2]


#######
#Creating the model
###################

top = (Formex("1").replic2(nx-1,ny,1,1) + Formex("2").replic2(nx,ny-1,1,1)).scale(dx)
top.setProp(3)
bottom = (Formex("1").replic2(nx,ny+1,1,1) + Formex("2").replic2(nx+1,ny,1,1)).scale(dx).translate([-dx/2,-dx/2,-ht])
bottom.setProp(0)
T0 = Formex(4*[[[0,0,0]]]) 				# 4 times the corner of the top deck
T4 = bottom.select([0,1,nx,nx+1]) 	# 4 nodes of corner module of bottom deck
dia = connect([T0,T4]).replic2(nx,ny,dx,dx)
dia.setProp(1)

F = (top+bottom+dia)

clear();draw(F)


############
#Creating FE-model
###################

nodes,elems=F.feModel()
nnod=nodes.shape[0]


###############
#Creating nodeprops-list
###################

nlist=arange(nnod)
count = zeros(nnod)
for n in elems.flat:
	count[n] += 1
field = nlist[count==8]
topedge = nlist[count==7]
topcorner = nlist[count==6]
bottomedge = nlist[count==5]
bottomcorner = nlist[count==3]

nodeprops=zeros(nnod)
nodeprops[field]=1
nodeprops[bottomedge]=0
nodeprops[bottomcorner]=0
nodeprops[topedge]=3
nodeprops[topcorner]=3


########################
#Defining and assigning the properties
#############################

from plugins.properties import *
from plugins.f2abq import *

Q = 0.5*q*dx*dx
support = NodeProperty(0, bound = [1,1,1,0,0,0])
edge = NodeProperty(3,cload = [0,0,Q/2,0,0,0])
loaded = NodeProperty(1,cload = [0,0,Q,0,0,0])

circ20 = ElemSection(section={'name':'circ20','radius':10, 'cross_section':314.159}, material={'name':'S500', 'young_modulus':210000, 'shear_modulus':81000, 'poisson_ratio':0.3, 'yield_stress' : 500,'density':0.000007850}, sectiontype='Circ')
diabar = ElemProperty(1,elemsection = circ20, elemtype='T3D2')
bottombar = ElemProperty(0,elemsection = circ20, elemtype='T3D2')
topbar = ElemProperty(3,elemsection = circ20, elemtype='T3D2')


#############
#Writing the inputfile
###################

step = Analysis()
db = Odb(type='field', variable='preselect')
forcedata = Dat(kind='element', ID=['S'])
model = Model(nodes, elems, nodeprops, F.p)
total = AbqData(model, [step], odb= [db], dat=[forcedata])
message("Writing the Abaqus file")
writeAbqInput(total,job='SpaceTruss')
