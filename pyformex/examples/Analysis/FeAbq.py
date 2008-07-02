#!/usr/bin/env pyformex
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

from plugins.fe import *
from plugins.properties import *
from plugins.fe_abq import *

def quad():
    """Return a unit quadrilateral Formex."""
    return Formex(mpattern('123'))

def triquad():
    """Return a triangularized unit quadrilateral Formex."""
    return Formex(mpattern('12-34'))


na,ma = 4,2  # Size of domain A   
nb,mb = 3,4  # size of domain B
# Make sure the property numbers never clash!
pa,pb,pc = 3,4,5 # Properties corresponding to domains A,B,C
pb1 = 2 # Property for part of domain B
pbc,pld = 1,6 # Properties corresponding to boundary/loaded nodes

A = triquad().replic2(na,ma,1,1).setProp(pa)
B = quad().replic2(nb,mb,1,1).translate([na,0,0]).setProp(pb)
# Change every second element of B to property pb1
B.p[arange(B.p.size) % 2 == 1] = pb1
C = A.rotate(90).setProp(pc)
parts = [A,B,C]

# Create the finite element model
# A model contains a single set of nodes and one or more sets of elements
M = mergedModel(*[p.feModel() for p in parts])

# Create a Formex with the nodes, mostly for drawing
F = Formex(M.nodes)

def printModel(M):
    """print the model M"""
    print "===================\nMERGED MODEL"
    print "NODES"
    print M.nodes
    for i,e in enumerate(M.elems):
        print "PART %s" %i
        print e
    print "==================="

def drawModel(M,nodes=True,elems=True,nodenrs=True,elemnrs=True):
    """draw the model M"""
    smoothwire()
    lights(False)
    transparent()
    clear()
    print F.p
    if nodes or nodenrs:
##         F = Formex(M.nodes)
        if nodes:
            draw(F)
        if nodenrs:
            drawNumbers(F)
    if elems or elemnrs:
##         G = [Formex(M.nodes[e],i+1) for i,e in enumerate(M.elems)]
        G = parts
        if elems:
            draw(G)
        if elemnrs:
            [ drawNumbers(i) for i in G ]
    zoomAll()

# Transfer the properties from the parts in a global set
elemprops = concatenate([part.p for part in parts])

# Create the property database
P = PropertyDB()

# In this simple example, we do not use a material/section database,
# but define the data directly
steel = {
    'name': 'steel',
    'young_modulus': 207000,
    'poisson_ratio': 0.3,
    'density': 0.1, # Not Used, but Abaqus does not like a material without
    }
thin_plate = { 
    'name': 'thin_plate',
    'sectiontype': 'solid',
    'thickness': 0.01,
    'material': 'steel',
    }
medium_plate = { 
    'name': 'thin_plate',
    'sectiontype': 'solid',
    'thickness': 0.015,
    'material': 'steel',
    }
thick_plate = { 
    'name': 'thick_plate',
    'sectiontype': 'solid',
    'thickness': 0.02,
    'material': 'steel',
    }
print thin_plate
print medium_plate
print thick_plate

# Create element sets according to the properties pa,pb,pb1,pc:
esets = {}
esets.update([(v,where(elemprops==v)[0]) for v in [pa,pb,pb1,pc]])

# Set the element properties
P.elemProp(set=esets[pa],eltype='CPS3',section=ElemSection(section=thin_plate,material=steel))
P.elemProp(set=esets[pb],eltype='CPS4',section=ElemSection(section=thick_plate,material=steel))
P.elemProp(set=esets[pb1],eltype='CPS4',section=ElemSection(section=thick_plate,material=steel))
P.elemProp(set=esets[pc],eltype='CPS3',section=ElemSection(section=medium_plate,material=steel))

print "Element properties"
for p in P.getProp('e'):
    print p

# Set the nodal properties
xmin,xmax = M.nodes.bbox()[:,0]
bnodes = where(M.nodes.test(min=xmax-0.01))[0] # Right end nodes
lnodes = where(M.nodes.test(max=xmin+0.01))[0] # Left end nodes

print "Boundary nodes: %s" % bnodes
print "Loaded nodes: %s" % lnodes

P.nodeProp(tag='init',set=bnodes,bound=[1,1,0,0,0,0])
P.nodeProp(tag='step1',set=lnodes,setname='Loaded',cload=[-10.,0.,0.,0.,0.,0.])
P.nodeProp(tag='step2',set='Loaded',cload=[-10.,10.,0.,0.,0.,0.])

F.setProp(0)
F.p[bnodes] = pbc
F.p[lnodes] = pld

print "Node properties"
for p in P.getProp('n'):
    print p


drawModel(M,elems=False)
#exit()

while ack("Renumber nodes?"):
    # renumber the nodes randomly
    old,new = M.renumber()
    drawModel(M)
    
#exit()

# Request default output plus output of S in elements of part B.
# If the abqdata are written with group_by_group==True (see at bottom),
# all elements of each group in elems will be collected in a set named
# Eset('grp',index), where index is the index of the group in the elems list.
# Thus, all elements of part B will end up in Eset('grp',1)
out = [ Output(type='history'),
        Output(type='field'),
        Output(type='field',kind='element',set=Eset('grp',1),keys=['S']),
        ]

# Create requests for output to the .fil file.
# - the displacements in all nodes
# - the stress components in all elements
# - the stresses averaged at the nodes
# - the principal stresses and stress invariants in the elements of part B.
# (add output='PRINT' to get the results printed in the .dat file)
res = [ Result(kind='NODE',keys=['U']),
        Result(kind='ELEMENT',keys=['S']),
        Result(kind='ELEMENT',keys=['S'],pos='AVERAGED AT NODES'),
        Result(kind='ELEMENT',keys=['SP','SINV'],set=Eset('grp',1)),
        ]

# Define steps (default is static)
step1 = Step(time=[1., 1., 0.01, 1.],tags=['step1'])
step2 = Step(time=[1., 1., 0.01, 1.],tags=['step2'])

# collect all data
#
# !! currently output/result request are global to all steps
# !! this will be changed in future
#
all = AbqData(M,prop=P,steps=[step1,step2],out=out,res=res,bound=['init'])

if ack('Export this model in ABAQUS input format?'):
    fn = askFilename(filter='*.inp')
    if fn:
        all.write(jobname=fn,group_by_group=True)

# End
