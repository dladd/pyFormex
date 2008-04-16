#!/usr/bin/env pyformex
# $Id$

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
pb,pl = 1,6 # Properties corresponding to boundary/loaded nodes

A = triquad().replic2(na,ma,1,1).setProp(pa)
B = quad().replic2(nb,mb,1,1).translate([na,0,0]).setProp(pb)
# Change every second element of B to property pb1
B.p[arange(B.p.size) % 2 == 1] = pb1
C = A.rotate(90).setProp(pc)
parts = [A,B,C]


draw(parts)
NRS = [ drawNumbers(i) for i in parts ]
zoomAll()

# NEW: the parts flagged NEW use the new properties DB
P = PropertiesDB()

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

# Attribute the element properties
ElemProperty(pa,ElemSection(section=thin_plate,material=steel),elemtype='CPS3')
ElemProperty(pb,ElemSection(section=thick_plate,material=steel),elemtype='CPS4')
ElemProperty(pb1,ElemSection(section=thick_plate,material=steel),elemtype='CPS4')
ElemProperty(pc,ElemSection(section=medium_plate,material=steel),elemtype='CPS3')

# Create the finite element model
femodels = [part.feModel() for part in parts]
nodes,elems = mergeModels(femodels)
print "===================\nMERGED MODEL"
print "NODES"
print nodes
for i,e in enumerate(elems):
    print "PART %s" %i
    print e
print "==================="
#nodes,index = mergeNodes([part.feModel()[0] for part in parts])
elemprops = concatenate([part.p for part in parts])

# Set the nodal properties
xmin,xmax = nodes.bbox()[:,0]
bnodes = where(nodes.test(min=xmax-0.01))[0] # Right end nodes
lnodes = where(nodes.test(max=xmin+0.01))[0] # Left end nodes

print "Boundary nodes: %s" % bnodes
print "Loaded nodes: %s" % lnodes

F = Formex(nodes).setProp(0) # To visualize the node properties
F.p[bnodes] = pb
F.p[lnodes] = pl
draw(F,marksize=8)
NRN = drawNumbers(F)

# The load/bc in the nodes
NodeProperty(pl,cload=[-10.,0.,0.,0.,0.,0.])
NodeProperty(pb,bound=[1,1,0,0,0,0])

# An alternative is to include the node sets explicitely:
# (the record numbers could be set to any unique number here)
# Remark that the following definitions do not create duplicate properties,
# but overwrite (destroy) the above definitions
NodeProperty(pl,nset=where(F.p==pl)[0],cload=[-10.,0.,0.,0.,0.,0.])
NodeProperty(pb,nset=where(F.p==pb)[0],bound=[1,1,0,0,0,0])


# Loads in the second step
NodeProperty(1000,nset=where(F.p==pl)[0],cload=[-10.,10.,0.,0.,0.,0.])

#NEW
P.nodeProp(nset=where(F.p==pb)[0],bound=[1,1,0,0,0,0])
P.nodeProp(tag='step1',nset=where(F.p==pl)[0],cload=[-10.,0.,0.,0.,0.,0.])
P.nodeProp(tag='step2',nset=where(F.p==pl)[0],cload=[-10.,10.,0.,0.,0.,0.])

print P.getProp('n')

# Create the Abaqus model
# A model contains a single set of nodes, one or more sets of elements,
# a global node property set and a global element property set.
#
model = Model(nodes, elems, F.p , elemprops)


# ask default output plus output of S in elements of part B
out = [ Output(type='history'),
        Output(type='field'),
        Output(type='field',kind='element',set=[pb],keys=['S']),
        ]

# Create output request to the .fil file.
# In this case we request :
# - the displacements in all nodes
# - the stress components in all elements
# - the principal stresses and stress invariants in the elements of part B.
# (add output='PRINT' to get the results printed in the .dat file)
res = [ Result(kind='NODE',keys=['U']),
        Result(kind='ELEMENT',keys=['S']),
        Result(kind='ELEMENT',keys=['S'],pos='AVERAGED AT NODES'),
        Result(kind='ELEMENT',keys=['SP','SINV'],set=[pb]),
        ]

# Static(default) step
step1 = Step(time=[1., 1., 0.01, 1.],cloadset=[pl])
step2 = Step(time=[1., 1., 0.01, 1.],cloadset=[1000])

# collect all data
#
# !! currently output/result request are global to all steps
# !! this will be changed in future
#
all = AbqData(model,[step1,step2],out=out,res=res)

if ack('Export this model in ABAQUS input format?'):
    fn = askFilename(filter='*.inp')
    if fn:
        all.write(jobname=fn)

# End
