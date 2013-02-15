# $Id$
##
##  This file is part of pyFormex 0.8.9  (Fri Nov  9 10:49:51 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2012 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""FeAbq

"""
from __future__ import print_function
_status = 'checked'
_level = 'advanced'
_topics = ['FEA']
_techniques = ['persistence', 'dialog', 'color'] 

from gui.draw import *

from plugins.fe import mergedModel
from plugins.properties import PropertyDB,ElemSection,Eset
from plugins.fe_abq import Step,Output,Result,AbqData


def base(nplex):
    """Return the pattern string to generate suqare base cells"""
    return {
        3: '3:012934',
        4: '4:0123',
        } [nplex]


def abq_eltype(nplex):
    """Return matching abaqus/calculix eltype for given pyFormex plexitude"""
    return {
        3 : 'CPS3',
        4 : 'CPS4',
        8 : 'CPS8',
        } [nplex]


def printModel(M):
    """print the model M"""
    print("===================\nMERGED MODEL")
    print("NODES")
    print(M.coords)
    for i,e in enumerate(M.elems):
        print("PART %s" %i)
        print(e)
    print("===================")


def drawFEModel(M,nodes=True,elems=True,nodenrs=True,elemnrs=True):
    """draw the model M"""
    clear()
    if nodes or nodenrs:
        if nodes:
            draw(M.coords)
        if nodenrs:
            drawNumbers(M.coords)
    if elems or elemnrs:
        meshes = [Mesh(M.coords,e,prop=i+1) for i,e in enumerate(M.elems)]
        if elems:
            draw(meshes)
        if elemnrs:
            [ drawNumbers(i,color=red) for i in meshes ]
    zoomAll()

def run():
    global parts,M,F
    
    na,ma = 4,3  # Size of domain A   
    nb,mb = 3,4  # size of domain B
    # Make sure the property numbers never clash!
    pa,pb,pc = 3,4,5 # Properties corresponding to domains A,B,C
    pb1 = 2 # Property for part of domain B
    pbc,pld = 1,6 # Properties corresponding to boundary/loaded nodes

    _choices = ["A mixture of tri3 and quad4","Only quad4","Only quad8"]
    ans = _choices.index(ask("What kind of elements do you want?",_choices))
    if ans == 0:
        baseA = Formex(base(3))
        baseB = Formex(base(4))
    else:
        baseA = Formex(base(4))
        baseB = Formex(base(4))

    A = baseA.replic2(na,ma,1,1).setProp(pa)
    B = baseB.replic2(nb,mb,1,1).translate([na,0,0]).setProp(pb)
    # Change every second element of B to property pb1
    B.prop[arange(B.prop.size) % 2 == 1] = pb1
    print(B.prop)
    C = A.rotate(90).setProp(pc)
    parts = [A,B,C]
    draw(parts)
    
    # First convert parts to Meshes
    parts = [ p.toMesh() for p in parts ]
    if ans == 2:
        # Convert meshes to 'quad8'
        parts = [ p.convert('quad8') for p in parts ]

    # Create the finite element model
    # A model contains a single set of nodes and one or more sets of elements
    M = mergedModel(parts)
    #drawFEModel(M)
    #return

    # Transfer the properties from the parts in a global set
    elemprops = concatenate([part.prop for part in parts])

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
    print(thin_plate)
    print(medium_plate)
    print(thick_plate)

    # Create element sets according to the properties pa,pb,pb1,pc:
    esets = {}
    esets.update([(v,where(elemprops==v)[0]) for v in [pa,pb,pb1,pc]])

    # Set the element properties
    ea,eb,ec = [ abq_eltype(p.nplex()) for p in parts ]
    P.elemProp(set=esets[pa],eltype=ea,section=ElemSection(section=thin_plate,material=steel))
    P.elemProp(set=esets[pb],eltype=eb,section=ElemSection(section=thick_plate,material=steel))
    P.elemProp(set=esets[pb1],eltype=eb,section=ElemSection(section=thick_plate,material=steel))
    P.elemProp(set=esets[pc],eltype=ec,section=ElemSection(section=medium_plate,material=steel))

    print("Element properties")
    for p in P.getProp('e'):
        print(p)

    # Set the nodal properties
    xmin,xmax = M.coords.bbox()[:,0]
    bnodes = where(M.coords.test(min=xmax-0.01))[0] # Right end nodes
    lnodes = where(M.coords.test(max=xmin+0.01))[0] # Left end nodes

    print("Boundary nodes: %s" % bnodes)
    print("Loaded nodes: %s" % lnodes)

    P.nodeProp(tag='init',set=bnodes,bound=[1,1,0,0,0,0])
    P.nodeProp(tag='step1',set=lnodes,name='Loaded',cload=[-10.,0.,0.,0.,0.,0.])
    P.nodeProp(tag='step2',set='Loaded',cload=[-10.,10.,0.,0.,0.,0.])

    # Coloring the nodes gets easier using a Formex
    F = Formex(M.coords)
    F.setProp(0)
    F.prop[bnodes] = pbc
    F.prop[lnodes] = pld

    print("Node properties")
    for p in P.getProp('n'):
        print(p)

    while ack("Renumber nodes?"):
        # renumber the nodes randomly
        print([e.eltype  for e in M.elems])
        old,new = M.renumber()
        print(old,new)
        print([e for e in M.elems])
        print([e.eltype  for e in M.elems])
        drawFEModel(M)
        if widgets.input_timeout > 0:
            break

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
    step1 = Step(time=[1., 1., 0.01, 1.],tags=['step1'],out=out,res=res)
    step2 = Step(time=[1., 1., 0.01, 1.],tags=['step2'],out=out,res=res)

    # collect all data
    #
    # !! currently output/result request are global to all steps
    # !! this will be changed in future
    #
    data = AbqData(M,prop=P,steps=[step1,step2],bound=['init'])

    if ack('Export this model in ABAQUS input format?',default='No'):
        fn = askNewFilename(filter='*.inp')
        if fn:
            data.write(jobname=fn,group_by_group=True)

# Initialize
smoothwire()
lights(False)
transparent()
clear()

if __name__ == 'draw':
    run()
    
# End
