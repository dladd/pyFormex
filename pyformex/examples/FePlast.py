# $Id$ *** pyformex ***
##
##  This file is part of pyFormex 0.8.6  (Mon Jan 16 21:15:46 CET 2012)
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Home page: http://pyformex.org
##  Project page:  http://savannah.nongnu.org/projects/pyformex/
##  Copyright 2004-2011 (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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

"""FePlast

This example shows how to create a Finite Element model of a rectangular
steel plate, how to add material properties, boundary conditions and loads,
and how to export the resulting model as an input file for Abaqus(tm) or
Calculix.
"""
from __future__ import print_function
_status = 'checked'
_level = 'advanced'
_topics = ['FEA']
_techniques = ['properties', 'export'] 

from gui.draw import *

from plugins.fe import *
from plugins.properties import *
from plugins.fe_abq import Step,Output,Result,AbqData

import simple

def sortElemsByLoadedFace(ind):
    """Sorted a set of face loaded elements by the loaded face local number

    ind is a (nelems,2) array, where ind[:,0] are element numbers and
    ind[:,1] are the local numbers of the loaded faces

    Returns a dict with the loaded face number as key and a list of
    element numbers as value.
    """
    edgset = unique(ind[:,1])
    d = {}
    for e in edgset:
        d[e] = ind[where(ind[:,1]==e)[0],0]
    return d

        
def run():
    reset()
    clear()

    # Create a thin rectangular plate.
    # Because it is thin, we use a 2D model (in the xy plane.
    # We actually only model 1/4 of the plate'
    # The full plate could be found from mirroring wrt x and y axes.
    L = 400. # length of the plate (mm)
    B = 100. # width of the plate (mm)
    th = 10. # thickness of the plate (mm)
    L2,B2 = L/2, B/2  # dimensions of the quarter plate
    nl,nb = 40,10     # number of elements along length, width
    plate = simple.rectangle(nl,nb).toMesh() # Create the plate
    plate = plate.scale([L2/nl,B2/nb,1.])    # Give it the correct size
    draw(plate)

    # model is completely shown, keep camera bbox fixed 
    setDrawOptions({'bbox':'last','marksize':8})

    # Assemble the FEmodel (this may renumber the nodes!)
    FEM = Model(meshes=[plate])

    # Create an empty property database
    P = PropertyDB()

    # Define the material data: here we use an elsto-plastic model
    # for the steel
    steel = {
        'name': 'steel',
        'young_modulus': 207000,
        'poisson_ratio': 0.3,
        'density': 7.85e-9,
        'plastic' : [
            (305.45,       0.),
            (306.52, 0.003507),
            (308.05, 0.008462),
            (310.96,  0.01784),
            (316.2, 0.018275),
            (367.5, 0.047015),
            (412.5, 0.093317),
            (448.11, 0.154839),
            (459.6, 0.180101),
            (494., 0.259978),
            (506.25, 0.297659),
            (497., 0.334071),
            (482.8, 0.348325),
            (422.5, 0.366015),
            (399.58,   0.3717),
            (1.,  0.37363),
            ],
        }
    # Define the thin steel plate section
    steel_plate = { 
        'name': 'steel_plate',
        'sectiontype': 'solid',
        'thickness': th,
        'material': 'steel',   # Reference to the material name above
        }

    # Give the elements their properties: this is simple here because
    # all elements have the same properties. The element type is 
    # for an Abaqus plain stress quadrilateral element with 4 nodes.
    P.elemProp(name='Plate',eltype='CPS4',section=ElemSection(section=steel_plate,material=steel))

    # Set the boundary conditions
    # The xz and yz planes should be defined as symmetry planes.
    # First, we find the node numbers along the x, y axes:
    elsize = min(L2/nl,B2/nb)  # smallest size of elements
    tol = 0.001*elsize         # a tolerance to avoid roundoff errors
    nyz = FEM.coords.test(dir=0,max=tol)  # test for points in the yz plane
    nxz = FEM.coords.test(dir=1,max=tol)  # test for points in the xz plane
    nyz = where(nyz)[0]  # the node numbers passing the above test
    nxz = where(nxz)[0]
    draw(FEM.coords[nyz],color=cyan)
    draw(FEM.coords[nxz],color=green)
    # Define the boundary conditions
    P.nodeProp(tag='init',set=nyz,name='YZ_plane',bound='XSYMM')
    P.nodeProp(tag='init',set=nxz,name='XZ_plane',bound='YSYMM')

    # The plate is loaded by a uniform tensile stress in the x-direction
    # First we detect the border
    brd,ind = FEM.meshes()[0].getBorder(return_indices=True)
    BRD = Mesh(FEM.coords,brd).compact()
    draw(BRD,color=red,linewidth=2)
    xmax = BRD.bbox()[1][0]   # the maximum x coordinate
    loaded = BRD.test(dir=0,min=xmax-tol)
    # The loaded border elements
    loaded = where(loaded)[0]
    draw(BRD.select(loaded),color=blue,linewidth=4)
    sortedelems = sortElemsByLoadedFace(ind[loaded])
    # Define the load
    load = 10.0   # tensile load in MPa
    for face in sortedelems:
        P.elemProp(tag='step1',set=sortedelems[face],name='Loaded-%s'%face,dload=ElemLoad('P%s'%face,-load))

    # Print the property database
    P.print()

    # Create requests for output to the .fil file.
    # - the displacements in all nodes
    # - the stress components in all elements
    # - the stresses averaged at the nodes
    # - the principal stresses and stress invariants in the elements of part B.
    # (add output='PRINT' to get the results printed in the .dat file)
    res = [ Result(kind='NODE',keys=['U']),
            Result(kind='ELEMENT',keys=['S'],set='Plate'),
            Result(kind='ELEMENT',keys=['S'],pos='AVERAGED AT NODES',set='Plate'),
            Result(kind='ELEMENT',keys=['SP','SINV'],set='Plate'),
            ]

    # Define a loading step
    step1 = Step('STATIC',time=[1., 1., 0.01, 1.],tags=['step1'])

    data = AbqData(FEM,prop=P,steps=[step1],res=res,bound=['init'])

    if ack('Export this model in ABAQUS input format?',default='No'):
        fn = askNewFilename(filter='*.inp')
        if fn:
            data.write(jobname=fn,group_by_group=True)


if __name__ == 'draw':
    run()
    
# End
