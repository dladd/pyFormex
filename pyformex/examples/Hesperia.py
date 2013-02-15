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

"""Hesperia

"""
from __future__ import print_function
_status = 'unchecked'
_level = 'advanced'
_topics = ['geometry', 'FEA', 'domes', 'surface']
_techniques = ['menu', 'dialog', 'persistence', 'color'] 

from gui.draw import *

import simple,utils
from connectivity import Connectivity
from plugins.trisurface import TriSurface
from plugins.properties import *
from plugins.fe_abq import *
from gui.colorscale import ColorScale,ColorLegend
from gui import menu,decors
import time


def howto():
    showInfo("""   How to use this menu?

1. If you want to save your work, start by opening a new project (File menu).

2. Create the geometry: it will be put in a Formex named 'F'.

3. Add (or read) properties to be used for the snow loading: enter a property number, then select the corresponding facets. Save the properties if you want to use them again later.

4. Create a Finite Element model : either shell or frame.

5. Perform the calculation:

   For a shell model:
     5a. Create an Abaqus input file
     5b. Send the job to the cluster (job menu)
     5c. Get the results back when finished (job menu)

   For a frame model:
     5a. Directly calculate using the CALPY module

6. Postprocess:

   For an Abaqus job: use the postproc menu

   For a Calpy job: use the hesperia menu
""")

         
def createGeometry():
    global F
    # Construct a triangle of an icosahedron oriented with a vertex in
    # the y-direction, and divide its edges in n parts
    n = 3

    # Add a few extra rows to close the gap after projection
    nplus = n+3

    clear()
    smoothwire()
    view('front')
    # Start with an equilateral triangle in the x-y-plane
    A = simple.triangle()
    A.setProp(1)
    draw(A)

    # Modular size
    a,b,c = A.sizes()
    pf.message("Cell width: %s; height: %s" % (a,b))

    # Create a mirrored triangle
    B = A.reflect(1)
    B.setProp(2)
    draw(B)

    # Replicate nplus times in 2 directions to create triangular pattern
    F = A.replic2(1,nplus,a,-b,0,1,bias=-a/2,taper=1)
    G = B.replic2(1,nplus-1,a,-b,0,1,bias=-a/2,taper=1)

    clear()
    F += G
    draw(F)

    # Get the top vertex and make it the origin
    P = F[0,-1]
    draw(Formex([P]),bbox='last')
    F = F.translate(-P)
    draw(F)

    # Now rotate around the x axis over an angle so that the projection on the
    # x-y plane is an isosceles triangle with top angle = 360/5 = 72 degrees.
    # The base angles thus are (180-72)/2 = 54 degrees.

    # Ratio of the height of the isosceles triangle over the icosaeder edge length.
    c = 0.5*tand(54.)
    angle = arccosd(tand(54.)/sqrt(3.))
    pf.message("Rotation Ratio: %s; Angle: %s degrees" % (c,angle)) 
    F = F.rotate(angle,0)
    clear()
    draw(F,colormap=['black','magenta','yellow','black'])

    # Project it on the circumscribing sphere
    # The sphere has radius ru
    golden_ratio = 0.5 * (1. + sqrt(5.))
    ru = 0.5 * a * sqrt(golden_ratio * sqrt(5.))
    pf.message("Radius of circumscribed sphere: %s" % ru)

    ru *= n
    C = [0.,0.,-ru]
    F = F.projectOnSphere(ru,center=C)
    draw(F)

    hx,hy,h = F.sizes()
    pf.message("Height of the dome: %s" % h)

    # The base circle goes through bottom corner of n-th row,
    # which will be the first point of the first triangle of the n-th row.
    # Draw the point to check it.
    
    i = (n-1)*n/2
    P = F[i][0]
    draw(Formex([P]),marksize=10,bbox='last')

    # Get the radius of the base circle from the point's coordinates
    x,y,z = P
    rb = sqrt(x*x+y*y)

    # Give the base points a z-coordinate 0
    F = F.translate([0.,0.,-z])
    clear()
    draw(F)
 
    # Draw the base circle
    H = simple.circle().scale(rb)
    draw(H)

    # Determine intersections with base plane
    P = [0.,0.,0.]
    N = [0.,0.,1.]

    newprops = [ 5,6,6,None,4,None,None ]
    F = F.cutWithPlane(P,N,side='+',newprops=newprops)#,atol=0.0001)
    #clear()
    draw(F)

    # Finally, create a rosette to make the circle complete
    # and rotate 90 degrees to orient it like in the paper
    clear()
    F = F.rosette(5,72.).rotate(90)

    def cutOut(F,c,r):
        """Remove all elements of F contained in a sphere (c,r)"""
        d = F.distanceFromPoint(c)
        return F.select((d < r).any(axis=-1) == False)

    # Cut out the door: remove all members having a point less than
    # edge-length a away from the base point
    p1 = [rb,0.,0.]
    F = cutOut(F,p1,1.1*a*n/6)      # a was a good size with n = 6

    # Scale to the real geometry
    scale = 7000. / F.sizes()[2]
    pf.message("Applying scale factor %s " % scale)
    print(F.bbox())
    F = F.scale(scale)
    print(F.bbox())

    clear()
    draw(F,alpha=0.4)
    export({'F':F})


def assignProperties():
    """Assign properties to the structure's facets"""
    # make sure we have only one actor
    clear()
    FA = draw(F)
    #drawNumbers(F)
    p = 0
    while True:
        res = askItems([('Property',p)])
        if not res:
            break

        p = res['Property']
        sel = pickElements()
        if 0 in sel:
            pf.debug("PICKED NUMBERS:%s" % sel)
            F.prop[sel[0]] = p
        undraw(FA)
        FA = draw(F,bbox='last')


def exportProperties():
    """Save the current properties under a name"""
    res = askItems([('Property Name','p')])
    if res:
        p = res['Property Name']
        if not p.startswith('prop:'):
            p = "prop:%s" % p
        export({p:F.prop})


def selectProperties():
    """Select one of the saved properties"""
    res = askItems([('Property Name','p')])
    if res:
        p = res['Property Name']
        if p in pf.PF:
            F.setProp(pf.PF[p])


def saveProperties(fn = None):
    """Save the current properties."""
    if not fn:
        fn = askNewFilename(dirname,filter="Property files (*.prop)")
    if fn:
        F.prop.tofile(fn,sep=',')

        
def readProperties(fn = None):
    """Read properties from file."""
    if not fn:
        fn = askFilename(filter="Property files (*.prop)")
    if fn:
        p = fromfile(fn,sep=',')
        F.setProp(p)
        clear()
        draw(F)


def connections(elems):
    """Create lists of connections to lower entities.

    Elems is an array giving the numbers of lower entities.
    The result is a sequence of maxnr+1 lists, where maxnr is the
    highest lower entity number. Each (possibly empty) list contains
    the numbers of the rows of elems that contain (at least) one value
    equal to the index of the list.
    """
    return [ (i,list(where(elems==i)[0])) for i in unique(elems.flat) ]
    

#####################################################################

def createFrameModel():
    """Create the Finite Element Model.

    It is supposed here that the Geometry has been created and is available
    as a global variable F.
    """
    wireframe()
    lights(False)
    
    # Turn the Formex structure into a TriSurface
    # This guarantees that element i of the Formex is element i of the TriSurface
    S = TriSurface(F)
    nodes = S.coords
    elems = S.elems  # the triangles

    # Create edges and faces from edges
    print("The structure has %s nodes, %s edges and %s faces" % (S.ncoords(),S.nedges(),S.nfaces()))

    # Remove the edges between to quad triangles
    drawNumbers(S.coords)
    quadtri = where(S.prop==6)[0]
    nquadtri = quadtri.shape[0]
    print("%s triangles are part of quadrilateral faces" % nquadtri)
    faces = S.getElemEdges()[quadtri]
    cnt,ind,xbin = histogram2(faces.reshape(-1),arange(faces.max()+1))
    rem = where(cnt==2)[0]
    print("Total edges %s" % len(S.edges))
    print("Removing %s edges" % len(rem))
    edges = S.edges[complement(rem,n=len(S.edges))]
    print("Remaining edges %s" % len(edges))

    # Create the steel structure
    E = Formex(nodes[edges])
    clear()
    draw(E)


    warning("Beware! This script is currently under revision.")
    
    conn = connections(quadtri)
    print(conn)

    # Filter out the single connection edges
    internal = [ c[0] for c in conn if len(c[1]) > 1 ]
    print("Internal edges in quadrilaterals: %s" % internal)
    
    E = Formex(nodes[edges],1)
    E.prop[internal] = 6
    wireframe()
    clear()
    draw(E)

    # Remove internal edges
    tubes = edges[E.prop != 6]

    print("Number of tube elements after removing %s internals: %s" % (len(internal),tubes.shape[0]))

    D = Formex(nodes[tubes],1)
    clear()
    draw(D)

    # Beam section and material properties
    b = 60
    h = 100
    t = 4
    b1 = b-2*t
    h1 = h-2*t
    A = b*h - b1*h1
    print(b*h**3)
    I1 = (b*h**3 - b1*h1**3) / 12
    I2 = (h*b**3 - h1*b1**3) / 12
    I12 = 0
    J = 4 * A**2 / (2*(b+h)/t)

    tube = { 
        'name':'tube',
        'cross_section': A,
        'moment_inertia_11': I1,
        'moment_inertia_22': I2,
        'moment_inertia_12': I12,
        'torsional_constant': J
        }
    steel = {
        'name':'steel',
        'young_modulus' : 206000,
        'shear_modulus' : 81500,
        'density' : 7.85e-9,
        }
    print(tube)
    print(steel)

    tubesection = ElemSection(section=tube,material=steel)

    # Calculate the nodal loads

    # Area of triangles
    area,normals = S.areaNormals()
    print("Area:\n%s" % area)
    # compute bar lengths
    bars = nodes[tubes]
    barV = bars[:,1,:] - bars[:,0,:]
    barL = sqrt((barV*barV).sum(axis=-1))
    print("Member length:\n%s" % barL)


    ### DEFINE LOAD CASE (ask user) ###
    res = askItems(
        [ _I('Steel',True),
          _I('Glass',True),
          _I('Snow',False),
          _I('Solver',choices=['Calpy','Abaqus']),
          ])
    if not res:
        return

    nlc = 0
    for lc in [ 'Steel','Glass','Snow' ]:
        if res[lc]:
            nlc += 1 
    NODLoad = zeros((nlc,S.ncoords(),3))

    nlc = 0
    if res['Steel']:
        # the STEEL weight
        lwgt = steel['density'] * tube['cross_section'] * 9810  # mm/s**2
        print("Weight per length %s" % lwgt)
        # assemble steel weight load
        for e,L in zip(tubes,barL):
            NODLoad[nlc,e] += [ 0., 0., - L * lwgt / 2 ]
        nlc += 1
        
    if res['Glass']:
        # the GLASS weight
        wgt = 450e-6 # N/mm**2
        # assemble uniform glass load
        for e,a in zip(S.elems,area):
            NODLoad[nlc,e] += [ 0., 0., - a * wgt / 3 ]
        nlc += 1
        
    if res['Snow']:
        # NON UNIFORM SNOW
        fn = '../data/hesperia-nieve.prop'
        snowp = fromfile(fn,sep=',')
        snow_uniform = 320e-6 # N/mm**2
        snow_non_uniform = { 1:333e-6, 2:133e-6, 3:133e-6, 4:266e-6, 5:266e-6, 6:667e-6 }

        # assemble non-uniform snow load
        for e,a,p in zip(S.elems,area,snowp):
            NODLoad[nlc,e] += [ 0., 0., - a * snow_non_uniform[p] / 3]
        nlc += 1

    # For Abaqus: put the nodal loads in the properties database
    print(NODLoad)
    PDB = PropertyDB()
    for lc in range(nlc):
        for i,P in enumerate(NODLoad[lc]):
            PDB.nodeProp(tag=lc,set=i,cload=[P[0],P[1],P[2],0.,0.,0.])

    # Get support nodes
    botnodes = where(isClose(nodes[:,2], 0.0))[0]
    bot = nodes[botnodes]
    pf.message("There are %s support nodes." % bot.shape[0])

    # Upper structure
    nnodes = nodes.shape[0]              # node number offset
    ntubes = tubes.shape[0]              # element number offset
    
    PDB.elemProp(set=arange(ntubes),section=tubesection,eltype='FRAME3D')    
    
    # Create support systems (vertical beams)
    bot2 = bot + [ 0.,0.,-200.]         # new nodes 200mm below bot
    botnodes2 = arange(botnodes.shape[0]) + nnodes  # node numbers
    nodes = concatenate([nodes,bot2])
    supports = column_stack([botnodes,botnodes2])
    elems = concatenate([tubes,supports])
    ## !!!
    ## THIS SHOULD BE FIXED !!!
    supportsection = ElemSection(material=steel,section={ 
        'name':'support',
        'cross_section': A,
        'moment_inertia_11': I1,
        'moment_inertia_22': I2,
        'moment_inertia_12': I12,
        'torsional_constant': J
        })
    PDB.elemProp(set=arange(ntubes,elems.shape[0]),section=supportsection,eltype='FRAME3D')

    # Finally, the botnodes2 get the support conditions
    botnodes = botnodes2

##     # Radial movement only
##     np_fixed = NodeProperty(1,bound=[0,1,1,0,0,0],coords='cylindrical',coordset=[0,0,0,0,0,1])
    
##     # No movement, since we left out the ring beam
##     for i in botnodes:
##         NodeProperty(i,bound=[1,1,1,0,0,0],coords='cylindrical',coordset=[0,0,0,0,0,1])

##     np_central_loaded = NodeProperty(3, displacement=[[1,radial_displacement]],coords='cylindrical',coordset=[0,0,0,0,0,1])
##     #np_transf = NodeProperty(0,coords='cylindrical',coordset=[0,0,0,0,0,1])
    
    # Draw the supports
    S = connect([Formex(bot),Formex(bot2)])
    draw(S,color='black')

    if res['Solver'] == 'Calpy':
        fe_model = Dict(dict(solver='Calpy',nodes=nodes,elems=elems,prop=PDB,loads=NODLoad,botnodes=botnodes,nsteps=nlc))
    else:
        fe_model = Dict(dict(solver='Abaqus',nodes=nodes,elems=elems,prop=PDB,botnodes=botnodes,nsteps=nlc))
    export({'fe_model':fe_model})
    print("FE model created and exported as 'fe_model'")


#################### SHELL MODEL ########################################
def createShellModel():
    """Create the Finite Element Model.

    It is supposed here that the Geometry has been created and is available
    as a global variable F.
    """
    
    # Turn the Formex structure into a TriSurface
    # This guarantees that element i of the Formex is element i of the TriSurface
    S = TriSurface(F)
    print("The structure has %s nodes, %s edges and %s faces" % (S.ncoords(),S.nedges(),S.nfaces()))
    nodes = S.coords
    elems = S.elems  # the triangles

    clear()
    draw(F)

    # Shell section and material properties
    # VALUES SHOULD BE SET CORRECTLY

    glass_plate = { 
        'name': 'glass_plate',
        'sectiontype': 'shell',
        'thickness': 18,
        'material': 'glass',
        }
    glass = {
        'name': 'glass',
        'young_modulus': 72000,
        'shear_modulus': 26200,
        'density': 2.5e-9,        # T/mm**3
        }
    print(glass_plate)
    print(glass)
    glasssection = ElemSection(section=glass_plate,material=glass)

    PDB = PropertyDB()
    # All elements have same property:
    PDB.elemProp(set=arange(len(elems)),section=glasssection,eltype='STRI3')    

    # Calculate the nodal loads

    # Area of triangles
    area,normals = S.areaNormals()
    print("Area:\n%s" % area)

    ### DEFINE LOAD CASE (ask user) ###
    res = askItems([('Glass',True),('Snow',False)])
    if not res:
        return

    step = 0
    if res['Glass']:
        step += 1
        NODLoad = zeros((S.ncoords(),3))
        # add the GLASS weight
        wgt = 450e-6 # N/mm**2
        # Or, calculate weight from density:
        # wgt = glass_plate['thickness'] * glass['density'] * 9810 
        # assemble uniform glass load
        for e,a in zip(S.elems,area):
            NODLoad[e] += [ 0., 0., - a * wgt / 3 ]
        # Put the nodal loads in the properties database
        for i,P in enumerate(NODLoad):
            PDB.nodeProp(tag=step,set=i,cload=[P[0],P[1],P[2],0.,0.,0.])

    if res['Snow']:
        step += 1
        NODLoad = zeros((S.ncoords(),3))
        # add NON UNIFORM SNOW
        fn = '../data/hesperia-nieve.prop'
        snowp = fromfile(fn,sep=',')
        snow_uniform = 320e-6 # N/mm**2
        snow_non_uniform = { 1:333e-6, 2:133e-6, 3:133e-6, 4:266e-6, 5:266e-6, 6:667e-6 }
        # assemble non-uniform snow load
        for e,a,p in zip(S.elems,area,snowp):
            NODLoad[e] += [ 0., 0., - a * snow_non_uniform[p] / 3]
        # Put the nodal loads in the properties database
        for i,P in enumerate(NODLoad):
            PDB.nodeProp(tag=step,set=[i],cload=[P[0],P[1],P[2],0.,0.,0.])

    # Get support nodes
    botnodes = where(isClose(nodes[:,2], 0.0))[0]
    bot = nodes[botnodes].reshape((-1,1,3))
    pf.message("There are %s support nodes." % bot.shape[0])

    botofs = bot + [ 0.,0.,-0.2]
    bbot2 = concatenate([bot,botofs],axis=1)
    print(bbot2.shape)
    S = Formex(bbot2)
    draw(S)
    
##     np_central_loaded = NodeProperty(3, displacement=[[1,radial_displacement]],coords='cylindrical',coordset=[0,0,0,0,0,1])
##     #np_transf = NodeProperty(0,coords='cylindrical',coordset=[0,0,0,0,0,1])

##     # Radial movement only
##     np_fixed = NodeProperty(1,bound=[0,1,1,0,0,0],coords='cylindrical',coordset=[0,0,0,0,0,1])
    
    # Since we left out the ring beam, we enforce no movement at the botnodes
    bc = PDB.nodeProp(set=botnodes,bound=[1,1,1,0,0,0],csys=CoordSystem('C',[0,0,0,0,0,1]))

    # And we record the name of the bottom nodes set
    botnodeset = Nset(bc.nr)

    fe_model = Dict(dict(nodes=nodes,elems=elems,prop=PDB,botnodeset=botnodeset,nsteps=step))
    export({'fe_model':fe_model})
    smooth()
    lights(False)


#####################################################################
#### Analyze the structure using Abaqus ####
    
def createAbaqusInput():
    """Write the Abaqus input file.

    It is supposed that the Finite Element model has been created and
    exported under the name 'fe_model'.
    """
    checkWorkdir()
    try:
        FE = named('fe_model')
        nodes = FE.nodes
        elems = FE.elems
        prop = FE.prop
        nsteps = FE.nsteps
    except:
        warning("I could not find the finite element model.\nMaybe you should try to create it first?")
        return
    
    # ask job name from user
    res = askItems([('JobName','hesperia_shell')])
    if not res:
        return

    jobname = res['JobName']
    if not jobname:
        print("No Job Name: writing to sys.stdout")
        jobname = None

    out = [ Output(type='history'),
            Output(type='field'),
            ]

    res = [ Result(kind='NODE',keys=['U','COORD']),
            Result(kind='ELEMENT',keys=['S'],pos='AVERAGED AT NODES'),
            Result(kind='ELEMENT',keys=['SINV'],pos='AVERAGED AT NODES'),
            Result(kind='ELEMENT',keys=['SF'],pos='AVERAGED AT NODES'),
            ]

    step1 = Step(time=[1.,1.,0.01,1.],nlgeom='no',tags=[1])
    step2 = Step(time=[1.,1.,0.01,1.],nlgeom='no',tags=[2])
    
    model = Model(nodes,elems)

    AbqData(model,prop,[step1,step2],out=out,res=res).write(jobname)


#############################################################################
#### perform analysis with the calpy module ####

def runCalpyAnalysis():
    """Create data for Calpy analysis module and run Calpy on the data.

    While we could write an analysis file in the Calpy format and then
    run the Calpy program on it (like we did with Abaqus), we can (and do)
    take another road here: Calpy has a Python/numpy interface, allowing
    us to directly present the numerical data in arrays to the analysis
    module.
    It is supposed that the Finite Element model has been created and
    exported under the name 'fe_model'.
    """
    checkWorkdir()

    ############################
    # Load the needed calpy modules
    # You can prepend your own path here to override the installed calpy
    # sys.path[0:0] = ['/home/bene/prj/calpy']
    from plugins import calpy_itf
    calpy_itf.check()
    import calpy
    print(calpy)
    calpy.options.optimize=True
    from calpy import fe_util,beam3d
    ############################

    try:
        FE = named('fe_model')
##         print FE.keys()
##         nodes = FE.nodes
##         elems = FE.elems
##         prop = FE.prop
##         nodloads = FE.loads
##         botnodes = FE.botnodes
##         nsteps = FE.nsteps
    except:
        warning("I could not find the finite element model.\nMaybe you should try to create it first?")
        return
    
    # ask job name from user
    res = askItems([('JobName','hesperia_frame'),('Verbose Mode',False)])
    if not res:
        return

    jobname = res['JobName']
    if not jobname:
        print("No Job Name: bailing out")
        return
    verbose = res['Verbose Mode']
   
    nnod = FE.nodes.shape[0]
    nel = FE.elems.shape[0]
    print("Number of nodes: %s" % nnod)
    print("Number of elements: %s" % nel)

    # Create an extra node for beam orientations
    #
    # !!! This is ok for the support beams, but for the structural beams
    # !!! this should be changed to the center of the sphere !!!
    extra_node = array([[0.0,0.0,0.0]])
    coords = concatenate([FE.nodes,extra_node])
    nnod = coords.shape[0]
    print("Adding a node for orientation: %s" % nnod)

    # We extract the materials/sections from the property database
    matprops = FE.prop.getProp(kind='e',attr=['section'])
    
    # Beam Properties in Calpy consist of 7 values:
    #   E, G, rho, A, Izz, Iyy, J
    # The beam y-axis lies in the plane of the 3 nodes i,j,k.
    mats = array([[mat.young_modulus,
                  mat.shear_modulus,
                  mat.density,
                  mat.cross_section,
                  mat.moment_inertia_11,
                  mat.moment_inertia_22,
                  mat.moment_inertia_12,
                  ] for mat in matprops]) 
    if verbose:
        print("Calpy.materials")
        print(mats)
    
    # Create element definitions:
    # In calpy, each beam element is represented by 4 integer numbers:
    #    i j k matnr,
    # where i,j are the node numbers,
    # k is an extra node for specifying orientation of beam (around its axis),
    # matnr refers to the material/section properties (i.e. the row nr in mats)
    # Also notice that Calpy numbering starts at 1, not at 0 as customary
    # in pyFormex; therefore we add 1 to elems.
    # The third node for all beams is the last (extra) node, numbered nnod.
    # We need to reshape tubeprops to allow concatenation
    matnr = zeros(nel,dtype=int32)
    for i,mat in enumerate(matprops):  # proces in same order as above!
        matnr[mat.set] = i+1
    elements = concatenate([FE.elems + 1,         # the normal node numbers
                            nnod * ones(shape=(nel,1),dtype=int), # extra node  
                            matnr.reshape((-1,1))],  # mat number
                           axis=1)
  
    if verbose:
        print("Calpy.elements")
        print(elements)

    # Boundary conditions
    # While we could get the boundary conditions from the node properties
    # database, we will formulate them directly from the numbers
    # of the supported nodes (botnodes).
    # Calpy (currently) only accepts boundary conditions in global
    # (cartesian) coordinates. However, as we only use fully fixed
    # (though hinged) support nodes, that presents no problem here.
    # For each supported node, a list of 6 codes can (should)be given,
    # corresponding to the six Degrees Of Freedom (DOFs): ux,uy,uz,rx,ry,rz.
    # The code has value 1 if the DOF is fixed (=0.0) and 0 if it is free.
    # The easiest way to set the correct boundary conditions array for Calpy
    # is to put these codes in a text field and have them read with
    # ReadBoundary.
    s = ""
    for n in FE.botnodes + 1:   # again, the +1 is to comply with Calpy numbering!
        s += "  %d  1  1  1  1  1  1\n" % n    # a fixed hinge
    # Also clamp the fake extra node
    s += "  %d  1  1  1  1  1  1\n" % nnod
    if verbose:
        print("Specified boundary conditions")
        print(s)
    bcon = fe_util.ReadBoundary(nnod,6,s)
    fe_util.NumberEquations(bcon)
    if verbose:
        print("Calpy.DOF numbering")
        print(bcon) # all DOFs are numbered from 1 to ndof

    # The number of free DOFs remaining
    ndof = bcon.max()
    print("Number of DOF's: %s" % ndof)

    # Create load vectors
    # Calpy allows for multiple load cases in a single analysis.
    # However, our script currently puts all loads together in a single
    # load case. So the processing hereafter is rather simple, especially
    # since Calpy provides a function to assemble a single concentrated
    # load into the load vector. We initialize the load vector to zeros
    # and then add all the concentrated loads from the properties database.
    # A single concentrated load consists of 6 components, corresponding
    # to the 6 DOFs of a node.
    #
    # AssembleVector takes 3 arguments: the global vector in which to
    # assemble a nodal vector (length ndof), the nodal vector values
    # (length 6), and a list of indices specifying the positions of the
    # nodal DOFs in the global vector.
    # Beware: The function does not change the global vector, but merely
    # returns the value after assembling.
    # Also notice that the indexing inside the bcon array uses numpy
    # convention (starting at 0), thus no adding 1 is needed!
    print("Assembling Concentrated Loads")
    nlc = 1
    loads = zeros((ndof,nlc),float)
    for p in FE.prop.getProp('n',attr=['cload']):
        cload = zeros(6)
        for i,v in p.cload:
            cload[i] += v
        print(cload)
        print(cload.shape)
        loads[:,0] = fe_util.AssembleVector(loads[:,0],cload,bcon[p.set,:])
    if verbose:
        print("Calpy.Loads")
        print(loads)
    # Perform analysis
    # OK, that is really everything there is to it. Now just run the
    # analysis, and hope for the best ;)
    # Enabling the Echo will print out the data.
    # The result consists of nodal displacements and stress resultants.
    print("Starting the Calpy analysis module --- this might take some time")
    pf.app.processEvents()
    starttime = time.clock()
    displ,frc = beam3d.static(coords,bcon,mats,elements,loads,Echo=True)
    print("Calpy analysis has finished --- Runtime was %s seconds." % (time.clock()-starttime))
    # Export the results, but throw way these for the extra (last) node
    export({'calpy_results':(displ[:-1],frc)})


def postCalpy():
    """Show results from the Calpy analysis."""
    from plugins.postproc import niceNumber,frameScale
    from plugins.postproc_menu import showResults
    try:
        FE = named('fe_model')
        displ,frc = named('calpy_results')
    except:
        warning("I could not find the finite element model and/or the calpy results. Maybe you should try to first create them?")
        raise
        return
    
    # The frc array returns element forces and has shape
    #  (nelems,nforcevalues,nloadcases)
    # nforcevalues = 8 (Nx,Vy,Vz,Mx,My1,Mz1,My2,Mz2)
    # Describe the nforcevalues element results in frc.
    # For each result we give a short and a long description:
    frc_contents = [('Nx','Normal force'),
                    ('Vy','Shear force in local y-direction'),
                    ('Vz','Shear force in local z-direction'),
                    ('Mx','Torsional moment'),
                    ('My','Bending moment around local y-axis'),
                    ('Mz','Bending moment around local z-axis'),
                    ('None','No results'),
                    ]
    # split in two lists
    frc_keys = [ c[0] for c in frc_contents ]
    frc_desc = [ c[1] for c in frc_contents ]

    # Ask the user which results he wants
    res = askItems([('Type of result',None,'select',frc_desc),
                    ('Load case',0),
                    ('Autocalculate deformation scale',True),
                    ('Deformation scale',100.),
                    ('Show undeformed configuration',False),
                    ('Animate results',False),
                    ('Amplitude shape','linear','select',['linear','sine']),
                    ('Animation cycle','updown','select',['up','updown','revert']),
                    ('Number of cycles',5),
                    ('Number of frames',10),
                    ('Animation sleeptime',0.1),
                    ])
    if res:
        frcindex = frc_desc.index(res['Type of result'])
        loadcase = res['Load case']
        autoscale = res['Autocalculate deformation scale']
        dscale = res['Deformation scale']
        showref = res['Show undeformed configuration']
        animate = res['Animate results']
        shape = res['Amplitude shape']
        cycle = res['Animation cycle']
        count = res['Number of cycles']
        nframes = res['Number of frames']
        sleeptime = res['Animation sleeptime']

        dis = displ[:,0:3,loadcase]
        if autoscale:
            siz0 = Coords(FE.nodes).sizes()
            siz1 = Coords(dis).sizes()
            print(siz0)
            print(siz1)
            dscale = niceNumber(1./(siz1/siz0).max())

        if animate:
            dscale = dscale * frameScale(nframes,cycle=cycle,shape=shape) 
        
        # Get the scalar element result values from the frc array.
        val = val1 = txt = None
        if frcindex <= 5:
            val = frc[:,frcindex,loadcase]
            txt = frc_desc[frcindex]
            if frcindex > 3:
                # bending moment values at second node
                val1 = frc[:,frcindex+2,loadcase]

        showResults(FE.nodes,FE.elems,dis,txt,val,showref,dscale,count,sleeptime)



#############################################################################
######### Create a menu with interactive tasks #############

def create_menu():
    """Create the Hesperia menu."""
    MenuData = [
        ("&How To Use",howto),
        ("---",None),
        ("&Create Geometry",createGeometry),
        ("&Assign Properties",assignProperties),
        ("&Export Properties",exportProperties),
        ("&Select Properties",selectProperties),
        ("&Save Properties",saveProperties),
        ("&Read Properties",readProperties),
        ("---",None),
        ("&Create Frame Model",createFrameModel),
        ("&Create Shell Model",createShellModel),
        ("---",None),
        ("&Write Abaqus input file",createAbaqusInput),
        ("&Run Calpy Analysis",runCalpyAnalysis),
        ("&Show Calpy Results",postCalpy),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return menu.Menu('Hesperia',items=MenuData,parent=pf.GUI.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item('Hesperia'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = pf.GUI.menu.item('Hesperia')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

def run():

    # The sole intent of running this script is to create a top level
    # menu 'Hesperia'. The typical action then might be 'show_menu()'.
    # However, during development, you might want to change the menu's
    # actions will pyFormex is running, so a 'reload' action seems
    # more appropriate.
    chdir(__file__)
    clear()
    reload_menu()

if __name__ == 'draw':
    run()
# End

