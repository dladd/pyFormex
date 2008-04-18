#!/usr/bin/env python pyformex.py
# $Id$
##  Hesperia Dome
##  (C) Benedict Verhegghe
##
##  All physical quantities are N,mm
##
import simple,utils
from plugins.surface import TriSurface, compactElems
from plugins.properties import *
from plugins.fe_abq import *
from gui.colorscale import ColorScale,ColorLegend
import gui.decors

import time

############################
# Load the needed calpy modules    
from plugins import calpy_itf
calpy_itf.check()
import calpy
calpy.options.optimize=True
from calpy.fe_util import *
from calpy.beam3d import *
############################


filename = GD.cfg['curfile']
dirname,basename = os.path.split(filename)
project = os.path.splitext(basename)[0]
formexfile = '%s.formex' % project
os.chdir(dirname)

smooth()
lights(False)


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
    n = 6

    # Add a few extra rows to close the gap after projection
    nplus = n+3

    clear()
    # Start with an equilateral triangle in the x-y-plane
    A = simple.triangle()
    A.setProp(1)
    draw(A)

    # Modular size
    a,b,c = A.sizes()
    GD.message("Cell width: %s; height: %s" % (a,b))

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
    draw(Formex([P]),bbox=None)
    F = F.translate(-P)
    draw(F)

    # Now rotate around the x axis over an angle so that the projection on the
    # x-y plane is an isosceles triangle with top angle = 360/5 = 72 degrees.
    # The base angles thus are (180-72)/2 = 54 degrees.

    # Ratio of the height of the isosceles triangle over the icosaeder edge length.
    c = 0.5*tand(54.)
    angle = arccos(tand(54.)/sqrt(3.))
    GD.message("Rotation Ratio: %s; Angle: %s, %s" % (c,angle,angle/rad)) 
    F = F.rotate(angle/rad,0)
    clear()
    draw(F,colormap=['black','magenta','yellow','black'])

    # Project it on the circumscribing sphere
    # The sphere has radius ru
    golden_ratio = 0.5 * (1. + sqrt(5.))
    ru = 0.5 * a * sqrt(golden_ratio * sqrt(5.))
    GD.message("Radius of circumscribed sphere: %s" % ru)

    ru *= n
    C = [0.,0.,-ru]
    F = F.projectOnSphere(ru,center=C)
    draw(F)

    hx,hy,h = F.sizes()
    GD.message("Height of the dome: %s" % h)

    # The base circle goes through bottom corner of n-th row,
    # which will be the first point of the first triangle of the n-th row.
    # Draw the point to check it.
    
    i = (n-1)*n/2
    P = F[i][0]
    draw(Formex([P]),marksize=10,bbox=None)

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
    F = F.cutAtPlane(P,N,newprops=newprops,side='+',atol=0.0001)
    clear()
    draw(F)

    # Finally, create a rosette to make the circle complete
    # and rotate 90 degrees to orient it like in the paper
    clear()
    F = F.rosette(5,72.).rotate(90)

    def cutOut(F,c,r):
        """Remove all elements of F contained in a sphere (c,r)"""
        d = distanceFromPoint(F.f,c)
        return F.select((d < r).any(axis=-1) == False)

    # Cut out the door: remove all members having a point less than
    # edge-length a away from the base point
    p1 = [rb,0.,0.]
    F = cutOut(F,p1,1.1*a*n/6)      # a was a good size with n = 6

    # Scale to the real geometry
    scale = 7000. / F.sizes()[2]
    GD.message("Applying scale factor %s " % scale)
    print F.bbox()
    F = F.scale(scale)
    print F.bbox()

    clear()
    draw(F,alpha=0.4)
    export({'F':F})



## def saveGeometry(fn=None):
##     if fn is None:
##         fn = formexfile
##     print "Saving %s triangles" % F.nelems()
##     fil = file(fn,'w')
##     F.write(fil)
##     fil.close()


## def readGeometry(fn=None):
##     global F
##     if fn is None:
##         fn = formexfile
##     fil = file(fn,'r')
##     F = Formex.read(fil)
##     fil.close()
##     print "Read %s triangles" % F.nelems()
##     print "BBOX: %s" % F.bbox()
##     draw(F)


## def showTransparent():
##     clear()
##     draw(F,color='lightgrey',alpha=0.5)
##     draw(D)
##     smoothwire()
##     transparency(True)


## def prepareProperties():
##     global FA
##     #Create a view for setting props
##     createView('myview1',(0.,0.,90.))
##     clear()
##     FA = draw(F,view='myview1')
##     drawNumbers(F)

    
## p=0
## def setProperty():
##     global p
##     p = 0
##     res = askItems([['Property',p]])
##     if res:
##         p = res['Property']
   

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
        if sel.has_key(0):
            GD.debug("PICKED NUMBERS:%s" % sel)
            F.p[sel[0]] = p
        undraw(FA)
        FA = draw(F,view=None,bbox=None)


def exportProperties():
    """Save the current properties under a name"""
    res = askItems([('Property Name','p')])
    if res:
        p = res['Property Name']
        if not p.startswith('prop:'):
            p = "prop:%s" % p
        export({p:F.p})


def selectProperties():
    """Select one of the saved properties"""
    res = askItems([('Property Name','p')])
    if res:
        p = res['Property Name']
        if GD.PF.has_key(p):
            F.setProp(GD.PF[p])


def saveProperties(fn = None):
    """Save the current properties."""
    if not fn:
        fn = askFilename(dirname,filter="Property files (*.prop)")
    if fn:
        F.p.tofile(fn,sep=',')

        
def readProperties(fn = None):
    """Read properties from file."""
    if not fn:
        fn = askFilename(dirname,filter="Property files (*.prop)",exist=True)
    if fn:
        p = fromfile(fn,sep=',')
        F.setProp(p)
        clear()
        draw(F)
    

def readOrCreate():
    """The full action."""
    if os.path.exists(formexfile):
        readGeometry()
        draw(F)
    else:
        createGeometry()
        saveGeometry()
    


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

    It is supposed the the Geometry has been created and is available
    as a global variable F.
    """
    
    # Turn the Formex structure into a TriSurface
    # This guarantees that element i of the Formex is element i of the TriSurface
    S = TriSurface(F)
    nodes = S.coords
    elems = S.elems  # the triangles

    # Create edges and faces from edges
    print "The structure has %s nodes, %s edges and %s faces" % (S.ncoords(),S.nedges(),S.nfaces())

    # Create the steel structure
    E = Formex(nodes[S.edges])
    wireframe()
    clear()
    draw(E)
    
    # Get the tri elements that are part of a quadrilateral:
    prop = F.p
    quadtri = S.faces[prop==6]
    nquadtri = quadtri.shape[0]
    print "%s triangles are part of quadrilateral faces" % nquadtri
    if nquadtri > 0:
        # Create triangle definitions of the quadtri faces
        tri = compactElems(S.edges,quadtri)
        D = Formex(nodes[tri])
        clear()
        flatwire()
        draw(D,color='yellow')

    conn = connections(quadtri)
    print conn

    # Filter out the single connection edges
    internal = [ c[0] for c in conn if len(c[1]) > 1 ]
    print "Internal edges in quadrilaterals: %s" % internal
    
    E = Formex(nodes[S.edges],1)
    E.p[internal] = 6
    wireframe()
    clear()
    draw(E)

    # Remove internal edges
    tubes = S.edges[E.p != 6]

    print "Number of tube elements after removing %s internals: %s" % (len(internal),tubes.shape[0])

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
    print b*h**3
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
        'torsional_rigidity': J
        }
    steel = {
        'name':'steel',
        'young_modulus' : 206000,
        'shear_modulus' : 81500,
        'density' : 7.85e-9,
        }
    print tube
    print steel
    tubesection = ElemSection(section=tube,material=steel)
    beam0 = ElemProperty(1, tubesection, elemtype='FRAME3D')    

    # Calculate the nodal loads

    # Area of triangles
    area,normals = S.areaNormals()
    print "Area:\n%s" % area
    # compute bar lengths
    bars = nodes[tubes]
    barV = bars[:,1,:] - bars[:,0,:]
    barL = sqrt((barV*barV).sum(axis=-1))
    print "Member length:\n%s" % barL


    ### DEFINE LOAD CASE (ask user) ###
    res = askItems([('JobName','hesperia_frame'),('Steel',True),('Glass',True),('Snow',False)])

    jobname = res['JobName']
    if not jobname:
        print "No Job Name: bailing out"
        return

    NODLoad = zeros((S.ncoords(),3))
    
    if res['Steel']:
        # add the STEEL weight
        lwgt = steel['density'] * tube['cross_section'] * 9810  # mm/s**2
        print "Weight per length %s" % lwgt
        # assemble steel weight load
        for e,L in zip(tubes,barL):
            NODLoad[e] += [ 0., 0., - L * lwgt / 2 ]

    if res['Glass']:
        # add the GLASS weight
        wgt = 450e-6 # N/mm**2
        # assemble uniform glass load
        for e,a in zip(S.elems,area):
            NODLoad[e] += [ 0., 0., - a * wgt / 3 ]

    if res['Snow']:
        # add NON UNIFORM SNOW
        fn = 'hesperia-nieve.prop'
        snowp = fromfile(fn,sep=',')
        snow_uniform = 320e-6 # N/mm**2
        snow_non_uniform = { 1:333e-6, 2:133e-6, 3:133e-6, 4:266e-6, 5:266e-6, 6:667e-6 }

        # assemble non-uniform snow load
        for e,a,p in zip(S.elems,area,snowp):
            NODLoad[e] += [ 0., 0., - a * snow_non_uniform[p] / 3]

    # Put the nodal loads in the properties database
    print NODLoad[e]
    for i,P in enumerate(NODLoad):
        NodeProperty(i,cload=[P[0],P[1],P[2],0.,0.,0.])

    # Get support nodes
    botnodes = where(isClose(nodes[:,2], 0.0))[0]
    bot = nodes[botnodes]
    GD.message("There are %s support nodes." % bot.shape[0])

    # Upper structure
    nnod0 = nodes.shape[0]              # node number offset
    ntub0 = tubes.shape[0]              # element number offset
    tubeprops = ones((ntub0),dtype=Int) # all elements have same props
    
    # Create support systems (vertical beams)
    bot2 = bot + [ 0.,0.,-200.]         # new nodes 200mm below bot
    botnodes2 = arange(botnodes.shape[0]) + nnod0  # node numbers
    nodes = concatenate([nodes,bot2])
    supports = column_stack([botnodes,botnodes2])
    supportprops = 2 * ones((supports.shape[0]),dtype=Int)
    tubes = concatenate([tubes,supports]) 
    tubeprops = concatenate([tubeprops,supportprops])
    ## !!!
    ## THIS SHOULD BE FIXED !!!
    ElemProperty(2, ElemSection(material=steel,section={ 
        'name':'support',
        'cross_section': A,
        'moment_inertia_11': I1,
        'moment_inertia_22': I2,
        'moment_inertia_12': I12,
        'torsional_rigidity': J
        }),
                 elemtype='FRAME3D')                         

    # Finally, the botnodes2 get the support conditions
    botnodes = botnodes2
    
    # Draw the supports
    S = connect([Formex(bot),Formex(bot2)])
    draw(S,color='black')

    nodeprops = arange(nodes.shape[0])   # each node has own property 
    export({'fe_model':(nodes,tubes,nodeprops,tubeprops,botnodes,jobname)})

##     # Radial movement only
##     np_fixed = NodeProperty(1,bound=[0,1,1,0,0,0],coords='cylindrical',coordset=[0,0,0,0,0,1])
    
##     # No movement, since we left out the ring beam
##     for i in botnodes:
##         NodeProperty(i,bound=[1,1,1,0,0,0],coords='cylindrical',coordset=[0,0,0,0,0,1])

##     np_central_loaded = NodeProperty(3, displacement=[[1,radial_displacement]],coords='cylindrical',coordset=[0,0,0,0,0,1])
##     #np_transf = NodeProperty(0,coords='cylindrical',coordset=[0,0,0,0,0,1])


def createShellModel():
    """Create the Finite Element Model.

    It is supposed here that the Geometry has been created and is available
    as a global variable F.
    """
    
    # Turn the Formex structure into a TriSurface
    # This guarantees that element i of the Formex is element i of the TriSurface
    S = TriSurface(F)
    print "The structure has %s nodes, %s edges and %s faces" % (S.ncoords(),S.nedges(),S.nfaces())
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
    print glass_plate
    print glass
    glasssection = ElemSection(section=glass_plate,material=glass)

    PDB = PropertyDB()
    # All elements have same property:
    PDB.elemProp(eset=arange(len(elems)),section=glasssection,eltype='STRI3')    

    # Calculate the nodal loads

    # Area of triangles
    area,normals = S.areaNormals()
    print "Area:\n%s" % area

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
            PDB.nodeProp(tag=step,nset=i,cload=[P[0],P[1],P[2],0.,0.,0.])

    if res['Snow']:
        step += 1
        NODLoad = zeros((S.ncoords(),3))
        # add NON UNIFORM SNOW
        fn = 'hesperia2-nieve.prop'
        snowp = fromfile(fn,sep=',')
        snow_uniform = 320e-6 # N/mm**2
        snow_non_uniform = { 1:333e-6, 2:133e-6, 3:133e-6, 4:266e-6, 5:266e-6, 6:667e-6 }
        # assemble non-uniform snow load
        for e,a,p in zip(S.elems,area,snowp):
            NODLoad[e] += [ 0., 0., - a * snow_non_uniform[p] / 3]
        # Put the nodal loads in the properties database
        for i,P in enumerate(NODLoad):
            PDB.nodeProp(tag=step,nset=[i],cload=[P[0],P[1],P[2],0.,0.,0.])

    # Get support nodes
    botnodes = where(isClose(nodes[:,2], 0.0))[0]
    bot = nodes[botnodes].reshape((-1,1,3))
    GD.message("There are %s support nodes." % bot.shape[0])

    botofs = bot + [ 0.,0.,-0.2]
    bbot2 = concatenate([bot,botofs],axis=1)
    print bbot2.shape
    S = Formex(bbot2)
    draw(S)
    
##     np_central_loaded = NodeProperty(3, displacement=[[1,radial_displacement]],coords='cylindrical',coordset=[0,0,0,0,0,1])
##     #np_transf = NodeProperty(0,coords='cylindrical',coordset=[0,0,0,0,0,1])

##     # Radial movement only
##     np_fixed = NodeProperty(1,bound=[0,1,1,0,0,0],coords='cylindrical',coordset=[0,0,0,0,0,1])
    
    # Since we left out the ring beam, we enforce no movement at the botnodes
    bc = PDB.nodeProp(nset=botnodes,bound=[1,1,1,0,0,0],csys=CoordSystem('C',[0,0,0,0,0,1]))

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
    try:
        FE = named('fe_model')
        nodes = FE.nodes
        elems = FE.elems
        prop = FE.prop
        nsteps = FE.nsteps
    except:
        warning("I could not find the finite element model.\nMaybe you should first try to create it?")
        return
    
    # ask job name from user
    res = askItems([('JobName','hesperia_shell')])
    if not res:
        return

    jobname = res['JobName']
    if not jobname:
        print "No Job Name: writinge to sys.stdout"
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
    import sys
    try:
        nodes,tubes,nodeprops,tubeprops,botnodes,jobname = named('fe_model')
    except:
        warning("I could not find the finite element model.\nMaybe you should first try to create it?")
        #raise
        return
    
    nnod = nodes.shape[0]
    nel = tubes.shape[0]
    print "Number of nodes: %s" % nnod
    print "Number of elements: %s" % nel

    # Create an extra node for beam orientations
    #
    # !!! This is ok for the support beams, but for the structural beams
    # !!! this should be changed to the center of the sphere !!!
    extra_node = array([[0.0,0.0,0.0]])
    coords = concatenate([nodes,extra_node])
    nnod = coords.shape[0]
    print "Adding a node for orientation: %s" % nnod

    # Create element definitions:
    # In calpy, each beam element is represented by 4 integer numbers:
    #    i j k matnr,
    # where i,j are the node numbers,
    # k is an extra node for specifying orientation of beam (around its axis),
    # matnr refers to the material/section properties.
    #
    # Also notice that Calpy numbering starts at 1, not at 0 as customary
    # in pyFormex; therefore we add 1 to tubes.
    # The third node for all beams is the last (extra) node, numbered nnod.
    # We need to reshape tubeprops to allow concatenation
    # Thus we get:

    print tubes.shape
    print tubeprops.shape
    elements = concatenate([tubes + 1,
                            nnod * ones(shape=(nel,1),dtype=int),
                            tubeprops.reshape((-1,1))],
                           axis=1)
    
    #print elements

    # Beam Properties in Calpy consist of 7 values:
    #   E, G, rho, A, Izz, Iyy, J
    # The beam y-axis lies in the plane of the 3 nodes i,j,k.
    # We initialize the materials to all zeros and then fill in the
    # values from the properties database created under createFeModel()
    # (this database is a global variable of the properties module, and
    #  is thus imported at the top of this file)
    matnrs = unique(tubeprops) # the used beam prop numbers 
    nmats = matnrs.shape[0]
    mats = zeros((nmats,7),dtype=float)
    print mats.shape
    for matnr in matnrs:
        print "Material %s" % matnr
        mat = the_elemproperties[matnr]
        print mat
        mats[matnr-1] = [ mat.young_modulus,
                          mat.shear_modulus,
                          mat.density,
                          mat.cross_section,
                          mat.moment_inertia_11,
                          mat.moment_inertia_22,
                          mat.moment_inertia_12,
                          ]
    #print mats

    #print the_nodeproperties
    
    # Boundary conditions
    # While we could get the boundary conditions from the node properties
    # database, we will formulate them directly from the numbers
    # of the supported nodes (botnodes).
    # Calpy (currently) only accepts boundary conditions in global
    # (cartesian) coordinates. However, as we only use fully fixed
    # (though hinged) support nodes, the solution is simple here.
    # For each supported node, a list of 6 codes can (should)be given,
    # corresponding to the six Degrees Of Freedom (DOFs): ux,uy,uz,rx,ry,rz.
    # The code has value 1 if the DOF is fixed (=0.0) and 0 if it is free.
    # The easiest way to set the correct boundary conditions array for Calpy
    # is to put these codes in a text field and have them read with
    # ReadBoundary.
    s = ""
    for n in botnodes + 1:   # NOTICE THE +1 to comply with Calpy numbering!
        s += "  %d  1  1  1  1  1  1\n" % n    # a fixed hinge
    # Also clamp the fake extra node
    s += "  %d  1  1  1  1  1  1\n" % nnod
    print "Specified boundary conditions"
    print s
    bcon = ReadBoundary(nnod,6,s)
    NumberEquations(bcon)
    #print bcon # now all DOFs are numbered from 1 to ndof

    # The number of free DOFs remaining
    ndof = bcon.max()

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
    print "Assembling Concentrated Loads"
    nlc = 1
    loads = zeros((ndof,nlc),float)
    for n,p in the_nodeproperties.items():
        cload = p.cload
        if cload is not None:
            #print "Node %s: %s" % (n,cload)
            loads[:,0] = AssembleVector(loads[:,0],cload,bcon[n,:])

    # Perform analysis
    # OK, that is really everything there is to it. Now just run the
    # analysis, and hope for the best ;)
    # Enabling the Echo will print out the data.
    # The result consists of nodal displacements and stress resultants.
    print "Starting the Calpy analysis module --- this might take some time"
    GD.app.processEvents()
    starttime = time.clock()
    displ,frc = static(coords,bcon,mats,elements,loads,Echo=True)
    print "Calpy analysis has finished --- Runtime was %s seconds." % (time.clock()-starttime)
    # Export the results, but throw way these for the extra (last) node
    export({'calpy_results':(displ[:-1],frc)})


def niceNumber(f,approx=floor):
    """Returns a nice number close to but not smaller than f."""
    n = int(approx(log10(f)))
    m = int(str(f)[0])
    return m*10**n


def frameScale(nframes=10,cycle='up',shape='linear'):
    """Return a sequence of scale values between -1 and +1.

    nframes is the number of steps between 0 and |1| values.

    cycle determines how subsequent cycles occur:
      'up' : ramping up
      'updown': ramping up and down
      'revert': ramping up and down then reverse up and down

    shape determines the shape of the amplitude curve:
      'linear': linear scaling
      'sine': sinusoidal scaling
    """
    s = arange(nframes+1)
    if cycle in [ 'updown', 'revert' ]:
        s = concatenate([s, fliplr(s[:-1].reshape((1,-1)))[0]])
    if cycle in [ 'revert' ]: 
        s = concatenate([s, -fliplr(s[:-1].reshape((1,-1)))[0]])
    return s.astype(float)/nframes


def showResults(nodes,elems,displ,text,val,val1=None,showref=False,dscale=100.,
                count=1,sleeptime=-1.):
    """Display a constant or linear scalar field on 1-dim elements.

    If dscale is a list of values, the results will be drawn with
    subsequent deformation scales, with a sleeptime intermission,
    and the whole cycle will be repeated count times.
    """
    clear()

    #bbox = ref.bbox()

    # draw undeformed structure
    if showref:
        ref = Formex(nodes[elems])
        draw(ref,bbox=None,color='green',linewidth=1)

    # compute the colors according to the values
    if val is None:
        # only display deformed geometry
        val = 'blue'
        val1 = None
    else:
        # create a colorscale and draw the colorlegend
        CS = ColorScale([blue,yellow,red],val.min(),val.max(),0.,2.,2.)
        cval = array(map(CS.color,val))
        cval1 = None
        if val1 is not None:
            cval1 = array(map(CS.color,val1))
            cval = column_stack([cval,cval1])
            print cval.shape
        CL = ColorLegend(CS,100)
        CLA = decors.ColorLegend(CL,10,10,30,200) 
        GD.canvas.addDecoration(CLA)

    # the supplied text
    if text:
        drawtext(text,150,30,'tr24')

    # create the frames while displaying them
    dscale = array(dscale)
    frames = []   # a place to store the drawn frames
    for dsc in dscale.flat:

        dnodes = nodes + dsc * displ
        deformed = Formex(dnodes[elems])

        # We store the changing parts of the display, so that we can
        # easily remove/redisplay them
        F = draw(deformed,color=cval,linewidth=3,view='__last__',wait=None)
        T = drawtext('Deformation scale = %s' % dsc,150,10,'tr18')

        # remove the last frame
        # This is a clever trick: we remove the old drawings only after
        # displaying new ones. This makes the animation a lot smoother
        # (though the code is less clear and compact).
        if len(frames) > 0:
            GD.canvas.removeActor(frames[-1][0])
            GD.canvas.removeDecoration(frames[-1][1])
        # add the latest frame to the stored list of frames
        frames.append((F,T))
        if sleeptime > 0.:
            sleep(sleeptime)

    # display the remaining cycles
    count -= 1
    FA,TA = frames[-1]
    #print frames
    #print count
    while count > 0:
        count -= 1

        for F,T in frames:
            #print count,F,T
            GD.canvas.addActor(F)
            GD.canvas.addDecoration(T)
            GD.canvas.removeActor(FA)
            GD.canvas.removeDecoration(TA)
            GD.canvas.display()
            GD.canvas.update()
            FA,TA = F,T
            if sleeptime > 0.:
                sleep(sleeptime)


def postCalpy():
    """Show results from the Calpy analysis."""
    try:
        nodes,tubes,nodeprops,tubeprops,botnodes,jobname = named('fe_model')
        print "OK"
        displ,frc = named('calpy_results')
        print "OK2"
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
            siz0 = Coords(nodes).sizes()
            siz1 = Coords(dis).sizes()
            print siz0
            print siz1
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

        showResults(nodes,tubes,dis,txt,val,val1,showref,dscale,count,sleeptime)


#############################################################################
######### Create a menu with interactive tasks #############

def create_menu():
    """Create the Hesperia menu."""
    MenuData = [
        ("&How To Use",howto),
        ("---",None),
        ("&Create Geometry",createGeometry),
##         ("&Save Geometry",saveGeometry),
##         ("&Read Geometry",readGeometry),
##         ("&Show Transparent",showTransparent),
##         ("&Prepare Properties",prepareProperties),
##         ("&Set Property",setProperty),
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
    return widgets.Menu('Hesperia',items=MenuData,parent=GD.gui.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not GD.gui.menu.item('Hesperia'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = GD.gui.menu.item('Hesperia')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    # The sole intent of running this script is to create a top level
    # menu 'Hesperia'. The typical action then might be 'show_menu()'.
    # However, during development, you might want to change the menu's
    # actions will pyFormex is running, so a 'reload' action seems
    # more appropriate.
    
    reload_menu()

# End

