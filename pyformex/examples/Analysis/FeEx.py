#!/usr/bin/env python pyformex.py
# $Id$
##
## This file is part of pyFormex 0.7.2 Release Tue Sep 23 16:18:43 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##
"""FeEx

level = 'advanced'
topics = ['FEA']
techniques = ['menu', 'dialog', 'persistence', 'colors'] 
"""

from simple import rectangle
from plugins.fe import *
from plugins.properties import *
from plugins.fe_abq import *
from plugins.fe_post import FeResult
from plugins import postproc_menu

myglobals = dict(
    PDB = None,
    parts = None,
    femodels = None,
    model = None,
)

globals().update(myglobals)

def deleteAll():
    global PDB,parts,femodels,model
    PDB = PropertyDB()
    parts = []
    femodels = []
    model = None
    clear()


x0,y0 = 0.,0.
x1,y1 = 1.,1.
nx,ny = 4,4
eltype = 'quad'

def createPart(res=None):
    """Create a rectangular domain from user input"""
    global x0,y0,x1,y1
    if model is not None:
        if ask('You have already merged the parts! I can not add new parts anymore.\nYou should first delete everything and recreate the parts.',['Delete','Cancel']) == 'Delete':
            deleteAll()
        else:
            return
    if res is None:
        res = askItems([('x0',x0),('y0',y0),
                        ('x1',x1),('y1',y1),
                        ('nx',nx),('ny',ny),
                        ('eltype',eltype,'select',['quad','tri-u','tri-d']),
                        ])
    if res:
        globals().update(res)
        if x0 > x1:
            x0,x1 = x1,x0
        if y0 > y1:
            y0,y1 = y1,y0
        diag = {'quad':'', 'tri-u':'u', 'tri-d':'d'}[eltype]
        F = rectangle(nx,ny,x1-x0,y1-y0,diag=diag).trl([x0,y0,0])
        addPart(F)
        drawParts()


def addPart(F):
    """Add a Formex to the parts list."""
    global parts
    n = len(parts)
    F.setProp(n)
    export({'part-%s'%n:F})
    parts.append(F)
    femodels.append(F.feModel())

def drawParts():
    """Draw all parts"""
    clear()
    draw(parts)
    [ drawNumbers(p) for p in parts ]
    [ drawNumbers(Formex(fem[0]),color=red) for fem in femodels ]
    zoomAll()


def createModel():
    """Merge all the parts into a Finite Element model."""
    global model
    model = Model(*mergeModels(femodels))
    drawModel()
    export({'model':model})

def drawModel(offset=0):
    """Draw the merged parts"""
    if model is None:
        warning("You should first merge the parts!")
        return
    clear()
    draw(parts)
    draw(Formex(model.nodes))
    drawNumbers(Formex(model.nodes),color=red,offset=offset)
    [ drawNumbers(p) for p in parts ]
    zoomAll()

def drawCalpy():
    """This should draw node numbers +1"""
    drawModel(offset=1)
    

def getPickedNodes(K):
    """Get the list of picked nodes."""
    # This relies on drawing all parts first, then drawing the nodes
    return getPickedElems(K,len(parts))


def getPickedElems(K,p):
    """Get the list of picked elems from part p."""
    if p in K.keys():
        return K[p]
    return []


################# Functions After merging ######################


section = {
    'name':'thin plate',
    'sectiontype': 'solid',
    'young_modulus': 207000,
    'poisson_ratio': 0.3,
    'thickness': 0.01,
    }


def setMaterial():
    """Set the material"""
    global section
    if model is None:
        warn()
        return
    keys = ['name','sectiontype','young_modulus','poisson_ratio','thickness']
    items = [ (k,section[k]) for k in keys ]
    res = askItems(items)
    if res:
        section.update(res)
        K = pickElements()
        if K:
            for k in range(len(parts)):
                e = getPickedElems(K,k) + model.celems[k]
                print k,e
                PDB.elemProp(set=e,eltype='CPS4',section=ElemSection(section=section))


xcon = True
ycon = True

def warn():
    warning("You should first merge the parts!")

def setBoundary():
    """Pick the points with boundary condition."""
    global PDB,xcon,ycon
    if model is None:
        warn()
        return
    res = askItems([('x-constraint',xcon),('y-constraint',ycon)])
    if res:
        xcon = res['x-constraint']
        ycon = res['y-constraint']
        K = pickPoints()
        if K:
            nodeset = getPickedNodes(K)
            if len(nodeset) > 0:
                print nodeset
                print [xcon,ycon,0,0,0,0]
                PDB.nodeProp(set=nodeset,bound=[xcon,ycon,0,0,0,0])
        
xload = 0.0
yload = 0.0

def setLoad():
    """Pick the points with load condition."""
    global xload,yload
    if model is None:
        warn()
        return
    res = askItems([('x-load',xload),('y-load',yload)])
    if res:
        xload = res['x-load']
        yload = res['y-load']
        K = pickPoints()
        if K:
            nodeset = getPickedNodes(K)
            if len(nodeset) > 0:
                PDB.nodeProp(set=nodeset,cload=[xload,yload,0.,0.,0.,0.])

def printModel():
    print "model:",model

def printDB():
    print PDB.nprop
    print PDB.eprop


############################# Abaqus ##############################
def createAbaqusInput():
    """Write the Abaqus input file."""
    
    # ask job name from user
    res = askItems([('JobName','FeEx')])
    if not res:
        return

    jobname = res['JobName']
    if not jobname:
        print "No Job Name: writing to sys.stdout"
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

    AbqData(model,PDB,[step1,step2],out=out,res=res).write(jobname)
    

##############################################################################

def runCalpyAnalysis(jobname=None,verbose=False):
    """Create data for Calpy analysis module and run Calpy on the data.

    While we could write an analysis file in the Calpy format and then
    run the Calpy program on it (like we did with Abaqus), we can (and do)
    take another road here: Calpy has a Python/numpy interface, allowing
    us to directly present the numerical data in arrays to the analysis
    module.
    It is supposed that the Finite Element model has been created and
    exported under the name 'fe_model'.
    """

    ############################
    # Load the needed calpy modules    
    from plugins import calpy_itf
    calpy_itf.check()
    import calpy
    calpy.options.optimize=True
    from calpy import femodel,fe_util,plane
    ############################

    if model is None:
        warn()
        return

    if jobname is None:
        # ask job name from user
        res = askItems([('JobName','FeEx'),('Verbose Mode',False)])
        if res:
            jobname = res['JobName']
            verbose = res['Verbose Mode']

    if not jobname:
        print "No Job Name: bailing out"
        return
   
    Model = femodel.FeModel(2,"elast","Plane_Stress")
    Model.nnodes = model.nodes.shape[0]
    Model.nelems = model.celems[-1]
    Model.nnodel = 4

    # 2D model in calpy needs 2D coordinates
    coords = model.nodes[:,:2]
    if verbose:
        fe_util.PrintNodes(coords)

    # Boundary conditions
    bcon = zeros((Model.nnodes,2),dtype=int32)
    bcon[:,2:6] = 1 # leave only ux and uy
    for p in PDB.getProp(kind='n',attr=['bound']):
        bnd = where(p.bound)[0]
        if p.set is None:
            nod = arange(Model.nnodes)
        else:
            nod = array(p.set)
        for i in bnd:
            bcon[p.set,i] = 1
    fe_util.NumberEquations(bcon)
    if verbose:
        fe_util.PrintDofs(bcon,header=['ux','uy'])

    # The number of free DOFs remaining
    Model.ndof = bcon.max()
    print "Number of DOF's: %s" % Model.ndof

    
    # We extract the materials/sections from the property database
    matprops = PDB.getProp(kind='e',attr=['section'])
    
    # E, nu, thickness, rho
    mats = array([[mat.young_modulus,
                   mat.poisson_ratio,
                   mat.thickness,
                   0.0,      # rho was not defined in material
                   ] for mat in matprops]) 
    Model.nmats = mats.shape[0]
    if verbose:
        fe_util.PrintMats(mats,header=['E','nu','thick','rho'])

   
    ########### Find number of load cases ############
    Model.nloads = 1
    Model.PrintModelData()
    ngp = 2
    nzem = 3
    Model.banded = True


    # Create element definitions:
    # In calpy, each element is represented by nnod + 1 integer numbers:
    #    matnr,  node1, node2, .... 
    # Also notice that Calpy numbering starts at 1, not at 0 as customary
    # in pyFormex; therefore we add 1 to elems.
    matnr = zeros(Model.nelems,dtype=int32)
    for i,mat in enumerate(matprops):  # proces in same order as above!
        matnr[mat.set] = i+1

    PlaneGrp = []
    
    for i,e in enumerate(model.elems):
        j,k = model.celems[i:i+2]
        Plane = plane.Quad("part-%s" % i,[ngp,ngp],Model)
        Plane.debug = 0
        fe_util.PrintElements(e+1,matnr[j:k])
        Plane.AddElements(e+1,matnr[j:k],mats,coords,bcon)
        PlaneGrp.append(Plane)
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
    Model.nloads = 1
    f = zeros((Model.ndof,Model.nloads),float)
    print f.shape
    for p in PDB.getProp('n',attr=['cload']):
        if p.set is None:
            nodeset = range(Model.nnodes)
        else:
            nodeset = p.set
        for n in nodeset:
            print p.cload[:2]
            print
            f[:,0] = fe_util.AssembleVector(f[:,0],p.cload[:2],bcon[n])
    if verbose:
        print "Calpy.Loads"
        print f
    # Perform analysis
    # OK, that is really everything there is to it. Now just run the
    # analysis, and hope for the best ;)
    # Enabling the Echo will print out the data.
    # The result consists of nodal displacements and stress resultants.
    print "Starting the Calpy analysis module --- this might take some time"
    GD.app.processEvents()
    starttime = time.clock()
    ############ Create global stiffness matrix ##########
    s = Model.ZeroStiffnessMatrix(0)
    for elgrp in PlaneGrp:
        s = elgrp.Assemble(s,mats,Model)
    #print "The complete stiffness matrix"
    #print s
    print f
    v = Model.SolveSystem(s,f)
    print "Displacements",v
    print "Calpy analysis has finished --- Runtime was %s seconds." % (time.clock()-starttime)
    displ = fe_util.selectDisplacements (v,bcon)
    print displ.shape
    print displ
    
    DB = FeResult()
    DB.nodes = model.nodes
    DB.nnodes = model.nodes.shape[0]
    DB.nodid = arange(DB.nnodes)
    DB.elems = dict(enumerate(model.elems))
    DB.nelems = model.celems[-1]
    DB.Finalize()
    print DB.elems
    for lc in range(Model.nloads):
        DB.Increment(lc,0)
        d = zeros((Model.nnodes,3))
        d[:,:2] = displ[:,:,lc]
        DB.R['U'] = d
    postproc_menu.setDB(DB)
    info("You can now use the postproc menu to display results")
    export({'FeResult':DB})
    

def autoRun():
    createPart({'x1':1.})
    createPart({'x1':-1.})
    createModel()
    nodenrs = arange(model.nodes.shape[0])
    PDB.elemProp(eltype='CPS4',section=ElemSection(section=section))
    PDB.nodeProp(set=nodenrs[:5],bound=[1,1,0,0,0,0])
    PDB.nodeProp(set=nodenrs[-5:],cload=[10.,0.,0.,0.,0.,0.])
    runCalpyAnalysis('FeEx',verbose=True)


def importAll():
    globals().update(GD.PF)

def exportAll():
    GD.PF.update(globals())
                 
#############################################################################
######### Create a menu with interactive tasks #############

def create_menu():
    """Create the FeEx menu."""
    MenuData = [
        ("&Delete All",deleteAll),
        ("&Create Part",createPart),
        ("&Show All",drawParts),
        ("---",None),
        ("&Merge Parts into Model",createModel),
        ("&Show Merged Model",drawModel),
        ("&Show Calpy Numbers",drawCalpy),
        ("&Print model",printModel),
        ("---",None),
        ("&Set material properties",setMaterial),
        ("&Set boundary conditions",setBoundary),
        ("&Set loading conditions",setLoad),
        ("&Print property database",printDB),
        ("---",None),
        ("&Create ABaqus input file",createAbaqusInput),
        ("&Run Calpy analysis",runCalpyAnalysis),
        ("---",None),
        ("&Import all",importAll),
        ("&Export all",exportAll),
        ("&Autorun example",autoRun),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return widgets.Menu('FeEx',items=MenuData,parent=GD.gui.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not GD.gui.menu.item('FeEx'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = GD.gui.menu.item('FeEx')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    reload_menu()
    deleteAll()
    smoothwire()
    transparent()
    lights(False)
    
# End

