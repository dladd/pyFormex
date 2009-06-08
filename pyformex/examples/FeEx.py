#!/usr/bin/env python pyformex.py
# $Id$
##
##  This file is part of pyFormex 0.8 Release Mon Jun  8 11:56:55 2009
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Website: http://pyformex.berlios.de/
##  Copyright (C) Benedict Verhegghe (bverheg@users.berlios.de) 
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
##  along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

# global data

parts = None,
femodels = None,
model = None,
PDB = None,


def resetData():
    global parts,femodels,model,PDB
    parts = []
    femodels = []
    model = None
    PDB = None
    
def reset():
    clear()
    smoothwire()
    transparent()
    lights(False)

def deleteAll():
    resetData()
    reset()

######################## parts ####################
    
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


######################## de model ####################

def createModel():
    """Merge all the parts into a Finite Element model."""
    global model,PDB
    model = Model(*mergeModels(femodels))
    PDB = PropertyDB()
    drawModel()


def drawModel(offset=0):
    """Draw the merged parts"""
    if model is None:
        warning("You should first merge the parts!")
        return
    clear()
    draw(parts)
    draw(Formex(model.coords))
    drawNumbers(Formex(model.coords),color=red,offset=offset)
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
        

def printModel():
    print "model:",model


################# Add properties ######################


def warn():
   warning("You should first merge the parts!")


section = {
    'name':'thin plate',
    'sectiontype': 'solid',
    'young_modulus': 207000,
    'poisson_ratio': 0.3,
    'thickness': 1.0,
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

def deleteAllMats():
    PDB.delProp(kind='e',attr=['eltype'])


xcon = True
ycon = True


## def merged():
##     ok = model is not None
##     if not ok:
##         ok = ack("The parts should be merged first!\nShall I do this now for you?")
##         if ok:
            
##     return notok


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

def deleteAllBcons():
    PDB.delProp(kind='n',attr=['bound'])



# Concentrated loads

xload = 0.0
yload = 0.0

def setCLoad():
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
                print "SETTING CLOAD",[xload,yload,0.,0.,0.,0.]
                PDB.nodeProp(set=nodeset,cload=[xload,yload,0.,0.,0.,0.])


def deleteAllCLoads():
    PDB.delProp(kind='n',attr=['cload'])


# Line loads

xlload = 0.0
ylload = 0.0


def setLLoad():
    """Pick the lines with load condition."""
    global xdload,ydload
    if model is None:
        warn()
        return
    res = askItems([('x-load',xdload),('y-load',ydload)])
    if res:
        xdload = res['x-load']
        ydload = res['y-load']
        K = pickLines()
        if K:
            nodeset = getPickedNodes(K)
            if len(nodeset) > 0:
                PDB.nodeProp(set=nodeset,lload=[xlload,ylload,0.])

def deleteAllLLoads():
    PDB.delProp(kind='n',attr=['lload'])


def printDB():
    print "\n*** Node properties:"
    for p in PDB.nprop:
        print p
    print "\n*** Element properties:"
    for p in PDB.eprop:
        print p


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

def runCalpyAnalysis(jobname=None,verbose=False,flavia=False):
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
    from calpy.arrayprint import aprint
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
   
    # OK, start calpy
    print "Starting the Calpy analysis module --- this might take some time"
    GD.app.processEvents()
    starttime = time.clock()

    Model = femodel.FeModel(2,"elast","Plane_Stress")
    Model.nnodes = model.coords.shape[0]
    Model.nelems = model.celems[-1]
    Model.nnodel = 4

    # 2D model in calpy needs 2D coordinates
    coords = model.coords[:,:2]
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
    for p in PDB.getProp('n',attr=['cload']):
        if p.set is None:
            nodeset = range(Model.nnodes)
        else:
            nodeset = p.set
        F = [0.0,0.0]
        for i,v in p.cload:
            if i in [0,1]:
                F[i] = v
        for n in nodeset:
            print F
            print "HALLO"
            f[:,0] = fe_util.AssembleVector(f[:,0],F,bcon[n])

    print "Assembling distributed loads"
    # This is a bit more complex. See Calpy for details
    # We first generate the input data, then read them with the
    # calpy femodel.ReadBoundaryLoads function and finally
    # assemble them with plane.addBoundaryLoads.
    for p in PDB.getProp('n',attr=['dload']):
        print p
    #idloads,dloads = plane.ReadBoundaryLoads(*nb,model.ndim)
    
    if verbose:
        print "Calpy.Loads"
        print f

    ############ Create global stiffness matrix ##########
    s = Model.ZeroStiffnessMatrix(0)
    for elgrp in PlaneGrp:
        s = elgrp.Assemble(s,mats,Model)
    #print "The complete stiffness matrix"
    #print s
    print f
    v = Model.SolveSystem(s,f)
    if verbose:
        print "Displacements",v
    print "Calpy analysis has finished --- Runtime was %s seconds." % (time.clock()-starttime)
    displ = fe_util.selectDisplacements (v,bcon)
    print displ.shape
    if verbose:
        print displ

    if flavia:
        flavia.WriteMeshFile(jobname,"Quadrilateral",model.nnodel,coord,nodes,matnr)
        res=flavia.ResultsFile(jobname)
        
    # compute stresses
    for l in range(Model.nloads):
        
        print "Results for load case %d" %(l+1)
        print "Displacements"
        aprint(displ[:,:,l],header=['x','y'],numbering=True)

        if flavia:
            flavia.WriteResultsHeader(res,'"Displacement" "Elastic Analysis"',l+1,'Vector OnNodes')
            flavia.WriteResults(res,displ[:,:,l])
            
        stresn = count = None
        i = 0
        for e,P in zip(model.elems,PlaneGrp):
            i += 1
            #P.debug = 1
            stresg = P.StressGP (v[:,l],mats)
            if verbose:
                print "elem group %d" % i
                print "GP Stress\n", stresg
            
            strese = P.GP2Nodes(stresg)
            if verbose:
                print "Nodal Element Stress\n", strese

            #print "Nodes",e+1
            stresn,count = P.NodalAcc(e+1,strese,nnod=Model.nnodes,nodata=stresn,nodn=count)
            #print stresn,count
            
        #print stresn.shape
        #print count.shape
        #print "TOTAL",stresn,count
        stresn /= count.reshape(-1,1)
        #print "AVG",stresn
        if verbose:
            print "Averaged Nodal Stress\n"
            aprint(stresn,header=['sxx','syy','sxy'],numbering=True)
                
        if flavia:
            flavia.WriteResultsHeader(res,'"Stress" "Elastic Analysis"',l+1,'Matrix OnNodes')
            flavia.WriteResults(res,stresn)

    
    DB = FeResult()
    DB.nodes = model.coords
    DB.nnodes = model.coords.shape[0]
    DB.nodid = arange(DB.nnodes)
    DB.elems = dict(enumerate(model.elems))
    DB.nelems = model.celems[-1]
    DB.Finalize()
    DB.data_size['S'] = 3
    #print DB.elems
    for lc in range(Model.nloads):
        DB.Increment(lc,0)
        d = zeros((Model.nnodes,3))
        d[:,:2] = displ[:,:,lc]
        DB.R['U'] = d
        DB.R['S'] = stresn
    postproc_menu.setDB(DB)
    info("You can now use the postproc menu to display results")
    export({'FeResult':DB})
    

def autoRun():
    createPart({'x1':1.})
    createPart({'x1':-1.})
    createModel()
    nodenrs = arange(model.coords.shape[0])
    PDB.elemProp(eltype='CPS4',section=ElemSection(section=section))
    PDB.nodeProp(set=nodenrs[:ny+1],bound=[1,1,0,0,0,0])
    PDB.nodeProp(set=nodenrs[-(ny+1):],cload=[10.,0.,0.,0.,0.,0.])
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
        ("&Add material properties",setMaterial),
        ("&Add boundary conditions",setBoundary),
        ("&Add concentrated loads",setCLoad),
        ("&Add line loads",setLLoad),
        ("&Print property database",printDB),
        ("&Delete all material properties",deleteAllMats),
        ("&Delete all boundary conditions",deleteAllBcons),
        ("&Delete all concentrated loads",deleteAllCLoads),
        ("&Delete all line loads",deleteAllLLoads),
        ("---",None),
        ("&Create Abaqus input file",createAbaqusInput),
        ("&Run Calpy analysis",runCalpyAnalysis),
        ("---",None),
        ("&Import all",importAll),
        ("&Export all",exportAll),
        ("&Autorun example",autoRun),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return widgets.Menu('FeEx',items=MenuData,parent=GD.GUI.menu,before='help')

 
def show_menu():
    """Show the menu."""
    if not GD.GUI.menu.item('FeEx'):
        create_menu()

def close_menu():
    """Close the menu."""
    m = GD.GUI.menu.item('FeEx')
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    resetData()
    reset()
    reload_menu()
    
# End

