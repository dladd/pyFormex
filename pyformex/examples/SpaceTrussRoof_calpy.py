#!/usr/bin/env pyformex
# $Id: SpaceTrussRoof_calpy.py 154 2006-11-03 19:08:25Z bverheg $
##
##  This file is part of pyFormex 0.8.2 Release Sat Jun  5 10:49:53 2010
##  pyFormex is a tool for generating, manipulating and transforming 3D
##  geometrical models by sequences of mathematical operations.
##  Homepage: http://pyformex.org   (http://pyformex.berlios.de)
##  Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
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
"""Double Layer Flat Space Truss Roof

level = 'advanced'
topics = ['FEA']
techniques = ['dialog', 'animation', 'persistence', 'colors'] 
"""

from plugins.mesh import Mesh
from plugins.properties import *
############################
# Load the needed calpy modules    
from plugins import calpy_itf
calpy_itf.check()
import calpy
calpy.options.optimize = False
from calpy.fe_util import *
from calpy.truss3d import *

############################

if not checkWorkdir():
    exit()

import time

####
#Data
###################

dx = 1800  # Modular size [mm]
ht = 1500  # Deck height [mm]
nx = 8     # number of bottom deck modules in x direction 
ny = 6     # number of bottom deck modules in y direction 

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

mesh = F.toMesh()

###############
#Creating elemsets
###################
# Remember: elements in mesh are in the same order as elements in F
topbar = where(F.prop==3)[0]
bottombar = where(F.prop==0)[0]
diabar = where(F.prop==1)[0]

###############
#Creating nodesets
###################

nnod = mesh.nnodes()
nlist = arange(nnod)
count = zeros(nnod)
for n in mesh.elems.flat:
    count[n] += 1
field = nlist[count==8]
topedge = nlist[count==7]
topcorner = nlist[count==6]
bottomedge = nlist[count==5]
bottomcorner = nlist[count==3]
edge =  concatenate([topedge,topcorner])
support = concatenate([bottomedge,bottomcorner])

########################
#Defining and assigning the properties
#############################

Q = 0.5*q*dx*dx

P = PropertyDB()
P.nodeProp(field,cload = [0,0,Q,0,0,0])
P.nodeProp(edge,cload = [0,0,Q/2,0,0,0])
P.nodeProp(support,bound = [1,1,1,0,0,0])

circ20 = ElemSection(section={'name':'circ20','sectiontype':'Circ','radius':10, 'cross_section':314.159}, material={'name':'S500', 'young_modulus':210000, 'shear_modulus':81000, 'poisson_ratio':0.3, 'yield_stress' : 500,'density':0.000007850})

props = [ \
     P.elemProp(topbar,section=circ20,eltype='T3D2'), \
     P.elemProp(bottombar,section=circ20,eltype='T3D2'), \
     P.elemProp(diabar,section=circ20,eltype='T3D2'), \
     ]

# Since all elems have same characteristics, we could just have used:
#   P.elemProp(section=circ20,elemtype='T3D2')
# But putting the elems in three sets allows for separate postprocessing 


######### 
#calpy analysis
###################

# boundary conditions
bcon = zeros([nnod,3],dtype=int)
bcon[support] = [ 1,1,1 ]
NumberEquations(bcon)

#materials
mats = array([ [p.young_modulus,p.density,p.cross_section] for p in props])
matnr = zeros_like(F.prop)
for i,p in enumerate(props):
    matnr[p.set] = i+1
matnod = concatenate([matnr.reshape((-1,1)),mesh.elems+1],axis=-1)
ndof = bcon.max()

# loads
nlc=1
loads=zeros((ndof,nlc),Float)
for n in field:
    loads[:,0] = AssembleVector(loads[:,0],[ 0.0, 0.0, Q ],bcon[n,:])
for n in edge:
    loads[:,0] = AssembleVector(loads[:,0],[ 0.0, 0.0, Q/2 ],bcon[n,:])

message("Performing analysis: this may take some time")

# Find a candidate for the output file
fullname = os.path.splitext(__file__)[0] + '.out'
basename = os.path.basename(fullname)
dirname = os.path.dirname(fullname)
outfilename = None
for candidate in [dirname,GD.cfg['workdir'],'/var/tmp']:
    if isWritable(candidate):
        fullname = os.path.join(candidate,basename)
        if not os.path.exists(fullname) or isWritable(fullname):
            outfilename = fullname
            break

if outfilename is None:
    error("No writeable path: I can not execute the simulation.\nCopy the script to a writeable path and try running from there.")
    exit()

outfile = file(outfilename,'w')
message("Output is written to file '%s'" % os.path.realpath(outfilename))
stdout_saved = sys.stdout
sys.stdout = outfile
print "# File created by pyFormex on %s" % time.ctime()
print "# Script name: %s" % GD.scriptName
displ,frc = static(mesh.coords,bcon,mats,matnod,loads,Echo=True)
print "# Analysis finished on %s" % time.ctime()
sys.stdout = stdout_saved
outfile.close()

################################
#Using pyFormex as postprocessor
################################

if GD.options.gui:

    from plugins.postproc import niceNumber,frameScale
    from gui.colorscale import *
    import gui.decors


    def showOutput():
        #showText(file(outfilename).read())
        showFile(outfilename)
        

    def showForces():
        # Give the mesh some meaningful colors.
        # The frc array returns element forces and has shape
        #  (nelems,nforcevalues,nloadcases)
        # In this case there is only one resultant force per element (the
        # normal force), and only load case; we still need to select the
        # scalar element result values from the array into a onedimensional
        # vector val. 
        val = frc[:,0,0]
        # create a colorscale
        CS = ColorScale([blue,yellow,red],val.min(),val.max(),0.,2.,2.)
        cval = array(map(CS.color,val))
        #aprint(cval,header=['Red','Green','Blue'])
        clear()
        linewidth(3)
        draw(mesh,color=cval)
        drawtext('Normal force in the truss members',300,50,size=24)
        CL = ColorLegend(CS,100)
        CLA = decors.ColorLegend(CL,10,10,30,200,size=24)
        decorate(CLA)


    # Show a deformed plot
    def deformed_plot(dscale=100.):
        """Shows a deformed plot with deformations scaled with a factor scale."""
        # deformed structure
        dnodes = mesh.coords + dscale * displ[:,:,0]
        deformed = Mesh(dnodes,mesh.elems,mesh.prop)
        FA = draw(deformed,bbox='last',view=None,wait=False)
        TA = drawtext('Deformed geometry (scale %.2f)' % dscale,300,50,size=24)
        return FA,TA

    def animate_deformed_plot(amplitude,sleeptime=1,count=1):
        """Shows an animation of the deformation plot using nframes."""
        FA = TA = None
        clear()
        while count > 0:
            count -= 1
            for s in amplitude:
                F,T = deformed_plot(s)
                if FA:
                    GD.canvas.removeActor(FA)
                if TA:
                    GD.canvas.removeDecoration(TA)
                TA,FA = T,F
                sleep(sleeptime)

    def getOptimscale():
        """Determine an optimal scale for displaying the deformation"""
        siz0 = F.sizes()
        dF = Formex(displ[:,:,0][mesh.elems])
        #clear(); draw(dF,color=black)
        siz1 = dF.sizes()
        return niceNumber(1./(siz1/siz0).max())


    def showDeformation():
        clear()
        linewidth(1)
        draw(F,color=black)
        linewidth(3)
        deformed_plot(optimscale)
        view('last',True)


    def showAnimatedDeformation():

        # Show animated deformation
        scale = optimscale
        nframes = 10
        form = 'revert'
        duration = 5./nframes
        ncycles = 2
        items = [ ['scale',scale], ['nframes',nframes],
                  ['form',form],
                  ['duration',duration], ['ncycles',ncycles] ]
        res = askItems(items,'Animation Parameters')
        if res:
            scale = float(res['scale'])
            nframes = int(res['nframes'])
            duration = float(res['duration'])
            ncycles = int(res['ncycles'])
            form = res['form']
            if form in [ 'up', 'updown', 'revert' ]:
                amp = scale * frameScale(nframes,form)
                animate_deformed_plot(amp,duration,ncycles)


    optimscale = getOptimscale()
    options = ['None','Output File','Member forces','Deformation','Animated deformation']
    functions = [None,showOutput,showForces,showDeformation,showAnimatedDeformation]
    while True:
        ans = ask("Which results do you want to see?",options)
        ind = options.index(ans)
        if ind <= 0:
            break
        functions[ind]()
        if widgets.input_timeout > 0:  #timeout
            break
        
# End
