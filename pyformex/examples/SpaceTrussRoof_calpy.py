#!/usr/bin/env pyformex
# $Id: SpaceTrussRoof_calpy.py 154 2006-11-03 19:08:25Z bverheg $
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
"""Double Layer Flat Space Truss Roof

level = 'advanced'
topics = ['FEA']
techniques = ['dialog', 'animation', 'persistence', 'colors'] 
"""

from plugins.properties import *
############################
# Load the needed calpy modules    
from plugins import calpy_itf
calpy_itf.check()

from calpy.fe_util import *
from calpy.truss3d import *

############################
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

nodes,elems=F.feModel()

###############
#Creating elemsets
###################
# Remember: elems are in the same order as elements in F
topbar = where(F.p==3)[0]
bottombar = where(F.p==0)[0]
diabar = where(F.p==1)[0]

###############
#Creating nodesets
###################

nnod=nodes.shape[0]
nlist=arange(nnod)
count = zeros(nnod)
for n in elems.flat:
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
nelems = elems.shape[0]
mats = array([ [p.young_modulus,p.density,p.cross_section] for p in props])
matnr = zeros_like(F.p)
for i,p in enumerate(props):
    matnr[p.set] = i+1
matnod = concatenate([matnr.reshape((-1,1)),elems+1],axis=-1)
ndof = bcon.max()

# loads
nlc=1
loads=zeros((ndof,nlc),Float)
for n in field:
    loads[:,0] = AssembleVector(loads[:,0],[ 0.0, 0.0, Q ],bcon[n,:])
for n in edge:
    loads[:,0] = AssembleVector(loads[:,0],[ 0.0, 0.0, Q/2 ],bcon[n,:])

message("Performing analysis: this may take some time")
outfilename = os.path.splitext(os.path.basename(GD.scriptName))[0] + '.out'
outfile = file(outfilename,'w')
message("Output is written to file '%s' in %s" % (outfilename,os.getcwd()))
stdout_saved = sys.stdout
sys.stdout = outfile
print "# File created by pyFormex on %s" % time.ctime()
print "# Script name: %s" % GD.scriptName
displ,frc = static(nodes,bcon,mats,matnod,loads,Echo=True)
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

    def showForces():
        # Creating a formex for displaying results is fairly easy
        results = Formex(nodes[elems],range(nelems))
        # Now give the formex some meaningful colors.
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
        draw(results,color=cval)
        drawtext('Normal force in the truss members',150,20,'tr24')
        CL = ColorLegend(CS,100)
        CLA = decors.ColorLegend(CL,10,10,30,200) 
        GD.canvas.addDecoration(CLA)
        GD.canvas.update()


    # Show a deformed plot
    def deformed_plot(dscale=100.):
        """Shows a deformed plot with deformations scaled with a factor scale."""
        dnodes = nodes + dscale * displ[:,:,0]
        deformed = Formex(dnodes[elems],F.p)
        # deformed structure
        FA = draw(deformed,bbox=None,wait=False)
        TA = decors.Text('Deformed geometry (scale %.2f)' % dscale,400,100,'tr24')
        decorate(TA)
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
        dF = Formex(displ[:,:,0][elems])
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
    options = ['Cancel','Member forces','Deformation','Animated deformation']
    functions = [None,showForces,showDeformation,showAnimatedDeformation]
    while True:
        ans = ask("Which results do you want to see?",options)
        if ans == '':   #timeout
            break
        ind = options.index(ans)
        if ind <= 0:
            break
        functions[ind]()
        
# End
