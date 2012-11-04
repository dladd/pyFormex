# $Id$
##
##  This file is part of pyFormex 0.8.8  (Sun Nov  4 17:22:49 CET 2012)
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

"""Vascular Sweeping Mesher

"""
from __future__ import print_function

##import os
from numpy import *
#from plugins import inertia
from plugins.trisurface import *
#import os,sys,commands
from formex import *
from coords import *
from simple import *
from connectivity import *
from plugins.curve import *
from plugins.isopar import *
from elements import *

from plugins.fe import *
from mesh import *
from project import *
from geomtools import triangleCircumCircle


def structuredQuadMeshGrid(sgx=3, sgy=3, isopquad=None):
    """it returns nodes (2D) and elems of a structured quadrilateral grid. nodes and elements are both ordered first vertically (y) and then orizontally (x). This function is the equivalent of simple.rectangularGrid but on the mesh level."""
    sg=regularGrid([0., 0.],[1., 1.],[sgx, sgy])
    sgc=sgy+1
    esg=array([array([[i,i+sgc, i+1+sgc,  i+1] for i in range(sgc-1)])+sgc*j for j in range(sgx)])
    if isopquad=='quad16': return sg.reshape(-1, 2), esg.reshape(-1, 4), regularGrid([0., 0.], [1., 1.], [3, 3]).reshape(-1, 2)#control points for the hex64 applied to a basic struct hex grid
    else:return sg.reshape(-1, 2), esg.reshape(-1, 4)

def structuredHexMeshGrid(dx, dy, dz, isophex='hex64'):
    """it builds a structured hexahedral grid with nodes and elements both numbered in a structured way: first along z, then along y,and then along x. The resulting hex cells are oriented along z. This function is the equivalent of simple.rectangularGrid but for a mesh. Additionally, dx,dy,dz can be either integers or div (1D list or array). In case of list/array, first and last numbers should be 0.0 and 1.0 if the desired grid has to be inside the region 0.,0.,0. to 1.,1.,1.
from __future__ import print_function
    If isopHex is specified, a convenient set of control points for the isoparametric transformation hex64 is also returned.
    TODO: include other optons to get the control points for other isoparametric transformation for hex."""
    sgx, sgy, sgz=dx, dy, dz
    if type(dx)!=int:sgx=len(dx)-1
    if type(dy)!=int:sgy=len(dy)-1
    if type(dz)!=int:sgz=len(dz)-1
    n3=regularGrid([0., 0., 0.],[1., 1., 1.],[sgx, sgy, sgz])
    if type(dx)!=int:n3[..., 0]=array(dx).reshape(-1, 1, 1)
    if type(dy)!=int:n3[..., 1]=array(dy).reshape(-1,  1)
    if type(dz)!=int:n3[..., 2]=array(dz).reshape(-1)
    nyz=(sgy+1)*(sgz+1)
    xh0= array([0, nyz, nyz+sgz+1,0+sgz+1 ])
    xh0= concatenate([xh0, xh0+1], axis=1)#first cell
    hz= array([xh0+j for j in range(sgz)])#z column
    hzy= array([hz+(sgz+1)*j for j in range(sgy)])#zy 2D rectangle
    hzyx=array([hzy+nyz*k for k in range(sgx)]).reshape(-1, 8)#zyx 3D
    if isophex=='hex64': return Coords(n3.reshape(-1, 3)), hzyx.reshape(-1, 8), regularGrid([0., 0., 0.], [1., 1., 1.], [3, 3, 3]).reshape(-1, 3)#control points for the hex64 applied to a basic struct hex grid
    else: return Coords(n3.reshape(-1, 3)), hzyx.reshape(-1, 8)



def findBisectrixUsingPlanes(cpx, centx):
    """it returns a bisectrix-points at each point of a Polygon (unit vector of the bisectrix). All the bisectrix-points are on the side of centx (inside the Polygon), regardless to the concavity or convexity of the angle, thus avoiding the problem of collinear or concave segments. The points will point towards the centx if the centx is offplane. It uses the lines from intersection of 2 planes."""
    cx=concatenate([cpx, [cpx[0]]], axis=0)
    cx=concatenate([cx[:-1], cx[1:]], axis=-1).reshape(-1, 2, 3)
#    draw(Formex(cx))
#    drawNumbers(Formex(cx))
#    drawNumbers(Formex(cx[0]))
    if centx==None:centx=cx.reshape(-1, 3).mean(axis=0)
    #draw(Formex(centx))
    nx0=[]
    for i in range(cx.shape[0]):
        v0=cx[i, 1]-cx[i, 0]
        z0=cross(v0, centx-cx[i, 0])
        nx0.append(normalize(cross(z0, v0)))
    c1=cx[:, ::-1]
    nx1=[]
    for i in range(c1.shape[0]):
        v0=c1[i, 1]-c1[i, 0]
        z0=cross(v0, centx-c1[i, 0])
        nx1.append(normalize(cross(z0, v0)))
    nx1=roll(nx1, 1, 0)
    nx=0.5*(nx0+nx1)
    nx=normalize(nx)#new added
    #[draw(Formex([[cpx[i], cpx[i]+nx[i]]])) for i in range(cpx[:].shape[0])]
    return nx



def cpBoundaryLayer(BS,  centr, issection0=False,  bl_rel=0.2):
    """it takes n points of a nearly circular section (for the isop transformation, n should be 24, 48 etc) and find the control points needed for the boundary layer. The center of the section has to be given separately.
    -issection0 needs to be True only for the section-0 of each branch of a bifurcation, which has to share the control points with the other branches. So it must be False for all other sections and single vessels.
    This implementation for the bl (separated from the inner lumen) is needed to ensure an optimal mesh quality at the boundary layer in terms of angular skewness, needed for WSS calculation."""
    if BS.shape[0]%2!=0:raise ValueError,"BE CAREFUL: the number of points along each circular section need to be even to split a vessel in 2 halves with the same connectivity!"
    bllength=length(BS-centr).reshape(-1, 1)*bl_rel
    blvecn=findBisectrixUsingPlanes(BS, centr)#unit vectors similar to bisectrix but obtained as intersection of planes
    #draw(Formex(centr))
    #print bisbs
    if issection0:#inside the bifurcation center the 3 half sections need to touch each others on 1 single line. Thus, because the control points on this line (the bifurcation axis) needs to be the same, these 2 special points are defined differently, but only at the first section of each branches.
        blvecn[0]=normalize(centr-BS[0])
        midp=BS.shape[0]/2#is 12 if use 24 control points, will be 24 with 48 cp.
        blvecn[midp]=normalize(centr-BS[midp])
    blvec=blvecn*bllength
    cpblayer=array([BS+blvec*i for i in [1., 2/3.,1/3.,0.]])
    cpblayer=swapaxes(cpblayer, 0, 1)
    #draw(Formex(cpblayer[:].reshape(-1, 3)))
    #drawNumbers(Formex(cpblayer[0]))
    r4= concatenate( [i*3+arange(4) for i in arange(cpblayer.shape[0]/3)])
    r4[-1]=0
    r4= r4.reshape(-1, 4)
    return cpblayer[r4].reshape(-1, 16, 3), cpblayer[:, 0]#control points of the boundaary layer, points on the border with the inner part of the lumen

def cpQuarterLumen(lumb, centp, edgesq=0.75, diag=0.6*2**0.5, verbos=False):
    """control points for 1 quarter of lumen mapped in quad regions. lumb is a set of points on a quarter of section. centp is the center of the section. The number of poin I found that edgesq=0.75, diag=0.6*2**0.5 give the better mapping. Also possible edgesq=0.4, diag=0.42*2**0.5. Currently, it is not perfect if the section is not planar."""
    arcp=lumb.copy()
    arcsh= arcp.shape[0]-1
    xcp1, xcp3=centp+(arcp[[arcsh, 0]]-centp)*edgesq
    xcp2=centp+(arcp[arcsh/2]-centp)*diag
    nc0=array([centp,xcp1, xcp2, xcp3])#new coord0
    grid16= regularGrid([0., 0., 0],[1., 1., 0.],[3, 3, 0]).reshape(-1, 3)#grid
    ncold=grid16[[0, 12, 15, 3]]#old coord
    fx=arcsh/6
    sc=array([1./fx, 1./fx, 0.])
    grid16=Formex([grid16]).replic2(fx, fx, 1., 1.).scale(sc)[:]  
    gridint= Isopar('quad4',nc0,ncold).transform(grid16)#4 internal grids
    xa0=Coords.interpolate(Coords(ncold[[3]]), Coords(ncold[[2]]),div=3*fx)
    xa1=Coords.interpolate(Coords(ncold[[2]]), Coords(ncold[[1]]),div=3*fx)
    xa= concatenate([xa0, xa1[1:]], axis=0)
    xa= Isopar('quad4',nc0,ncold).transform(xa).reshape(-1, 3)
    xar3=concatenate( [i*3+arange(4) for i in arange(2*fx)])
    gridext=array(zip(xa[xar3], arcp[xar3]))
    gridext=Coords.interpolate(Coords(gridext[:, 0]), Coords(gridext[:, 1]),div=3)
    gridext=swapaxes(gridext, 0, 1) .reshape(2*fx,16,  3)
    if verbos:print('---one Quarter of section is submapped in %d internal and %d transitional quad regions---'%(gridint.shape[0],gridext.shape[0] ))
#    gridG=concatenate([gridint, gridext], axis=0)
#    print '---one Quarter of section is submapped in %d quad regions---'%gridG.shape[0]
#    [draw(Formex(fo).setProp(i)) for  i, fo in enumerate(gridG) ]
#    for i in gridG:
#        di= [drawNumbers(Formex(i))]
#        zoomAll()
#        sleep(0.2)
#        undraw(di)
#    exit()
    return gridint, gridext





def visualizeSubmappingQuadRegion(sqr, timewait=None):
    """visualilze the control points (-1,16,3) in each submapped region and check the quality of the region (which will be inherited by the mesh crossectionally)"""
    sqr3=regularGrid([0., 0., 0.],[1., 1., 0.],[3, 3, 0]).reshape(-1, 3)#base old coords
    sqrn3, sqre3=structuredQuadMeshGrid(3, 3)#quad mesh to map
    for  i, f in enumerate(sqr):
        sqr0=Formex(sqrn3).isopar('quad16',f,sqr3)[:].reshape(-1, 3)
        sqr0=Formex(sqr0[sqre3])
        draw(Formex(f).setProp(i+1))
        draw(sqr0.setProp(i+1))
        di= [drawNumbers(Formex(f))]
        zoomAll()
        ###here check quality of G0: if it is not good enough, change the parameters of the cpQuarterLumen(quartsec[0], oc, edgesq= ..., diag= ...)
        if timewait!=None:sleep(timewait)
        undraw(di)
#clear()
##draw(Formex([0., 0., 0.]))
#hc=circle(2, 2, 360).scale(2).rotate(90).rotate(180, 1)[:]
#hc=hc[:, 0].append([hc[-1, 1]])
#hc=divPolyLine(hc,24*1, closed=True)[:-1]#points that define 1 circular section (normally 24*int)
##oc=[-0.3, 0., 0.]#center of the circular section
#oc=Coords([0., 0., 0.2])#center of the circular section
#hc[:, 2]=hc[:, 2]+0.2*abs(hc[:, 0])**2.
##draw(hc)
##draw(Formex(oc))
##drawNumbers(hc)
##draw(PolyLine(hc, closed=True))
#
#HC=[hc]
#OC=[oc]
#for i in range(5):
#    HC.append(HC[-1].translate([0.1, -0.2, 3.]).rotate(0,0 ))
#    OC.append(OC[-1].translate([0.1, -0.2, 3.]).rotate(0,0 ))
#OC, HC=array(OC), array(HC)
#OC[:, 1]+=0.001*OC[:, 2]**3
#OC[:, 0]+=0.02*OC[:, 2]
#HC[:, :, 1]+=0.001*HC[:,:,  2]**3
#
#[draw(Formex(i), color='red') for i in OC]
#[draw(PolyLine(i, closed=True)) for i in HC]

def cpOneSection(hc, oc=None,isBranchingSection=False, verbos=False ):
    """hc is a numbers of points on the boundary line of 1 almost circular section. oc is the center point of the section. It returns 3 groups of control points: for the inner part, for the transitional part and for the boundary layer of one single section"""
    
    ##if the center is not given, it is calculated from the first and the half points of the section
    if oc==None:oc=(hc[0]+hc[hc.shape[0]/2])*0.5
    
    ##create control points for the boundary layer of 1 full section.
    if verbos:
        if isBranchingSection:print("--BRANCHING SECTION:section located at the center of the bifurcation")
    cpbl, hlum=cpBoundaryLayer(hc,  centr=oc, issection0=isBranchingSection)
    
    ##split the inner lumen in quarters and check if the isop can be applied
    sectype= hlum.shape[0]/24.
    if sectype!=float(int(sectype)): raise ValueError,"BE CAREFUL: the number of points along each circular section need to be 24*int in order to allow mapping!"
    else: 
        if verbos:print("----the number of points on 1 slice is suitable for ISOP MESHING----")
    npq= sectype*6
    hlum1=concatenate([hlum, [hlum[0]]], axis=0)
    quartsec=[hlum1[npq*i:npq*(i+1)+1] for i in range(4)]#split in quarters
    
    ##created control points of each quarter
    cpis, cpts=[], []
    for q in quartsec:
        i=cpQuarterLumen(q, oc)
        cpis.append(i[0] )#control points of the inner part of a section
        cpts.append(i[1])#control points of the transitional part of a section
    #visualizeSubmappingQuadRegion(cpss, timewait=None)#
    return array(cpis).reshape(-1, 16, 3), array(cpts).reshape(-1, 16, 3), array(cpbl)

def cpAllSections(HC, OC, start_end_branching=[False, False]):
    """control points of all sections divided  in 3 groups of control points: for the inner part, for the transitional part and for the boundary layer. if start_end_branching is [True,True] the first and the last section are considered bifurcation sections and therefore meshed differently. """
    isBranching=zeros([HC.shape[0]], dtype=bool)
    isBranching[[0, -1]]=start_end_branching#if first section is a branching section
    #print isBranching
    cpain, cpatr, cpabl=[], [], []
    for hc, oc, isBr in zip(HC, OC, isBranching):
        i=cpOneSection(hc, oc,  isBranchingSection=isBr )
        cpain.append(i[0]), cpatr.append(i[1]), cpabl.append(i[2])
    cpain, cpatr, cpabl=[ array(i) for i in [cpain, cpatr, cpabl] ]
    print('# sections= %d,  # inner quad reg = %d, # trans quad reg = %d, # boundary-layer quad reg = %d' %(cpain.shape[0],cpain.shape[1], cpatr.shape[1], cpabl.shape[1]))
    if start_end_branching==[True, True]: print('--this vessel BIFURCATES both at FIRST AND LAST section')
    if start_end_branching==[True, False]: print('--this vessel BIFURCATES at FIRST section')
    if start_end_branching==[False, True]: print('--this vessel BIFURCATES at LAST section')
    if start_end_branching==[False, False]: print('--this vessel DOES NOT BIFURCATE')
    #[visualizeSubmappingQuadRegion(i) for i in cpain]
    #[visualizeSubmappingQuadRegion(i) for i in cpatr]
    #[visualizeSubmappingQuadRegion(i) for i in cpabl]
    return cpain, cpatr, cpabl
    
##FIRST STEP ----- FROM SPLINE-PTS to CONTROL-POINTS-QUAD16-------------------------
#cpAin, cpAtr, cpAbl=cpAllSections(HC, OC, [False, False])#control points of all sections grouped in inner, trans and boundary layer. Each contains number of long_slice, number of hex-reg, 16, 3.
##[visualizeSubmappingQuadRegion(i) for i in cpAin]
##pause()
def cpStackQ16toH64(cpq16):
    """sweeping trick: from sweeping sections longitudinally to mapping hex64: ittakes -1,16,3 (cp of the quad16) and groups them in -1,64,3 (cp of the hex63) but slice after slice: [0,1,2,3],[1,2,3,4],[2,3,4,5],... It is a trick to use the hex64 for sweeping along an arbitrary number of sections."""
    cpqindex= concatenate( [i+arange(4) for i in arange(cpq16.shape[0]-3)]).reshape(-1, 4)
    cpq16t=swapaxes(cpq16[cpqindex], 1, 2)
    shcp=cpq16t.shape
    return cpq16t.reshape( [ shcp[0], shcp[1],64, 3 ] )

##SECOND STEP ----- FROM CONTROL-POINTS-QUAD16 to CONTROL-POINTS-HEX64-------------------------
#hex_cp=[cpStackQ16toH64(i) for i in [cpAin, cpAtr, cpAbl] ]#control points for hex64 divided in 3 groups: central, transition, and boundary layer. The stacking uses the TRICK for sweeping.
#
##THIRD STEP ----- specifing mesh_blocks parameters and build 3 mesh blocks
#ncirc=3
#nlong=2
#ntr=2#int or div
#nbl=[0., 0.4, 0.7, 0.8, 0.9, 1.]#int or div
#in_block=structuredHexMeshGrid(nlong, ncirc,ncirc,  isophex='hex64')
#tr_block=structuredHexMeshGrid(nlong, ncirc,ntr,  isophex='hex64')
#bl_block=structuredHexMeshGrid(nlong, ncirc,nbl,  isophex='hex64')

def mapHexLong(mesh_block, cpvr):
    """map a structured mesh (n_block, e_block, cp_block are in mesh_block) into a volume defined by the control points cpvr (# regions longitudinally, # regions in 1 cross sectionsm, 64, 3 ). cp_block are the control points of the mesh block. It returns nodes and elements. Nodes are repeated in subsequently mapped regions !
    TRICK: in order to make the mapping working for an arbitrary number of sections the following trick is used: of the whole mesh_block, only the part located between the points 1--2 is meshed and mapped between 2 slices only. Thus, the other parts 0--1 and 2--3 are not mapped. To do so, the first and the last slice need to be meshed separately: n_start 0--1 and n_end 2--3."""
    n_block, e_block, cp_block=mesh_block
    n_start, n_body, n_end=[n_block.scale([1./3., 1., 1.]).translate([i/3., 0., 0.]) for i in range(3)]
    cp_start, cp_body, cp_end=cpvr[0], cpvr.reshape(-1, 64, 3), cpvr[-1]
    n=[ [Coords(n_tract).isopar('hex64',cpi,cp_block) for cpi in cp_tract] for n_tract, cp_tract in zip([n_start, n_body, n_end], [cp_start, cp_body, cp_end]) ]
    n=concatenate(n, axis=0)
    return n, e_block


def mapQuadLong(mesh_block, cpvr):
    """TRICK: in order to make the mapping working for an arbitrary number of sections the following trick is used: of the whole mesh_block, only the part located between the points 1--2 is meshed and mapped between 2 slices only. Thus, the other parts 0--1 and 2--3 are not mapped. To do so, the first and the last slice need to be meshed separately: n_start 0--1 and n_end 2--3."""
    n_block, e_block, cp_block=mesh_block
    n_start, n_body, n_end=[n_block.scale([1./3., 1., 1.]).translate([i/3., 0., 0.]) for i in range(3)]
    cp_body=cpvr.reshape(-1, 16, 3)
    cp_start=cp_body[:8]
    cp_end=cp_body[-8:]
    n=[ [Coords(n_tract).isopar('quad16',cpi,cp_block) for cpi in cp_tract] for n_tract, cp_tract in zip([n_start, n_body, n_end], [cp_start, cp_body, cp_end]) ]
    n=concatenate(n, axis=0)
    return n, e_block

##FORTH STEP ----- map the blocks of the 3 regions
#in_mesh, tr_mesh, bl_mesh=[mapHexLong(v_block, v_cp) for v_block, v_cp in zip([in_block, tr_block, bl_block], hex_cp)]
#M= [m[0][:, m[1]].reshape(-1, 8, 3) for m in [in_mesh, tr_mesh, bl_mesh] ]
#
#[draw(Formex(M[i], eltype='Hex8').setProp(i)) for  i in range(3)]








