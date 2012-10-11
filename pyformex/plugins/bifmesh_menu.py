# $Id$   pyformex
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
"""Bifurcation meshing menu.

This menu provides functions for meshing a vascular bifurcation with
a structured hexahedral mesh. These tools were developed as part of the
PhD research by Gianluca De Santis at bioMMeda (Ghent University).

"""
from __future__ import print_function
from gui.draw import *

import pyformex as pf
import plugins.vascularsweepingmesher as vsm
import plugins.geometry_menu as gm

from plugins.fe import mergedModel
from connectivity import connectedLineElems
from plugins.trisurface import fillBorder
from plugins.draw2d import *
from gui import menu
from utils import runCommand

import sys

# global data

_name_ = 'bifmesh_menu'

_surface = None


##################### Functions ####################################


def importGeometry():
    gm.importSurface()


def positionGeometry():
    pass


def inputCentralPoint():
    clear()
    view('front')
    drawSurface()
    transparent()
    obj = drawObject2D('point',npoints=1,zvalue=0.)
    obj.specular = 0.
    if obj is not None:
        export({'central_point':obj})
        drawCentralPoint()


def getData(*args):
    """Return global data"""
    try:
        res = [ named(a) for a in args ]
        if len(res) == 1:
            res = res[0]
        return res
    except:
        error("I could not find all the data")

def getDataItem(data,item):
    """Return an item from a dict"""
    items = getData(data)
    return items.get(item,None)

def drawOption(item):
    """Return draw option"""
    return getDataItem('draw_options',item)

def sliceData(item):
    """Return slice data"""
    return getDataItem('slice_data',item)


def vecbisectrix(p0, p1, cx):
    return normalize(normalize(p0-cx)+normalize(p1-cx))


def longestItem(geomlist):
    """Return the geometric part with the most elements."""
    return geomlist[argmax([m.nelems() for m in geomlist])]

def largestSubMesh(M):
    """Return the largest part with single prop value."""
    p = M.propSet()
    n = array([ (M.prop==i).sum() for i in p ])
    i = n.argmax()
    return M.withProp(p[i])


def selectPart(M,x0,x1):
    """
    Select part of section between x0 and x1

    section is a list of Mesh objects.
    All meshes are clipped to the parts between planes through x0 and x1
    perpendicular to the line x0-x1.
    The mesh with the largest remaining part is returned.
    """
    #meshlist = [ m.clipAtPlane(x1,x0-x1).clipAtPlane(x0,x1-x0) for m in meshlist ]
    #return meshToPolyLine(longestItem(meshlist))
    M =  M.clipAtPlane(x1,x0-x1, nodes='all').clipAtPlane(x0,x1-x0, nodes='all')
    if M.prop is not None:
        M = largestSubMesh(M)
    return meshToPolyLine2(M)


def cutPart(M,x0,x1):
    """Cut part of section at plane thru x0 and return part between x0 and x1"""
    M = M.clipAtPlane(x1,x0-x1, nodes='all')

    meshlist = M.splitProp().values()

    meshlist = [ meshToPolyLine2(m) for m in meshlist ]
    #meshlist = [ p.cutWithPlane(x0,x1-x0,side='+') for p in meshlist ]
    draw(meshlist, flat=True, alpha=1, linewidth=3)


    #print erf
    meshlist = olist.flatten(meshlist)
    if len(meshlist) == 1:
        pl = meshlist[0]
    elif len(meshlist) == 2:
        p1,p0 = meshlist
        pl = PolyLine(Coords.concatenate([p0.coords,p1.coords]))
    else:
        print([ p.nelems() for p in meshlist])
        [ draw(p,color=c) for p,c in zip(meshlist,pf.canvas.settings.colormap) ]
        pl = longestItem(meshlist)
    return pl


def getPart(section,x0,x1,cutat=-1):
    """Return a cut or a selected part, depending on cutat parameter."""
    if cutat == 0:
        #cut at plane x0
        return cutPart(section,x0,x1)

    elif cutat == 1:
        #cut at plane x1
        return cutPart(section,x1,x0)

    else:
        # select parts between x0 and x1 only
       return selectPart(section,x0,x1)


def connectPolyLines(plist):
    """Connect a set of PolyLines into a single one.

    The PolyLines are connected in order of the distance between the
    endpoints.
    """
    pass

def meshList(coords,elems):
    """Create a list of Meshes from intersection lines"""
    return [ Mesh(coords,e) for e in elems ]


def meshToPolyLine1(m):
    "Convert a 2-plex mesh with ordered segments to a PolyLine."""
    pts = m.elems[:,0]
    if m.elems[-1,-1] != m.elems[0,0]:
        pts = concatenate([pts,m.elems[-1:,-1]])
    return PolyLine(m.coords[pts])

def meshToPolyLine2(m2):
    "Convert a 2-plex mesh with ordered segments to a PolyLine."""
    'order line elements from boundary points'
    # Remove degenerate and doubles
    m2=m2.fuse().compact()
    ##m2 = Mesh(m2.coords,m2.elems.removeDegenerate().removeDoubles())
    m2 = Mesh(m2.coords,m2.elems.removeDegenerate().removeDuplicate())
    # Split in connected loops 
    parts = connectedLineElems(m2.elems)
    prop = concatenate([ [i]*p.nelems() for i,p in enumerate(parts)])
    elems = concatenate(parts,axis=0)
    m2= Mesh(m2.coords,elems,prop=prop)
    pts = m2.elems[:,0]
    if m2.elems[-1,-1] != m2.elems[0,0]:
        pts = concatenate([pts,m2.elems[-1:,-1]])
        return PolyLine(m2.coords[pts], closed=False)
    else: return PolyLine(m2.coords[pts], closed=True)


def slicer(S,s0,s1,cutat=-1,visual=False):
    """Cut a surface with a series of planes.

    - S: surface
    - s0,s1: Coords arrays with the same number of points

    Each pair of corresponding points of s0 and s1 defines a plane parallel
    to the z-axis
    """
    if not isinstance(S,TriSurface):
        S = TriSurface(S)
    if not isinstance(s0,Coords):
        s0 = s0.coords
    if not isinstance(s1,Coords):
        s1 = s1.coords

    v1 = s1-s0 # direction of the knife
    zdir = unitVector(2) # direction of the z axis
    ncut = cross(zdir,v1) # normals on the cut planes

    sections = [ S.intersectionWithPlane(p0,n0) for p0,n0 in zip(s0,ncut)]

    if visual:
        clear()
        draw(sections,color='magenta')


    sections = [ getPart(s,x0,x1,cutat=cutat) for s,x0,x1 in zip(sections,s0,s1)]

    if visual:
        clear()
        draw(sections,color='cyan')

    # Orient all PolyLines so that first points is at max. z-value
    for i,s in enumerate(sections):
        if s.coords[0,2] < s.coords[-1,2]:
            #print "Reversing PL %s" % i
            sections[i] = s.reverse()
    
    #[draw([s.coords[-1] for s in sections], marksize=1) ]

    return sections


def sliceBranch(S,cp,s0,s1,cl,nslices):
    """Slice a single branch of the bifurcation

    - `S`: the bifurcation surface, oriented parallel to xy. 
    - `cp` : the center of the bifurcation. 
    - `s0`, `s1`: the control polylines along the branch. 
    - `cl`: the centerline of the branch.
    - `nslices`: the number of slices used to approximate the branch
      surface. 
    """
    visual = drawOption('visual')

    cl = cl.approx(ntot=nslices)
    s0 = s0.approx(ntot=nslices)
    s1 = s1.approx(ntot=nslices)

    h0 = slicer(S,s0,cl,cutat=-1,visual=visual)
    if visual:
        clear()
        draw(h0,color='black')
    h1 = slicer(S,cl,s1,cutat=-1,visual=visual)
#    if visual:
#        draw(h0,color='red')
#        draw(h1,color='blue')
    return [h0,h1]


def sliceIt():
    """Slice a surface using the provided slicing data

    """
    S,cp = getData('surface','central_point')
    try:
        hl = named('control_lines')
        cl = named('center_lines')
    except:
        centerlines()
        hl = named('control_lines')
        cl = named('center_lines')

    nslice = sliceData('nslice')

    pf.GUI.setBusy(True)
    h = [sliceBranch(S,cp,hl[2*i],hl[2*i+1],cl[i],nslice[i]) for i in range(3) ]

    ##corrections of some points of the cross-PolyLines because sometimes, due to the cutting, different pts appear at the connections between semi-branches (0,1) and branches (at center-splines top and bottom)
    ##merge the points between half branches
    for ibr in [0, 1, 2]:
        pfl0=array([hi.coords[[0, -1]] for hi in h[ibr][0]])#point first and last of each half cross section
        pfl1=array([hi.coords[[0, -1]] for hi in h[ibr][1]])
        temp=(pfl0+ pfl1)*0.5
        for i in range(len(temp)):h[ibr][0][i].coords[[0, -1]]=h[ibr][1][i].coords[[0, -1]]=temp[i]

    ##mergethe 3 points on the top  and the 3 points on the bottom (at the bif center)
    t0= array([ h[ibr][sbr][0].coords[0] for ibr in [0, 1, 2] for sbr in [0, 1]]).mean(axis=0)
    t1= array([ h[ibr][sbr][0].coords[-1] for ibr in [0, 1, 2] for sbr in [0, 1]]).mean(axis=0)
    for ibr in [0, 1, 2]:
        for sbr in [0, 1]:
            h[ibr][sbr][0].coords[0]=t0 
            h[ibr][sbr][0].coords[-1]=t1 

    export({'cross_sections':olist.flatten(h)})
    export({'cross_sections_backup':olist.flatten(h)})#to recover from overwriting when creating the outer cross sections
    clear()
    drawCenterLines()
    drawCrossSections()
    pf.GUI.setBusy(False)



def splineIt():
#    print 'how many splines circumferentially? with tangency between branches ?'
#    res = askItems([
#    ['niso', 12],
#    ['with_tangence', True]
#    ])
#
#    niso= (res['niso'])
#    smoothconnections=res['with_tangence']
    
    niso=12
    smoothconnections=True
#
    
    nslice =  sliceData('nslice')
    cs = getData('cross_sections')

    # cross splines
    spc = [[ ci.approx(ntot=niso) for ci in c ] for c in cs]
    export({'cross_splines':spc})
    clear()
    drawCrossSplines()

    # axis spline
    [hi0,  hi1, hi2,hi3, hi4, hi5]=[array([h.coords for h in hi]) for hi in spc]
    clv=[ PolyLine( (hi[:, 0]+hi[:, -1])*0.5  ) for hi in [hi0, hi2, hi4] ]
    axis = [BezierSpline(c.coords,curl=1./3.).approx(ntot=nsl) for c,nsl in zip(clv,nslice)]
    export({'axis_splines':axis})
    drawAxisSplines()

    # longitudinal splines
    # 2 options: with or without continuity of the tangent at the connections between contiguous splines, except for the center-Top and center-Bottom

    if (smoothconnections)==True:##with continuity
        TBspl = [BezierSpline([ci.coords[j] for ci in spc[i]],curl=1./3.) for j, i in zip([0, niso, 0, niso, 0, niso], range(6))]#6 splines at the center-Top and center-Bottom
        prolspc=[[spc[i][1]]+spc[j] for i, j in [[5, 0], [2, 1], [1, 2], [4, 3], [3, 4], [0, 5] ] ]#prolonged sections, to give correct tangence
        mspl = [ [BezierSpline([ci.coords[j] for ci in prolspc[i]],curl=1./3.) for j in range(1, niso)] for i in range(6)]#all splines between Top and Bottom splines
        spl=[]
        for i in range(6):#6 semi-branches
            smspl=[TBspl[i]]#splines of 1 semi-branch
            for ispl in mspl[i]:


                ### problem with revision revision 1513 solved
                #coordb= ispl.coords[1:] #remove the last piece (only needed for the tangence)#valid up to revision -r 1512
                coordb=append(ispl.coords, [[0., 0., 0.]]*2, axis=0).reshape(-1, 3, 3)[1:]#from revision -r 1513


                sct=1.# factor to move the last control point=amount of tangence between contiguous splines
                coordb[0, 1]=(coordb[0, 1]-coordb[0, 0])*sct+coordb[0, 0]       
                smspl=smspl+[BezierSpline(coords=coordb[:, 0], control=coordb[:-1, [1, 2]])]
            spl.append(smspl)
    else: ##without continuity
        spl = [[BezierSpline([ci.coords[j] for ci in c],curl=1./3.) for j in range(niso+1)] for c in spc]
    export({'long_splines':spl})
    drawLongSplines()


def divPolyLine(pts, div, closed=None):
    """it takes the points, build a PolyLine and divide it in pieces. If div is an integer, it returns the point used to divide the polyline in div-equal pieces, otherwise it returns the points needed to divide the PolyLine at the values of curvilinear abscissa of the array div. So div=array([0., 0.5, 0.7, 0.85, 0.95, 0.97, 1.  ])
or div=4. If closed =='closed' the last point has to be coincident to the 1st one !!! Even if it is closed the last point (==to the first one) div=...1. needs to be present because it will be removed."""
    Pts = PolyLine(pts)
    if type(div)==int:
        at = Pts.atLength(div)
        if closed=='closed':return Pts.pointsAt(at)[:-1]
        return Pts.pointsAt(at)  
    if type(div)==ndarray:
        At= Pts.atLength(div)
        if closed=='closed':return Pts.pointsAt(At) [:-1]
        return Pts.pointsAt(At)
    else:  raise ValueError,"wrong type for div"


def seeding3zones(nseeds=[10, 10],zonesizes=[0.3, 0.3]):
    """it creates a 1D array of floats between 0.0 and 1.0 splitting this range in 3 zones: the first and the last have sizes determined in zonesizes and are populated with the numbers in nseeds. A trnasitional zone is calculated approximating a geometrical series (the next segment is as long as the previous*power)."""
    #nseeds = number of seeds in zone0 and zone1
    #sizesizes are the zone sizes in percentages   
    #transition zone is seeded using an approximated geometrical series

    if sum(zonesizes)>1.:raise 'the sum of zone lengths has to be < 1.'
    seed0, seed1= [arange(nseeds[i]+1.)/float(nseeds[i])*zonesizes[i] for i in range(2) ]
    seed1=1.+seed1-seed1[-1]

    transzone=seed1[0]-seed0[-1]#transition zone
    near0=seed0[-1]-seed0[-2]
    near1=seed1[-1]-seed1[-2]
    #geometrical series: nextsegment = previoussegment**power
    ntransseeds= (  (log(near1/near0)) / log ( (transzone+near1)/(transzone+near0) )  ) -1.
    powertrans=(near1/near0)**(1./(ntransseeds+1.))
    napproxseeds=int(round(ntransseeds))
    xtrans= near0*array([(powertrans**i) for i in range(1, napproxseeds+1)])
    xtrans= cumsum(xtrans)
    xtrans=xtrans[:-1]*transzone/xtrans[-1]
    #if  len(xtrans)==0: warning('There is not enough space to fit a transition zone!')
    seedingarray=[seed0,xtrans+seed0[-1], seed1 ]  
    return seedingarray#concatenate(seedingarray)


def drawLongitudinalSeeding(H, at):
    for i, j in zip([0, 2, 4], range(3)):#semibranch,at
        l0=H[i][0].subPoints(10)#0,2,4,napproxlong
        draw(divPolyLine(l0,at[j][0]), color='green', marksize=7  ,flat=True, alpha=1)
        draw(divPolyLine(l0,at[j][1]), color='yellow', marksize=7 ,flat=True, alpha=1)
        draw(divPolyLine(l0,at[j][2]), color='red', marksize=7 ,flat=True, alpha=1)
    zoomAll()

dialog = None
SA = None
def inputLongitudinalSeeds():
    """Interactive input of the longitudinal meshing seeds.

    """
    global dialog
    
    def show():
        """Show the current seeds"""
        global dialog,SA
        undraw(SA)
        dialog.acceptData()
        res = dialog.results
        if not res:
            return
        sat = [ seeding3zones(nseeds=eval(res['nseeds%s'%i]),zonesizes=eval(res['ratios%s'%i])) for i in range(3)]
        print('# of seeds in branches: %s'% [sum([len(i) for i in j])-1 for j in sat])
        SA = drawLongitudinalSeeding(long_splines,sat)
        return sat

    def accept():
        """Accept the current seeds"""
        sat = show()
        if sat:
            export({'sat':[concatenate(ati) for ati in sat]})
        dialog.accept()

    def help():
        showInfo("Set the seeding parameters for the 3 branches. Each branch is divided in 3 parts: close to the bifurcation, far from the bifurcation and the central part between these two. Only the data for the first two need to be entered. The 3rd (central) is calculated as transition.")
        
    clear()
    long_splines=named('long_splines')
    draw(long_splines, linewidth=1, color='gray' ,flat=True, alpha=1)
    drawNumbers(Formex( [long_splines[i][0].subPoints(1)[-1]  for i in  [0, 2, 4] ]))

    nseeds = '[5,3 ]'
    ratios = '[0.3, 0.4]'
    dialog = Dialog(
        caption = 'Ratio Seeding Spline',
        items = [
            _I('nseeds0',nseeds),
            _I('ratios0',ratios),
            _I('nseeds1',nseeds),
            _I('ratios1',ratios),
            _I('nseeds2',nseeds),
            _I('ratios2',ratios),
            ],
        actions = [('Cancel',),('Show',show),('Accept',accept),('Help',help)],
        )
    dialog.move(100,100)
    dialog.getResult()


def seedLongSplines (H, at,  curvedSection=True, nPushedSections=6, napproxlong=40, napproxcut=40 ):
    """it takes the Longitudinal Splines (6 semi-braches, 12 longitudinal_splines) and return the control points on them (n_long, n_cross=24) and on the centerlines (n_long)."""
    #at=[concatenate(ati) for ati in at]
    nplong=[ at[0].shape[0]-1, at[1].shape[0]-1, at[2].shape[0] -1]
    H=[ H[i]+[H[i+(-1)**i][0]] for i in range(6) ]#add 1 splines at each semibranch
    H=[[h.subPoints(napproxlong) for h in hi] for hi in H ]
    B=[array( [ divPolyLine(longi,at[bra]) for longi in h]) for h, bra in zip(H, [0, 0, 1, 1, 2, 2])]#seeds control points longitudinally

    def BezierCurve(X):
        """Create a Bezier curve between 4 points"""
        ns = (X.shape[0]-1) / 3
        ip = 3*arange(ns+1)
        P = X[ip]
        ip = 3*arange(ns)
        ic = column_stack([ip+1,ip+2]).ravel()
        C = X[ic].reshape(-1,2,3)
        return BezierSpline(P,control=C,closed=False)

    def nearestPoints2D(pt0, pt1):
        """P0 and P1 and 2D arryas. It takes the closest point of 2 arrays of points and finds the 2 closest points. It returns the 2 indices."""
        if pt0.shape[1]!=2:raise ValueError,"only for 2D arras (no z)"
        np= (pt0.reshape(-1, 1, 2)-pt1.reshape(1, -1, 2))#create a matrix!!!
        npl=(np[:,:,  0]**2+np[:,:,  1]**2)**0.5
        nearest= where(npl==npl.min())
        return nearest[0][0],nearest[1][0]

    def cutLongSplinesWithCurvedProfile(sideA, sideB, curvA, curvB, npb):
        #create 2D cutting curved profiles given 3 points
        ssh= sideA.shape[0]
        hsh=int(ssh/2)#half point
        s12=(sideA[0]+ sideB[0])*0.5
        cutProfilePts=array([sideA[hsh], s12, s12, sideB[hsh]])
        
        #[draw(BezierCurve(cutProfilePts[:, i]), alpha=1, flat=True) for i in range(1, sideA[0].shape[0]-1)]
        cutProfile2D=array([BezierCurve(cutProfilePts[:, i]).subPoints(npb)[:, :2] for i in range(1, sideA[0].shape[0]-1)])
        #cuts long splines with curved profiles
        cutcurvA, cutcurvB=[], []
        for ind in range(ssh):
            hlongA, hlongB=curvA[ind], curvB[ind]
            iA,iB =[0], [0]
            for il in range(0, cutProfile2D.shape[0]):
                iA.append(nearestPoints2D(hlongA[:, :2], cutProfile2D[il])[0])
                iB.append(nearestPoints2D(hlongB[:, :2], cutProfile2D[il])[0])
            iA, iB= append(iA, -1), append(iB, -1)
            #draw(Formex(hlongA[iA]), marksize=3, flat=True, alpha=1)
            #draw(Formex(hlongB[iB]), marksize=3, flat=True, alpha=1.)
            cutcurvA.append(hlongA[iA])
            cutcurvB.append(hlongB[iB])
        return array(cutcurvA).reshape(ssh, -1, 3), array(cutcurvB).reshape(ssh, -1, 3)


    if curvedSection==True:
        TB=[  cutLongSplinesWithCurvedProfile(B[i], B[i+1], H[i], H[i+1], napproxcut) for i in [0, 2, 4]  ]
        TB=TB[0]+TB[1]+TB[2]
        if nPushedSections>0:#this part pushes 'nPushedSections' sections closer to the bifurcation center!
            cv=range(nPushedSections+2)
            mcv= array(cv, dtype=float)/cv[-1]
            for tbi, bi in zip(TB,B ):
                for i in cv:  tbi[:, i]=(tbi[:, i]*mcv[i] +bi[:,i]*(1.-mcv[i]) )
        B=TB

    B=[swapaxes(b, 0, 1) for b in B]
    cent012=[(b[:, 0]+b[:, -1])*0.5 for b in [B[0], B[2], B[4]] ]
    lum012=[ concatenate([ B[i][:, ::-1],B[i+1][:, 1:-1] ], axis=1) for i in [0, 2, 4] ]
    return zip( lum012, cent012 )


def seedLongitudinalSplines():
#    res= askItems([
#    ('curvedSection', True),
#    ['nPushedSections', 3], 
#    ['napprox', 60],     
#    ])
#    curvedsecs=res['curvedSection']
#    numpushed=res['nPushedSections']
#    splineapprox=res['napprox']

    curvedsecs=True
    numpushed=3
    splineapprox=60

    try:
        seeds = named('sat')
    except:
        warning("You need to set the longitudinal seeds first")
        return
    
    seededBif = seedLongSplines (named('long_splines'),seeds,  curvedSection=curvedsecs, nPushedSections=numpushed, napproxlong=splineapprox, napproxcut=splineapprox)
    export({'seededBif':seededBif})
    [ drawSeededBranch(seededBif[i][0], seededBif[i][1], propbranch=i+1) for i in range(3) ]


def meshBranch(HC,OC,nlong,ncirc,ntr,nbl):
    """Convenient function: from sections and centerlines to parametric volume mesh and outer surface mesh."""
    cpAin,cpAtr,cpAbl = vsm.cpAllSections(HC,OC,[True,True])
    hex_cp = [vsm.cpStackQ16toH64(i) for i in [cpAin,cpAtr,cpAbl] ]

    in_block = vsm.structuredHexMeshGrid(nlong,ncirc,ncirc,isophex='hex64')
    tr_block = vsm.structuredHexMeshGrid(nlong,ncirc,ntr,isophex='hex64')
    bl_block = vsm.structuredHexMeshGrid(nlong,ncirc,nbl,isophex='hex64')
    in_mesh,tr_mesh,bl_mesh = [vsm.mapHexLong(v_block,v_cp) for v_block,v_cp in zip([in_block,tr_block,bl_block],hex_cp)]
    M = [m[0][:,m[1]].reshape(-1,8,3) for m in [in_mesh,tr_mesh,bl_mesh] ]
    M = [Formex(m).toMesh() for m in M]
    #M=[correctHexMeshOrientation(m) for m in M ]
    #crate quad mesh on the external surface
    nq,eq = vsm.structuredQuadMeshGrid(nlong,ncirc)
    nq = Coords(column_stack([nq,ones([len(nq)])]) )
    gnq,eq = vsm.mapHexLong([nq,eq,bl_block[2] ],hex_cp[2])#group of nodes 
    xsurf = gnq[:, eq].reshape(-1,4,3)
    return M,xsurf


def getMeshingParameters():
    try:
        res = named('mesh_block_params')
    except:
        res = {
            'n_longit': 2,
            'n_circum': 3,
            's_radial': '[0.0, 0.6, 1.0]',
            's_boundary': '[0.0,0.4, 0.8,1.0]'
            }
    return res


def inputMeshingParameters():
    """Dialog for input of the meshing parameters.

    """
    dialog = Dialog(
        caption='Meshing parameters',store=getMeshingParameters(),
        items = [
            _I('n_longit',min=1,max=16,tooltip="Number of hex elements in longitudinal direction of a block"),
            _I('n_circum',min=1,max=16,tooltip="Number of hex elements over the circumference of a 1/4 section"),
            _I('s_radial',tooltip="Number of hex elements radially from inner pattern to boundary layer. It can be an integer or a list of seeds in the range 0.0 to 1.0"),
            _I('s_boundary',tooltip="Number of hex elements radially in the boundary layer. It can be an integer or a list of seeds in the range 0.0 to 1.0"),
            ]
        )
    res = dialog.getResult()
    if res:
        export({'mesh_block_params':res})


def sweepingMesher():
    """Sweeping hexahedral bifurcation mesher

    Creates a hexahedral mesh inside the branches of the bifurcation.
    Currently only the lumen mesh is included in this GPL3 version.
    """
    # Get the domain
    dialog = Dialog(
        caption='Domain selection',
        items=[
            _I('domain','Lumen','radio',choices=['Lumen', ]),
            ]
        )
    res = dialog.getResult()
    if not res:
        return
    domain = res['domain']
    print("Meshing %s" % domain)

    # Get the meshing parameters
    res = getMeshingParameters()
    longp,longc = res['n_longit'], res['n_circum']
    longr,longbl = eval(res['s_radial']), eval(res['s_boundary'])

    # Create the meshes
    pf.GUI.setBusy(True)
    Vmesh, Smesh=[], []
    bifbranches = named('seededBif')
    for branch in bifbranches:
        vmesh, smesh=meshBranch(branch[0], branch[1], longp, longc, longr, longbl)
        Vmesh.extend(vmesh)
        Smesh.extend(smesh)
    pf.GUI.setBusy(False)

    if domain == 'Lumen':
        ## m=[]
        ## [ m.extend(i) for i in Vmesh ]
        ## n, e=mergeMeshes(m)
        M = mergedModel(Vmesh)
        export({'CFD_lumen_model':M})
        export({'inner_surface_mesh':Smesh})
        clear()
        drawLumenMesh()

    if domain == 'Wall':
        export({'outer_surface_mesh':Smesh})
        clear()
        [draw(Formex(smesh), linewidth=3, color='red') for smesh in named('outer_surface_mesh') ]


def surfMesh():
    spc = getData('cross_splines')
    Fspc = [[ci.toFormex() for ci in c] for c in spc]
    surf = [ [ connect([Fi,Fi,Fj,Fj],nodid=[0,1,1,0]) for Fi,Fj in zip(f[:-1],f[1:]) ] for f in Fspc ]
    draw(surf)
    export({'surface_mesh':surf})


################## Create center line ##########################

def divideControlLines():
    """Divide the control lines."""
    br,slice_data = getData('branch','slice_data')

    npcent = slice_data['nslice']
    cl = [ br[i].approx(ntot=npcent[i//2]) for i in range(len(br))]

    export({'control_lines':cl})
    if drawOption('visual'):
        drawControlLines()


def center2D(x,x0,x1):
    """Find the center of the section x between the points x0 and x1"""
    x[:,2] = 0. # clear the z-coordinates
    i = x.distanceFromPoint(x0).argmin()
    j = x.distanceFromPoint(x1).argmin()
    xm = 0.5*(x[i]+x[j])
    return xm


def centerline2D(S,s0,s1):
    """Find the centerline of a tubular surface.

    - `S`: a tubular surface
    - `s0`,`s1`: helper polylines on opposing sides of the surface.

    """
    sections = slicer(S,s0,s1,visual=False)
    ## if drawOption('visual'):
    ##     draw(sections,color='red', alpha=1, flat=True)
    draw(sections,color='black', alpha=1, flat=True)
    cl = PolyLine([center2D(s.coords,x0,x1) for s,x0,x1 in zip(sections,s0.coords,s1.coords)])
    return cl


def extendedCenterline(cl,s0,s1,cp):
    """Extend the center line to the central point.

    cl: center line
    s0,s1: helper lines
    cp: central point.

    Each center line gets two extra points at the start:
    the central point and a point on the bisectrix at half distance.
    """
    vb = vecbisectrix(s0.coords[0],s1.coords[0],cp)
    d0 = cl.coords[0].distanceFromPoint(cp)
    return PolyLine(Coords.concatenate([cp,cp+vb*d0/2.,cl.coords]))


def centerlines():
    S,cp = getData('surface','central_point')
    try:
        hl = named('control_lines')
    except:
        divideControlLines()
        hl = named('control_lines')

    pf.GUI.setBusy(True)
    cl = [ centerline2D(S,hl[2*i],hl[2*i+1]) for i in range(3) ]
    cl = [ extendedCenterline(cl[i],hl[2*i],hl[2*i+1],cp) for i in range(3) ]
    pf.GUI.setBusy(False)

    export({'center_lines':cl})
    #if drawOption('visual'):
    drawCenterLines()

_draw_options = [
    ['visual',False],
    ['numbers',False],
    ['fill_cross',False],
    ['fill_surf',False],
    ]
try:
    updateData(_draw_options,named('draw_options'))
except:
    export({'draw_options':dict(_draw_options)})


def inputDrawOptions():
    res = askItems(_draw_options)
    if res:
        export({'draw_options':res})

_slice_data = [
    ['nslice', [10, 10, 10]],
    ]
try:
    updateData(_slice_data,named('slice_data'))
except:
    export({'slice_data':dict(_slice_data)})


def inputSlicingParameters():
    """Input the slicing parameters"""
    
    res = askItems(_slice_data, caption='long cuts')

    if res:
        export({'slice_data':res})


def createBranches(branch):
    """Create the branch control lines for the input lines."""
    # roll the first part to the end
    branch = branch[1:]+branch[:1]

    # reverse the uneven branches
    for i in range(1,6,2):
        branch[i] = branch[i].reverse()
    
    print("Branch control lines:")
    for i in range(6):
        print(" %s, %s" % divmod(i,2))
        print(branch[i].coords)
    export({'branch':branch})
    drawHelperLines()


def inputControlLines():
    """Enter three polyline paths in counterclockwise direction."""
    branch = []
    BA = []
    perspective(False)
    for i in range(6):
        pf.message("""..

** Input Branch %s **
""" % i)
        if i % 2 == 0:
            coords = None
        else:
            coords = branch[i-1].coords[-1:]

        # draw curve
        obj_params.update([('curl',0.),('closed',False)])
        points = drawPoints2D('curve',npoints=-1,coords=coords,zvalue=0.)
        obj = PolyLine(points)
        
        ###why this line does not work anymore?
        #obj = drawObject2D(mode='polyline',npoints=-1,coords=coords,zvalue=0.)

        obj.specular = 0.
        #pf.canvas.removeHighlights()
        if obj is not None:
            BA.append(draw(obj,color='blue',flat=True))
            branch.append(obj)
        else:
            break
        
    if len(branch) == 6:
        undraw(BA)
        createBranches(branch)
    else:
        warning("Incorrect definition of helper lines")

####################### DRAWING ######################################

color_half_branch = ['red','cyan','green','magenta','blue','yellow']

def drawSurface():
    S = named('surface')
    draw(S,color='red', alpha=0.3)

def drawHelperLines():
     branch = named('branch')
     for i in range(3):
        draw(branch[2*i:2*i+2],color=['red','green','blue'][i],flat=True, alpha=1, linewidth=3)

def drawControlLines():
    hl = named('control_lines')
    draw(hl,color='red',flat=True, alpha=1, linewidth=3)
    if drawOption('numbers'):
        [drawNumbers(h.coords) for h in hl]

def drawCentralPoint():
    cp = named('central_point')
    print("Central Point = %s" % cp)
    draw(cp,bbox='last',color='black', marksize=8,flat=True, alpha=1)

def drawCenterLines():
    cl = named('center_lines')
    draw(cl,color='blue',flat=True, alpha=1, linewidth=3)
    if drawOption('numbers'):
        [drawNumbers(li.coords) for li in cl]

def drawCrossSections():
    cs = named('cross_sections')
    #draw(cs[0:6:2],color='red')
    #draw(cs[1:6:2],color='blue')
    draw(Mesh.concatenate([i.toMesh() for i in olist.flatten(cs[0:6:2]) ]),color='red',flat=True, alpha=1, linewidth=3)
    draw(Mesh.concatenate([i.toMesh() for i in olist.flatten(cs[1:6:2]) ]),color='blue',flat=True, alpha=1, linewidth=3)
    
def drawSeededBranch(branchsections, branchcl, propbranch=0):
    [draw(PolyLine(sec, closed=True), flat=True, alpha=1, linewidth=3) for sec in branchsections]
    #draw(PolyLine(branchcl))
    #draw(Formex(branchcl).setProp(propbranch))
    #[draw(Formex(sec).setProp(propbranch)) for sec in branchsections]
#from plugins.objects import *
def drawOuterCrossSections():
    cs = named('cross_sections')
    draw(cs[0:6:2],color='red', linewidth=4,flat=True, alpha=1)
    draw(cs[1:6:2],color='blue', linewidth=4,flat=True, alpha=1)

def drawSurfaceMesh():
    [draw(Formex(smesh), linewidth=3, color='green') for smesh in named('inner_surface_mesh') ]
    
def drawLumenMesh():
    lumen_model=named('CFD_lumen_model')
    n, el=lumen_model.coords, lumen_model.elems
    [draw(Mesh(coords=n, elems=e).getBorderMesh().setProp(i+1)) for  i, e in enumerate(el)]


def drawCrossSplines():
    sp = getData('cross_splines')
    if drawOption('fill_cross'):
        [draw(Formex([si.coords for si in s]),color='black' ,flat=True, alpha=1) for s in sp] 
    else:
        [draw(s,color=c,flat=True, alpha=1) for s,c in zip(sp,color_half_branch)]
        if drawOption('numbers'):
            [[drawNumbers(si.coords) for si in s] for s in sp]



def drawLongSplines():
    sp = getData('long_splines')
    [draw(s,color=c,flat=True, alpha=1) for s,c in zip(sp,color_half_branch)]


def drawAxisSplines():
    sp = getData('axis_splines')
    draw(sp,color='black',flat=True, alpha=1)
    #drawNumbers( Formex( [i.coords[-1] for i in sp]  ).scale(1.1)  )

def drawSurfMesh():
    surf = getData('surface_mesh')
    draw(surf,color='black')


def drawCSys(ax=Formex([[[1., 0., 0.]],[[0., 1., 0.]],[[0., 0., 1.]], [[0., 0., 0.]]]), color='black'):
    """it draws the coordinate system with origin in ax[0] and directions determined by the 3 points in ax[1:4]"""
    assex=array([ax[3], ax[0]])
    assey=array([ax[3], ax[1]])
    assez=array([ax[3], ax[2]])
    for asse in [assex, assey, assez]:draw(Formex(asse.reshape(1, 2, 3)), color=color)
    drawNumbers(Formex([assex[1], assey[1],assez[1] ]))#

def flyThru():
    cl = named('axis_splines')
    for li in cl:
        flyAlong(li,upvector=[0.,0.,1.])

def drawAll():
    drawSurface()
    drawHelperLines()
    drawCentralPoint()
    drawCenterLines()


#############################################################################
######### Create a menu with interactive tasks #############

def nextStep(msg):
    global stepwise
    if stepwise:
        ans = ask(msg,['Quit','Continue','Step'],align='--')
        if ans == 'Continue':
            stepwise = False
        return ans != 'Quit'
    else:
        return True


def example():
    global stepwise
    stepwise = True

    clear()
    wireframe()
    view('front')
    if not nextStep("This example guides you through the subsequent steps to create a hexahedral mesh in a bifurcation. At each step you can opt to execute a single step, continue the whole procedure, or quit the example.\n\n1. Input the bifurcation surface model"):
        return
    
    examplefile = os.path.join(getcfg('datadir'),'bifurcation.off')
    print(examplefile)
    export({'surface':TriSurface.read(examplefile)})
    drawSurface()
    
    if not nextStep('2. Create the central point of the bifurcation'):
        return
    cp = Coords([-1.92753589,  0.94010758, -0.1379855])
    export({'central_point': cp})
    smooth()
    transparent(True)
    drawCentralPoint()

    if not nextStep('3. Create the helper lines for the mesher. This step is best done with perspective off.'):
        return
    setDrawOptions({'bbox':'last'})
    perspective(False)
    C = [[-33.93232346,   7.50834751,   0.        ],
         [ -1.96555257,   6.90520096,   0.        ],
         [ 19.08426476,  10.4637661 ,   0.        ],
         [ 19.14457893,   1.2959373 ,   0.        ],
         [  2.61836171,   0.87373471,   0.        ],
         [ 19.08426476,  -1.59916639,   0.        ],
         [ 19.02395058, -12.6970644 ,   0.        ],
         [ -1.84492326,  -6.30370998,   0.        ],
         [-34.29421234,  -4.61489964,   0.        ]]
    C = Coords(C)
    drawNumbers(C)
    C = C.reshape(3,3,3)
    branch = []
    for i in range(3):
        for j in range(2):
            branch.append(PolyLine(C[i,j:j+2]))
    createBranches(branch)

    if not nextStep('Notice the order of the input points!\n\n4. Create the Center Lines'):
        return
    centerlines()

    if not nextStep('5. Slice the bifurcation'):
        return
    sliceIt()

    if not nextStep('6. Create Spline Mesh'):
        return
    perspective(True)
    splineIt()

    if not nextStep('7. Seed the longitudinal splines'):
        return
    inputLongitudinalSeeds()
    seedLongitudinalSplines()

    if not nextStep('8. Run the sweeping hex mesher'):
        return
    sweepingMesher()
    setDrawOptions({'bbox':'auto'})


def updateData(data,newdata):
    """Update the input data fields with new data values"""
    if newdata:
        for d in data:
            v = newdata.get(d[0],None)
            if v is not None:
                d[1] = v



def resetData():
    pass


def reset():
    clear()
    smooth()
    transparent(False)
    lights(True)
    setDrawOptions({'bbox':'last'})
    linewidth(2)


def deleteAll():
    resetData()
    reset()

_menu_ = 'BifMesh'

def create_menu():
    """Create the %s menu.""" % _menu_
    MenuData = [
        ("&Run through example",example),
        ("---",None),
        ("&1.  Import Bifurcation Geometry",importGeometry),
        ("&2.  Input Central Point",inputCentralPoint),
        ("&3.  Input Helper Lines",inputControlLines),
        ("&4.  Create Center Lines",centerlines),
        ("&5a. Input Slicing Parameters",inputSlicingParameters),
        ("&5b. Slice the bifurcation",sliceIt),
        ("&6.  Create Spline Mesh",splineIt),
        ("&7a. Input Longitudinal Seeds",inputLongitudinalSeeds), 
        ("&7b. Seed Longitudinal Splines",seedLongitudinalSplines),
        ("&8a. Input Meshing Parameters",inputMeshingParameters), 
        ("&8b. Sweeping Mesher",sweepingMesher), 
        ("---",None),
        ("&Create Surface Mesh",surfMesh),
        ("---",None),
        ("&Draw", [
            ("&All",drawAll),
            ("&Surface",drawSurface),
            ("&Helper Lines",drawHelperLines),
            ("&Control Lines",drawControlLines),
            ("&Central Point",drawCentralPoint),
            ("&Center Lines2D",drawCenterLines),
            ("&Center Lines3D",drawAxisSplines), 
            ("&Cross Sections",drawCrossSections),
            ("&Cross Splines",drawCrossSplines),
            ("&Long Splines",drawLongSplines),
            ("&SurfaceMesh",drawSurfaceMesh),
            ("&Surface Spline_Mesh",drawSurfMesh),
            ("Set Draw Options",inputDrawOptions),
            ]),
        ("---",None),
        ("&Fly Along Center Lines",flyThru),
        ("---",None),
        ("&Close Menu",close_menu),
        ]
    return menu.Menu(_menu_,items=MenuData,parent=pf.GUI.menu,before='help')


def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item(_menu_):
        create_menu()

def close_menu():
    """Close the menu."""
    m = pf.GUI.menu.item(_menu_)
    if m :
        m.remove()

def reload_menu():
    """Reload the menu."""
    from plugins import refresh
    close_menu()
    #refresh(_menu_)
    show_menu()


####################################################################
######### What to do when the script is executed ###################

def run():
    resetGUI()
    resetData()
    reset()
    reload_menu()

if __name__ == "draw":
    run()

# End




