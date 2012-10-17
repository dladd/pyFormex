# $Id$
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

"""surface_menu.py

Surface operations plugin menu for pyFormex.
"""
from __future__ import print_function

import pyformex as pf
from gui import actors,colors,decors,widgets,menu
from gui.colorscale import ColorScale,ColorLegend
from gui.draw import *
from plugins.trisurface import *
from plugins.objects import *
from plugins import plot2d,formex_menu,fe_abq
import simple
from plugins.tools import Plane
from pyformex.arraytools import niceLogSize

from gui.widgets import simpleInputItem as _I, groupInputItem as _G
import os, timer

_name_ = '_surface_menu_'
##################### selection and annotations ##########################


def draw_edge_numbers(n):
    """Draw the edge numbers of the named surface."""
    S = named(n)
    F = Formex(S.coords[S.getEdges()]) 
    return drawNumbers(F,color='green')

def draw_normals(n,avg=False):
    """Draw the surface normals at centers or averaged normals at the nodes."""
    S = named(n)
    if avg:
        C = S.coords
        N = S.avgVertexNormals()
    else:
        C = S.centroids()
        A,N = S.areaNormals()
    siz = pf.cfg['draw/normalsize']
    if siz == 'area' and not avg:
        siz = sqrt(A).reshape(-1,1)
    else:
        try:
            siz = float(siz)
        except:
            siz = 0.05 * C.dsize()
    if avg:
        color = 'orange'
    else:
        color = 'red'
    return drawVectors(C,N,size=siz,color=color,wait=False)

def draw_avg_normals(n):
    return draw_normals(n,True)


class SurfaceObjects(DrawableObjects):
    def __init__(self):
        DrawableObjects.__init__(self,clas=TriSurface)
    def toggleEdgeNumbers(self,onoff=None):
        self.toggleAnnotation(draw_edge_numbers,onoff)
    def toggleNormals(self,onoff=None):
        self.toggleAnnotation(draw_normals,onoff)
    def toggleAvgNormals(self,onoff=None):
        self.toggleAnnotation(draw_avg_normals,onoff)

selection = SurfaceObjects()


##################### select, read and write ##########################

def read_Surface(fn,exportName=True):
    pf.message("Reading file %s" % fn)
    t = timer.Timer()
    S = TriSurface.read(fn)
    pf.message("Read surface with %d vertices, %d edges, %d triangles in %s seconds" % (S.ncoords(),S.nedges(),S.nelems(),t.seconds()))
    if exportName:
        name = utils.projectName(fn)
        export({name:S})
        selection.set([name])
    return S


def readSelection(select=True,draw=True,multi=True):
    """Read a Surface (or list) from asked file name(s).

    If select is True (default), this becomes the current selection.
    If select and draw are True (default), the selection is drawn.
    """
    types = map(utils.fileDescription,['surface','all'])
    fn = askFilename(pf.cfg['workdir'],types,multi=multi)
    if fn:
        if not multi:
            fn = [ fn ]
        chdir(fn[0])
        names = map(utils.projectName,fn)
        pf.GUI.setBusy()
        surfaces = [ read_Surface(f,False) for f in fn ]
        for i,S in enumerate(surfaces):
            S.setProp(i)
        pf.GUI.setBusy(False)
        export(dict(zip(names,surfaces)))
        if select:
            pf.message("Set selection to %s" % str(names))
            selection.set(names)
            if draw:
                if max([named(s).nfaces() for s in selection]) < 100000 or ack("""
This is a large model and drawing could take quite some time.
You should consider coarsening the model before drawing it.
Shall I proceed with the drawing now?
"""):
                    selection.draw()
    return fn
    

def printSize():
    for s in selection.names:
        S = named(s)
        pf.message("Surface %s has %d vertices, %s edges and %d faces" %
                   (s,S.ncoords(),S.nedges(),S.nelems()))

def printType():
    for s in selection.names:
        S = named(s)
        if S.isClosedManifold():
            pf.message("Surface %s is a closed manifold" % s)
        elif S.isManifold():
            pf.message("Surface %s is an open manifold" % s)
        else:
            pf.message("Surface %s is not a manifold" % s)

def printArea():
    for s in selection.names:
        S = named(s)
        pf.message("Surface %s has area %s" % (s,S.area()))

def printVolume():
    for s in selection.names:
        S = named(s)
        pf.message("Surface %s has volume %s" % (s,S.volume()))


def printStats():
    for s in selection.names:
        S = named(s)
        pf.message("Statistics for surface %s" % s)
        print(S.stats())

def toFormex(suffix=''):
    """Transform the selection to Formices.

    If a suffix is given, the Formices are stored with names equal to the
    surface names plus the suffix, else, the surface names will be used
    (and the surfaces will thus be cleared from memory).
    """
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    newnames = selection.names
    if suffix:
        newnames = [ n + suffix for n in newnames ]

    newvalues = [ named(n).toFormex() for n in newnames ]
    export2(newnames,newvalues)

    if not suffix:
        selection.clear()
    formex_menu.selection.set(newnames)
    clear()
    formex_menu.selection.draw()
    

def fromFormex(suffix=''):
    """Transform the Formex selection to TriSurfaces.

    If a suffix is given, the TriSurfaces are stored with names equal to the
    Formex names plus the suffix, else, the Formex names will be used
    (and the Formices will thus be cleared from memory).
    """
    if not formex_menu.selection.check():
        formex_menu.selection.ask()

    if not formex_menu.selection.names:
        return

    names = formex_menu.selection.names
    formices = [ named(n) for n in names ]
    if suffix:
        names = [ n + suffix for n in names ]

    t = timer.Timer()
    surfaces =  dict([ (n,TriSurface(F)) for n,F in zip(names,formices) if F.nplex() == 3])
    print("Converted in %s seconds" % t.seconds())
    print(surfaces.keys())
    export(surfaces)

    if not suffix:
        formex_menu.selection.clear()
    selection.set(surfaces.keys())


def toMesh(suffix=''):
    """Transform the selection to Meshes.

    If a suffix is given, the Meshes are stored with names equal to the
    surface names plus the suffix, else, the surface names will be used
    (and the surfaces will thus be cleared from memory).
    """
    from plugins import geometry_menu
    if not selection.check():
        selection.ask()

    if not selection.names:
        return

    newnames = selection.names
    if suffix:
        newnames = [ n + suffix for n in newnames ]

    newvalues = [ named(n).toMesh() for n in newnames ]
    export2(newnames,newvalues)

    if not suffix:
        selection.clear()
    geometry_menu.selection.set(newnames)
    clear()
    geometry_menu.selection.draw()
    

def fromMesh(suffix=''):
    """Transform the Mesh selection to TriSurfaces.

    If a suffix is given, the TriSurfaces are stored with names equal to the
    Mesh names plus the suffix, else, the Mesh names will be used
    (and the Meshes will thus be cleared from memory).
    """
    from plugins import geometry_menu
    if not geometry_menu.selection.check():
        geometry_menu.selection.ask()

    if not geometry_menu.selection.names:
        return

    names = geometry_menu.selection.names
    meshes = [ named(n) for n in names ]
    if suffix:
        names = [ n + suffix for n in names ]

    t = timer.Timer()
    surfaces =  dict([ (n,TriSurface(M)) for n,M in zip(names,meshes) if M.eltype == 'tri3'])
    print("Converted in %s seconds" % t.seconds())
    print(surfaces.keys())
    export(surfaces)

    if not suffix:
        geometry_menu.selection.clear()
    selection.set(surfaces.keys())


def toggle_shrink():
    """Toggle the shrink mode"""
    if selection.shrink is None:
        selection.shrink = 0.8
    else:
        selection.shrink = None
    selection.draw()


def toggle_auto_draw():
    global autodraw
    autodraw = not autodraw


def fixNormals():
    """Fix the normals of the selected surfaces."""
    SL = selection.check()
    if SL:
        SL = [ S.fixNormals() for S in SL ]
        export2(selection.names,SL)
        selection.draw()


def reverseNormals():
    """Reverse the normals of the selected surfaces."""
    SL = selection.check()
    if SL:
        SL = [ S.reverse() for S in SL ]
        export2(selection.names,SL)
        selection.draw()


def merge():
    """Merge the selected surfaces."""
    SL = selection.check(warn=False)
    if len(SL) < 2:
        warning("You should at least select two surfaces!")
        return

    S = TriSurface.concatenate(SL)
    name = '--merged-surface--'
    export({name:S})
    selection.set(name)
    selection.draw()


def write_surface(types=['surface','gts','stl','off','neu','smesh']):
    F = selection.check(single=True)
    if F:
        if type(types) == str:
            types = [ types ]
        types = map(utils.fileDescription,types)
        fn = askNewFilename(pf.cfg['workdir'],types)
        if fn:
            pf.message("Exporting surface model to %s" % fn)
            pf.GUI.setBusy()
            F.write(fn)   
            pf.GUI.setBusy(False)

#
# Operations with surface type, border, ...
#
def showBorder():
    S = selection.check(single=True)
    if S:
        border = S.border()
        if border:
            print("The border consists of %s parts" % len(border))
            print("The sorted border edges are: ")
            print('\n'.join([" %s: %s" % (i,b.elems) for i,b in enumerate(border)]))
            coloredB = [ b.compact().setProp(i+1) for i,b in enumerate(border) ]
            draw(coloredB,linewidth=3)
            for i,b in enumerate(coloredB):
                c = roll(pf.canvas.settings.colormap,i+1,axis=0)
                drawText3D(b.center(),str(i),color=c,font='sans',size=18,ontop=True)
            export({'border':coloredB})
        else:
            warning("The surface %s does not have a border" % selection[0])
            forget('border')
    return S


def fillBorders():
    _data_ = _name_+'fillBorders_data'
    S = showBorder()
    try:
        B = named('border')
    except:
        return
    if B:
        props = [ b.prop[0] for b in B ]
        dia = Dialog([
            _I('Fill which borders',itemtype='radio',choices=['All','One']),
            _I('Filling method',itemtype='radio',choices=['radial','border']),
            _I('merge',False,text='Merge fills into current surface'),
            ])
        if _data_ in pf.PF:
            dia.updateData(pf.PF[_data_])
        res = dia.getResults()
        if res:
            pf.PF[_data_] = res
        
            if res['Fill which borders'] == 'One':
                B = B[:1]
            fills = [ fillBorder(b,method=res['Filling method']).setProp(i+1) for i,b in enumerate(B) ]
            if res['merge']:
                name = selection.names[0]
                S = named(name)
                for f in fills:
                    S += f
                #print "MERGE",type(S)
                export({name:S})
                selection.draw()
            else:
                draw(fills)
                export(dict([('fill-%s'%i,f) for i,f in enumerate(fills)]))

        
def deleteTriangles():
    S = selection.check(single=True)
    if S:
        picked = pick('element')
        #print picked
        if picked:
            picked = picked[0]
            #print picked
            if len(picked) > 0:
                #print S.nelems()
                S = S.cselect(picked)
                #print "DELETE",type(S)
                name = selection.names[0]
                #print name
                #print S.nelems()
                export({name:S})
                selection.draw()


# Selectable values for display/histogram
# Each key is a description of a result
# Each value consist of a tuple
#  - function to calculate the values
#  - domain to display: True to display on edges, False to display on elements

SelectableStatsValues = odict.ODict([
    ('Quality', (TriSurface.quality,False)),
    ('Aspect ratio', (TriSurface.aspectRatio,False)),
    ('Facet Area', (TriSurface.facetArea,False)),
    ('Facet Perimeter', (TriSurface.perimeters,False)),
    ('Smallest altitude', (TriSurface.smallestAltitude,False)),
    ('Longest edge', (TriSurface.longestEdge,False)),
    ('Shortest edge', (TriSurface.shortestEdge,False)),
    ('Number of node adjacent elements', (TriSurface.nNodeAdjacent,False)),
    ('Number of edge adjacent elements', (TriSurface.nEdgeAdjacent,False)),
    ('Edge angle', (TriSurface.edgeAngles,True)),
    ('Number of connected elements', (TriSurface.nEdgeConnected,True)),
    ('Curvature', (TriSurface.curvature,False)),
    ])

CurvatureValues = ['Gaussian curvature','Mean curvature','Shape index','Curvedness','First principal curvature','Second principal curvature']


def showHistogram(key,val,cumulative):
    y,x = plot2d.createHistogram(val,cumulative=cumulative)
    plot2d.showHistogram(x,y,key)


_stat_dia = None

def showStatistics(key=None,domain=True,dist=False,cumdist=False,clip=None,vmin=None,vmax=None,percentile=False):
    """Show the values corresponding with key in the specified mode.

    key is one of the keys of SelectableStatsValues
    mode is one of ['On Domain','Histogram','Cumulative Histogram']
    """
    S = selection.check(single=True)
    if S:
        func,onEdges = SelectableStatsValues[key]
        kargs = {}
        if key == 'Curvature':
            kargs['neighbours'] = _stat_dia.results['neighbours']
        val = func(S,**kargs)
        if key == 'Curvature':
            ind = CurvatureValues.index(_stat_dia.results['curval'])
            val = val[ind]
            val = val[S.elems]

        # !! THIS SHOULD BE IMPLEMENTED AS A GENERAL VALUE CLIPPER
        # !! e.g popping up when clicking the legend
        # !! and the values should be changeable 

        if clip:
            clip = clip.lower()
            if percentile:
                try:
                    from scipy.stats.stats import scoreatpercentile
                except:
                    warning("""..
                
**The **percentile** clipping option is not available.
Most likely because 'python-scipy' is not installed on your system.""")
                    return

                Q1 = scoreatpercentile(val,vmin)
                Q3 = scoreatpercentile(val,vmax)
                factor = 3
                if vmin:
                    vmin = Q1-factor*(Q3-Q1)
                if vmax:
                    vmax = Q3+factor*(Q3-Q1)
                
            if clip == 'top':
                val = val.clip(max=vmax)
            elif clip == 'bottom':
                val = val.clip(min=vmin)
            else:
                val = val.clip(vmin,vmax)

        if domain:
            clear()
            lights(False)
            showSurfaceValue(S,key,val,onEdges)
        if dist:
            showHistogram(key,val,cumulative=False)
        if cumdist:
            showHistogram(key,val,cumulative=True)


def _show_stats(domain,dist):
    _stat_dia.acceptData()
    res = _stat_dia.results
    key = res['Value']
    if dist and res['Cumulative Distribution']:
        cumdist = True
        dist = False
    else:
        cumdist = False
    clip = res['clip']
    if clip == 'None':
        clip = None
    percentile = res['Clip Mode'] != 'Range'
    minval = res['Bottom']
    maxval = res['Top']
    showStatistics(key,domain,dist,cumdist,clip=clip,vmin=minval,vmax=maxval,percentile=percentile)

def _show_domain():
    _show_stats(True,False)
def _show_dist():
    _show_stats(False,True)

def _close_stats_dia():
    global _stat_dia
    if _stat_dia:
        _stat_dia.close()
        _stat_dia = None

    
def showStatisticsDialog():
    global _stat_dia
    if _stat_dia:
        _close_stats_dia()
        
    dispmodes = ['On Domain','Histogram','Cumulative Histogram']
    keys = SelectableStatsValues.keys()
    _stat_dia = widgets.InputDialog(
        caption='Surface Statistics',items=[
            _I('Value',itemtype='vradio',choices=keys),
            _I('neighbours',text='Curvature Neighbourhood',value=1),
            _I('curval',text='Curvature Value',itemtype='vradio',choices=CurvatureValues),
            _I('clip',itemtype='hradio',choices=['None','Top','Bottom','Both']),
            _I('Clip Mode',itemtype='hradio',choices=['Range','Percentile']),
            _G('Clip Values',checkable=True,items=[
                _I('Top',1.0),
                _I('Bottom',0.0),
                ],
              ),
            _I('Cumulative Distribution',False),
            ],
        actions=[
            ('Close',_close_stats_dia),
            ('Distribution',_show_dist),
            ('Show on domain',_show_domain)],
        default='Show on domain'
        )
    _stat_dia.show()
    
            

def showSurfaceValue(S,txt,val,onEdges):
    val = nan_to_num(val)
    mi,ma = val.min(),val.max()
    # Test: replace min with max
    dec = min(abs(mi),abs(ma))
    if dec > 0.0:
        dec = max(0,3-int(log10(dec)))
    else:
        dec = 2
    # create a colorscale and draw the colorlegend
    CS = ColorScale('RAINBOW',mi,ma,0.5*(mi+ma),1.)
    cval = array(map(CS.color,ravel(val)))
    cval = cval.reshape(append(val.shape,cval.shape[-1]))
    if onEdges:
        F = Formex(S.coords[S.getEdges()])
        draw(F,color=cval)
    else:
        draw(S,color=cval)
    CL = ColorLegend(CS,100)
    CLA = decors.ColorLegend(CL,10,10,30,200,dec=dec) 
    pf.canvas.addDecoration(CLA)
    drawText(txt,10,230,font='hv18')


def colorByFront():
    S = selection.check(single=True)
    if S:
        res  = askItems([_I('front type',choices=['node','edge']),
                         _I('number of colors',-1),
                         _I('front width',1),
                         _I('start at',0),
                         _I('first prop',0),
                         ])
        pf.app.processEvents()
        if res:
            selection.remember()
            t = timer.Timer()
            ftype = res['front type']
            nwidth = res['front width']
            maxval = nwidth * res['number of colors']
            startat = res['start at']
            firstprop = res['first prop']
            if ftype == 'node':
                p = S.frontWalk(level=0,maxval=maxval,startat=startat)
            else:
                p = S.frontWalk(level=1,maxval=maxval,startat=startat)
            S.setProp(p/nwidth + firstprop)
            print("Colored in %s parts (%s seconds)" % (S.prop.max()+1,t.seconds()))
            selection.draw()


def partitionByConnection():
    S = selection.check(single=True)
    if S:
        selection.remember()
        t = timer.Timer()
        S.prop = S.partitionByConnection()
        print("Partitioned in %s parts (%s seconds)" % (S.prop.max()+1,t.seconds()))
        selection.draw()


def partitionByAngle():
    S = selection.check(single=True)
    if S:
        res  = askItems([_I('angle',60.),
                         _I('firstprop',1),
                         _I('sort by',choices=['number','area','none']),
#                         _I('old algorithm',False),
                         ])
        pf.app.processEvents()
        if res:
            selection.remember()
            t = timer.Timer()
            p = S.partitionByAngle(angle=res['angle'],sort=res['sort by'])#,alt=res['old algorithm'])
            S = S.setProp(p + res['firstprop'])
            print("Partitioned in %s parts (%s seconds)" % (len(S.propSet()),t.seconds()))
            for p in S.propSet():
                print(" p: %s; n: %s" % (p,(S.prop==p).sum()))
            selection.draw()
         
 
def showFeatureEdges():
    S = selection.check(single=True)
    if S:
        selection.draw()
        res  = askItems([
            _I('angle',60.),
            _I('ontop',False),
            ])
        pf.app.processEvents()
        if res:
            p = S.featureEdges(angle=res['angle'])
            M = Mesh(S.coords,S.edges[p])
            draw(M,color='red',linewidth=3,bbox='last',nolight=True,ontop=res['ontop'])
        

#############################################################################
# Transformation of the vertex coordinates (based on Coords)

#
# !! These functions could be made identical to those in Formex_menu
# !! (and thus could be unified) if the surface transfromations were not done
# !! inplace but returned a new surface instance instead.
#
            
def scaleSelection():
    """Scale the selection."""
    FL = selection.check()
    if FL:
        res = askItems([_I('scale',1.0),
                        ],caption = 'Scaling Factor')
        if res:
            scale = float(res['scale'])
            selection.remember(True)
            selection.changeValues([F.scale(scale) for F in FL])
            selection.drawChanges()

            
def scale3Selection():
    """Scale the selection with 3 scale values."""
    FL = selection.check()
    if FL:
        res = askItems([_I('scale',[1.0,1.0,1.0],itemtype='point'),
                        ],caption = 'Scaling Factors')
        if res:
            scale = res['scale']
            selection.remember(True)
            selection.changeValues([F.scale(scale) for F in FL])
            selection.drawChanges()


def translateSelection():
    """Translate the selection in axes direction."""
    FL = selection.check()
    modes = ['Axis direction','General direction']
    if FL:
        res = askItems(
            [   _I('mode',choices=modes),
                _I('axis',0),
                _I('direction',[1.,0.,0.],itemtype='point'),
                _I('distance',1.0),
                ],
            enablers=[
                ('mode',modes[0],'axis'),
                ('mode',modes[1],'direction'),
                ],
            caption = 'Translation Parameters',
            )
        if res:
            mode = res['mode']
            if mode[0] == 'A':
                dir = res['axis']
            else:
                dir = res['direction']
            dist = res['distance']
            selection.remember(True)
            selection.changeValues([F.translate(dir,dist) for F in FL])
            selection.drawChanges()


def centerSelection():
    """Center the selection."""
    FL = selection.check()
    if FL:
        selection.remember(True)
        selection.changeValues([F.translate(-F.coords.center()) for F in FL])
        selection.drawChanges()


def rotate(mode='global'):
    """Rotate the selection.

    mode is one of 'global','parallel','central','general'
    """
    FL = selection.check()
    if FL:
        if mode == 'global':
            res = askItems([_I('angle',90.0),
                            _I('axis',2),
                            ])
            if res:
                angle = res['angle']
                axis = res['axis']
                around = None
        elif mode == 'parallel':
            res = askItems([_I('angle',90.0),
                            _I('axis',2),
                            _I('point',[0.0,0.0,0.0],itemtype='point'),
                            ])
            if res:
                angle = res['angle']
                axis = res['axis']
                around = res['point']
        elif mode == 'central':
            res = askItems([_I('angle',90.0),
                            _I('axis',[1.0,0.0,0.0],itemtype='point'),
                            ])
            if res:
                angle = res['angle']
                axis = res['axis']
                around = None
        elif mode == 'general':
            res = askItems([_I('angle',90.0),
                            _I('axis',[1.0,0.0,0.0],itemtype='point'),
                            _I('point',[0.0,0.0,0.0],itemtype='point'),
                            ])
            if res:
                angle = res['angle']
                axis = res['axis']
                around = res['point']
        if res:
            selection.remember(True)
            selection.changeValues([F.rotate(angle,axis,around) for F in FL])
            selection.drawChanges()


def rotateGlobal():
    rotate('global')

def rotateParallel():
    rotate('parallel')

def rotateCentral():
    rotate('central')

def rotateGeneral():
    rotate('general')


def rollAxes():
    """Rotate the selection."""
    FL = selection.check()
    if FL:
        selection.remember(True)
        selection.changeValues([F.rollAxes() for F in FL])
        selection.drawChanges()


def clipSelection():
    """Clip the selection.
    
    The coords list is not changed.
    """
    FL = selection.check()
    if FL:
        res = askItems([_I('axis',0),
                        _I('begin',0.0),
                        _I('end',1.0),
                        _I('nodes','all',choices=['all','any','none']),
                        ],caption='Clipping Parameters')
        if res:
            bb = bbox(FL)
            axis = res['axis']
            xmi = bb[0][axis]
            xma = bb[1][axis]
            dx = xma-xmi
            xc1 = xmi + float(res['begin']) * dx
            xc2 = xmi + float(res['end']) * dx
            selection.changeValues([F.clip(F.test(nodes=res['nodes'],dir=axis,min=xc1,max=xc2)) for F in FL])
            selection.drawChanges()


def clipAtPlane():
    """Clip the selection with a plane."""
    FL = selection.check()
    if not FL:
        return
    
    dsize = bbox(FL).dsize()
    esize = 10 ** (niceLogSize(dsize)-5)

    res = askItems([_I('Point',[0.0,0.0,0.0],itemtype='point'),
                    _I('Normal',[1.0,0.0,0.0],itemtype='point'),
                    _I('Keep side',itemtype='radio',choices=['positive','negative']),
                    _I('Nodes',itemtype='radio',choices=['all','any','none']),
                    _I('Tolerance',esize),
                    _I('Property',1),
                    ],caption = 'Define the clipping plane')
    if res:
        P = res['Point']
        N = res['Normal']
        side = res['Keep side']
        nodes = res['Nodes']
        atol = res['Tolerance']
        prop = res['Property']
        selection.remember(True)
        if side == 'positive':
            func = TriSurface.clip
        else:
            func = TriSurface.cclip
        FL = [ func(F,F.test(nodes=nodes,dir=N,min=P,atol=atol)) for F in FL]
        FL = [ F.setProp(prop) for F in FL ]
        export(dict([('%s/clip' % n,F) for n,F in zip(selection,FL)]))
        selection.set(['%s/clip' % n for n in selection])
        selection.draw()


def cutWithPlane():
    """Cut the selection with a plane."""
    FL = selection.check()
    if not FL:
        return
    
    dsize = bbox(FL).dsize()
    esize = 10 ** (niceLogSize(dsize)-5)

    res = askItems([_I('Point',[0.0,0.0,0.0],itemtype='point'),
                    _I('Normal',[1.0,0.0,0.0],itemtype='point'),
                    _I('New props',[1,2,2,3,4,5,6]),
                    _I('Side','positive',itemtype='radio',choices=['positive','negative','both']),
                    _I('Tolerance',esize),
                    ],caption = 'Define the cutting plane')
    if res:
        P = res['Point']
        N = res['Normal']
        p = res['New props']
        side = res['Side']
        atol = res['Tolerance']
        selection.remember(True)
        if side == 'both':
            G = [F.toFormex().cutWithPlane(P,N,side=side,atol=atol,newprops=p) for F in FL]
            G_pos = []
            G_neg  =[]
            for F in G:
                G_pos.append(TriSurface(F[0]))
                G_neg.append(TriSurface(F[1]))
            export(dict([('%s/pos' % n,g) for n,g in zip(selection,G_pos)]))
            export(dict([('%s/neg' % n,g) for n,g in zip(selection,G_neg)]))
            selection.set(['%s/pos' % n for n in selection] + ['%s/neg' % n for n in selection])
            selection.draw()
        else:
            selection.changeValues([F.cutWithPlane(P,N,newprops=p,side=side,atol=atol) for F in FL])
            selection.drawChanges()


def cutSelectionByPlanes():
    """Cut the selection with one or more planes, which are already created."""
    S = selection.check(single=True)
    if not S:
        return

    planes = listAll(clas=Plane)
    if len(planes) == 0:
        warning("You have to define some planes first.")
        return
        
    res1 = widgets.ListSelection(planes,caption='Known %sobjects' % selection.object_type(),sort=True).getResult()
    if res1:
        res2 = askItems([_I('Tolerance',0.),
                         _I('Color by','side',itemtype='radio',choices=['side', 'element type']), 
                         _I('Side','both',itemtype='radio',choices=['positive','negative','both']),
                         ],caption = 'Cutting parameters')
        if res2:
            planes = map(named, res1)
            p = [plane.P for plane in planes]
            n = [plane.n for plane in planes]
            atol = res2['Tolerance']
            color = res2['Color by']
            side = res2['Side']
            if color == 'element type':
                newprops = [1,2,2,3,4,5,6]
            else:
                newprops = None
            if side == 'both':
                Spos, Sneg = S.toFormex().cutWithPlane(p,n,newprops=newprops,side=side,atol=atol)
            elif side == 'positive':
                Spos = S.toFormex().cutWithPlane(p,n,newprops=newprops,side=side,atol=atol)
                Sneg = Formex()
            elif side == 'negative':
                Sneg = S.toFormex().cutWithPlane(p,n,newprops=newprops,side=side,atol=atol)
                Spos = Formex()
            if Spos.nelems() !=0:
                Spos = TriSurface(Spos)
                if color == 'side':
                    Spos.setProp(2)
            else:
                Spos = None
            if Sneg.nelems() != 0:
                Sneg = TriSurface(Sneg)
                if color == 'side':
                    Sneg.setProp(3)
            else:
                Sneg = None
            name = selection.names[0]
            export({name+"/pos":Spos})
            export({name+"/neg":Sneg})
            selection.set([name+"/pos",name+"/neg"]+res1)
            selection.draw()


def intersectWithPlane():
    """Intersect the selection with a plane."""
    FL = selection.check()
    if not FL:
        return
    res = askItems([_I('Name suffix','intersect'),
                    _I('Point',(0.0,0.0,0.0)),
                    _I('Normal',(1.0,0.0,0.0)),
                    ],caption = 'Define the cutting plane')
    if res:
        suffix = res['Name suffix']
        P = res['Point']
        N = res['Normal']
        M = [ S.intersectionWithPlane(P,N) for S in FL ]
        draw(M,color='red')
        export(dict([('%s/%s' % (n,suffix), m) for (n,m) in zip(selection,M)]))
            

def slicer():
    """Slice the surface to a sequence of cross sections."""
    S = selection.check(single=True)
    if not S:
        return
    res = askItems([_I('Direction',[1.,0.,0.]),
                    _I('# slices',20),
                    ],caption = 'Define the slicing planes')
    if res:
        axis = res['Direction']
        nslices = res['# slices']
        pf.GUI.setBusy(True)
        t = timer.Timer()
        slices = S.slice(dir=axis,nplanes=nslices)
        print("Sliced in %s seconds" % t.seconds())
        pf.GUI.setBusy(False)
        print([ s.nelems() for s in slices ])
        draw([ s for s in slices if s.nelems() > 0],color='red',bbox='last',view=None)
        export({'%s/slices' % selection[0]:slices}) 


def spliner():
    """Slice the surface to a sequence of cross sections."""
    import olist
    from plugins.curve import BezierSpline
    S = selection.check(single=True)
    if not S:
        return
    res = askItems([_I('Direction',[1.,0.,0.]),
                    _I('# slices',20),
                    _I('remove_invalid',False),
                    ],caption = 'Define the slicing planes')
    if res:
        axis = res['Direction']
        nslices = res['# slices']
        remove_cruft = res['remove_invalid']
        pf.GUI.setBusy(True)
        slices = S.slice(dir=axis,nplanes=nslices)
        pf.GUI.setBusy(False)
        print([ s.nelems() for s in slices ])
        split = [ s.splitProp().values() for s in slices if s.nelems() > 0 ]
        split = olist.flatten(split)
        hasnan = [ isnan(s.coords).any() for s in split ]
        print(hasnan)
        print(sum(hasnan))
        #print [s.closed for s in split]
        export({'%s/split' % selection[0]:split}) 
        draw(split,color='blue',bbox='last',view=None)
        splines = [ BezierSpline(s.coords[s.elems[:,0]],closed=True) for s in split ]
        draw(splines,color='red',bbox='last',view=None)
        export({'%s/splines' % selection[0]:splines}) 


##################  Smooth the selected surface #############################

def smooth():
    """Smooth the selected surface."""
    S = selection.check(single=True)
    if S:
        res = askItems(
            [ _I('method','lowpass',itemtype='select',choices=['lowpass','laplace']),
              _I('iterations',1,min=1),
              _I('lambda_value',0.5,min=0.0,max=1.0),
              _I('neighbourhood',1),
              _I('alpha',0.0),
              _I('beta',0.2),
              ], enablers=[
                ('method','lowpass','neighbourhood'),
                ('method','laplace','alpha','beta'),
                ],
            )
        if res:
            if not 0.0 <= res['lambda_value'] <= 1.0:
                warning("Lambda should be between 0 and 1.")
                return
            selection.remember(True)
            S = S.smooth(**res)
            selection.changeValues([S])
            selection.drawChanges()


def refine():
    S = selection.check(single=True)
    if S:
        res = askItems([_I('max_edges',-1),
                        _I('min_cost',-1.0),
                        ])
        if res:
            selection.remember()
            if res['max_edges'] <= 0:
                res['max_edges'] = None
            if res['min_cost'] <= 0:
                res['min_cost'] = None
            S=S.refine(**res)
            selection.changeValues([S])
            selection.drawChanges()


###################################################################
########### The following functions are in need of a make-over



def flytru_stl():
    """Fly through the stl model."""
    global ctr
    Fc = Formex(array(ctr).reshape((-1,1,3)))
    path = connect([Fc,Fc],bias=[0,1])
    flyAlong(path)


def create_volume():
    """Generate a volume tetraeder mesh inside a surface."""
    S = selection.check(single=True)
    if S:
        import tetgen
        M = tetgen.meshInsideSurface(S,quality=True)
        export({'tetmesh':M})
        

def export_surface():
    S = selection.check(single=True)
    if S:
        types = [ "Abaqus INP files (*.inp)" ]
        fn = askNewFilename(pf.cfg['workdir'],types)
        if fn:
            print("Exporting surface model to %s" % fn)
            updateGUI()
            fe_abq.exportMesh(fn,S,eltype='S3',header="Abaqus model generated by pyFormex from input file %s" % os.path.basename(fn))


def export_volume():
    if PF['volume'] is None:
        return
    types = [ "Abaqus INP files (*.inp)" ]
    fn = askNewFilename(pf.cfg['workdir'],types)
    if fn:
        print("Exporting volume model to %s" % fn)
        updateGUI()
        mesh = Mesh(PF['volume'])
        fe_abq.exportMesh(fn,mesh,eltype='C3D%d' % elems.shape[1],header="Abaqus model generated by tetgen from surface in STL file %s.stl" % PF['project'])


def show_nodes():
    n = 0
    data = askItems([_I('node number',n),
                     ])
    n = int(data['node number'])
    if n > 0:
        nodes,elems = PF['surface']
        print("Node %s = %s",(n,nodes[n]))


def trim_border(elems,nodes,nb,visual=False):
    """Removes the triangles with nb or more border edges.

    Returns an array with the remaining elements.
    """
    b = border(elems)
    b = b.sum(axis=1)
    trim = where(b>=nb)[0]
    keep = where(b<nb)[0]
    nelems = elems.shape[0]
    ntrim = trim.shape[0]
    nkeep = keep.shape[0]
    print("Selected %s of %s elements, leaving %s" % (ntrim,nelems,nkeep) )

    if visual and ntrim > 0:
        prop = zeros(shape=(F.nelems(),),dtype=int32)
        prop[trim] = 2 # red
        prop[keep] = 1 # yellow
        F = Formex(nodes[elems],prop)
        clear()
        draw(F,view='left')
        
    return elems[keep]


def trim_surface():
    check_surface()
    res = askItems([_I('Number of trim rounds',1),
                    _I('Minimum number of border edges',1),
                    ])
    n = int(res['Number of trim rounds'])
    nb = int(res['Minimum number of border edges'])
    print("Initial number of elements: %s" % elems.shape[0])
    for i in range(n):
        elems = trim_border(elems,nodes,nb)
        print("Number of elements after border removal: %s" % elems.shape[0])


def read_tetgen(surface=True, volume=True):
    """Read a tetgen model from files  fn.node, fn.ele, fn.smesh."""
    ftype = ''
    if surface:
        ftype += ' *.smesh'
    if volume:
        ftype += ' *.ele'
    fn = askFilename(pf.cfg['workdir'],"Tetgen files (%s)" % ftype)
    nodes = elems =surf = None
    if fn:
        chdir(fn)
        project = utils.projectName(fn)
        set_project(project)
        nodes,nodenrs = tetgen.readNodes(project+'.node')
#        print("Read %d nodes" % nodes.shape[0])
        if volume:
            elems,elemnrs,elemattr = tetgen.readElems(project+'.ele')
            print("Read %d tetraeders" % elems.shape[0])
            PF['volume'] = (nodes,elems)
        if surface:
            surf = tetgen.readSurface(project+'.smesh')
            print("Read %d triangles" % surf.shape[0])
            PF['surface'] = (nodes,surf)
    if surface:
        show_surface()
    else:
        show_volume()


def read_tetgen_surface():
    read_tetgen(volume=False)

def read_tetgen_volume():
    read_tetgen(surface=False)


def scale_volume():
    if PF['volume'] is None:
        return
    nodes,elems = PF['volume']
    nodes *= 0.01
    PF['volume'] = (nodes,elems) 
    



def show_volume():
    """Display the volume model."""
    if PF['volume'] is None:
        return
    nodes,elems = PF['volume']
    F = Formex(nodes[elems],eltype='tet4')
    pf.message("BBOX = %s" % F.bbox())
    clear()
    draw(F,color='random')
    PF['vol_model'] = F


def createCube():
    res = askItems([_I('name','__auto__'),
                    ])
    if res:
        name = res['name']
        S = Cube()
        export({name:S})
        selection.set([name])
        selection.draw()

def createSphere():
    res = askItems([_I('name','__auto__'),
                    _I('ndiv',8,min=1),
                    ])
    if res:
        name = res['name']
        ndiv = res['ndiv']
        S = simple.sphere(ndiv)
        export({name:S})
        selection.set([name])
        selection.draw()


###################  Operations using gts library  ########################


def check():
    S = selection.check(single=True)
    if S:
        pf.message(S.check())


def split():
    S = selection.check(single=True)
    if S:
        pf.message(S.split(base=selection[0],verbose=True))


def coarsen():
    S = selection.check(single=True)
    if S:
        res = askItems([_I('min_edges',-1),
                        _I('max_cost',-1.0),
                        _I('mid_vertex',False),
                        _I('length_cost',False),
                        _I('max_fold',1.0),
                        _I('volume_weight',0.5),
                        _I('boundary_weight',0.5),
                        _I('shape_weight',0.0),
                        _I('progressive',False),
                        _I('log',False),
                        _I('verbose',False),
                        ])
        if res:
            selection.remember()
            if res['min_edges'] <= 0:
                res['min_edges'] = None
            if res['max_cost'] <= 0:
                res['max_cost'] = None
            S=S.coarsen(**res)
            selection.changeValues([S])
            selection.drawChanges()


def boolean():
    """Boolean operation on two surfaces.

    op is one of
    '+' : union,
    '-' : difference,
    '*' : interesection
    """
    surfs = listAll(clas=TriSurface)
    if len(surfs) == 0:
        warning("You currently have no exported surfaces!")
        return
    
    ops = ['+ (Union)','- (Difference)','* (Intersection)']
    res = askItems([_I('surface 1',choices=surfs),
                    _I('surface 2',choices=surfs),
                    _I('operation',choices=ops),
                    _I('check self intersection',False),
                    _I('verbose',False),
                    ],'Boolean Operation')
    if res:
        SA = pf.PF[res['surface 1']]
        SB = pf.PF[res['surface 2']]
        SC = SA.boolean(SB,op=res['operation'].strip()[0],
                        check=res['check self intersection'],
                        verbose=res['verbose'])
        export({'__auto__':SC})
        selection.set('__auto__')
        selection.draw()


def intersection():
    """Intersection curve of two surfaces."""
    surfs = listAll(clas=TriSurface)
    if len(surfs) == 0:
        warning("You currently have no exported surfaces!")
        return
    
    res = askItems([_I('surface 1',choices=surfs),
                    _I('surface 2',choices=surfs),
                    _I('check self intersection',False),
                    _I('verbose',False),
                    ],'Intersection Curve')
    if res:
        SA = pf.PF[res['surface 1']]
        SB = pf.PF[res['surface 2']]
        SC = SA.intersection(SB,check=res['check self intersection'],
                             verbose=res['verbose'])
        export({'__intersection_curve__':SC})
        draw(SC,color=red,linewidth=3)

    
################### menu #################

_menu = 'Surface'

def create_menu():
    """Create the Surface menu."""
    MenuData = [
        ("&Read Surface Files",readSelection),
        ("&Select Surface(s)",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ("&Convert to Formex",toFormex),
        ("&Convert from Formex",fromFormex),
        ("&Convert from Mesh",fromMesh),
        ("&Write Surface Model",write_surface),
        ("---",None),
        ("&Create surface",
         [('&Cube',createCube),
          ('&Sphere',createSphere),
          ]),
        ("&Merge Selection",merge),
        ("&Fix Normals",fixNormals),
        ("&Reverse Normals",reverseNormals),
        #("&Set Property",selection.setProp),
        ("---",None),
        ("Print &Information",
         [('&Data Size',printSize),
          ('&Bounding Box',selection.printbbox),
          ('&Surface Type',printType),
          ('&Total Area',printArea),
          ('&Enclosed Volume',printVolume),
          ('&All Statistics',printStats),
          ]),
        ("&Shrink",toggle_shrink),
        ("Toggle &Annotations",
         [("&Names",selection.toggleNames,dict(checkable=True)),
          ("&Face Numbers",selection.toggleNumbers,dict(checkable=True)),
          ("&Edge Numbers",selection.toggleEdgeNumbers,dict(checkable=True)),
          ("&Node Numbers",selection.toggleNodeNumbers,dict(checkable=True)),
          ("&Normals",selection.toggleNormals,dict(checkable=True)),
          ("&AvgNormals",selection.toggleAvgNormals,dict(checkable=True)),
          ('&Toggle Bbox',selection.toggleBbox,dict(checkable=True)),
          ]),
        ("&Statistics",showStatisticsDialog),
        ("---",None),
        ("&Frontal Methods",
         [("&Color By Front",colorByFront),
          ("&Partition By Connection",partitionByConnection),
          ("&Partition By Angle",partitionByAngle),
          ("&Show Feature Edges",showFeatureEdges),
          ]),
        ("&Show Border",showBorder),
        ("&Fill Border",fillBorders),
#        ("&Fill Holes",fillHoles),
        ("&Delete Triangles",deleteTriangles),
        ("---",None),
        ("&Transform",
         [("&Scale",scaleSelection),
          ("&Scale non-uniformly",scale3Selection),
          ("&Translate",translateSelection),
          ("&Center",centerSelection),
          ("&Rotate",
           [("&Around Global Axis",rotateGlobal),
            ("&Around Parallel Axis",rotateParallel),
            ("&Around Central Axis",rotateCentral),
            ("&Around General Axis",rotateGeneral),
            ]),
          ("&Roll Axes",rollAxes),
          ]),
        ("&Clip/Cut",
         [("&Clip",clipSelection),
          ("&Clip At Plane",clipAtPlane),
          ("&Cut With Plane",cutWithPlane),
          ("&Multiple Cut",cutSelectionByPlanes),
          ("&Intersection With Plane",intersectWithPlane),
          ("&Slicer",slicer),
          ("&Spliner",spliner),
           ]),
        ("&Smooth",smooth),
        ("&Refine",refine),
        ("&Undo Last Changes",selection.undoChanges),
        ("---",None),
        ('&GTS functions',
         [('&Check surface',check),
          ('&Split surface',split),
          ("&Coarsen surface",coarsen),
          ("&Refine",refine),
          ## ("&Smooth surface",smooth),
          ("&Boolean operation on two surfaces",boolean),
          ("&Intersection curve of two surfaces",intersection),
          ]),
#        ("&Show volume model",show_volume),
        # ("&Print Nodal Coordinates",show_nodes),
        # ("&Convert STL file to OFF file",convert_stl_to_off),
        # ("&Sanitize STL file to OFF file",sanitize_stl_to_off),
#        ("&Trim border",trim_surface),
        ("&Create volume mesh",create_volume),
#        ("&Read Tetgen Volume",read_tetgen_volume),
        ("&Export surface to Abaqus",export_surface),
        ("&Export volume to Abaqus",export_volume),
        ("---",None),
        ("&Reload Menu",reload_menu),
        ("&Close Menu",close_menu),
        ]
    return menu.Menu(_menu,items=MenuData,parent=pf.GUI.menu,before='Help')


def show_menu():
    """Show the menu."""
    if not pf.GUI.menu.item(_menu):
        create_menu()


def close_menu():
    """Close the menu."""
    pf.GUI.menu.removeItem(_menu)


def reload_menu():
    """Reload the menu."""
    close_menu()
    show_menu()


####################################################################
######### What to do when the script is executed ###################

if __name__ == "draw":

    reload_menu()

# End
