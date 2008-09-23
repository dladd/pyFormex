#!/usr/bin/env pyformex
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

"""surface_menu.py

STL plugin menu for pyFormex.
"""

import pyformex as GD
from gui import actors,colors,decors,widgets
from gui.colorscale import ColorScale,ColorLegend
from gui.draw import *
from plugins.surface import *
from plugins.objects import *
from plugins import formex_menu,surface_abq
import simple
from plugins.tools import Plane

import os, timer

##################### selection and annotations ##########################


def draw_edge_numbers(n):
    """Draw the edge numbers of the named surface."""
    S = named(n)
    F = Formex(S.coords[S.edges]) 
    return drawNumbers(F,color=colors.red)

def draw_node_numbers(n):
    """Draw the node numbers of the named surface."""
    S = named(n)
    F = Formex(S.coords) 
    return drawNumbers(F,color=colors.blue)

def draw_normals(n):
    S = named(n)
    C = S.centroids()
    A,N = S.areaNormals()
    D = C + sqrt(A).reshape((-1,1))*N
    F = connect([Formex(C),Formex(D)])
    return draw(F,color='red')

def draw_avg_normals(n):
    S = named(n)
    C = S.coords
    N = S.avgVertexNormals()
    try:
        siz = float(GD.cfg['mark/avgnormalsize'])
    except:
        siz = 0.05 * C.dsize()
    D = C + siz * N
    F = connect([Formex(C),Formex(D)])
    return draw(F,color='orange')
    
selection = DrawableObjects(clas=TriSurface)
ntoggles = len(selection.annotations)
def toggleEdgeNumbers():
    selection.toggleAnnotation(0+ntoggles)
def toggleNodeNumbers():
    selection.toggleAnnotation(1+ntoggles)
def toggleNormals():
    selection.toggleAnnotation(2+ntoggles)
def toggleAvgNormals():
    selection.toggleAnnotation(3+ntoggles)


selection.annotations.extend([[draw_edge_numbers,False],
                              [draw_node_numbers,False],
                              [draw_normals,False],
                              [draw_avg_normals,False],
                              ])

##################### select, read and write ##########################

def read_Surface(fn):
    GD.message("Reading file %s" % fn)
    t = timer.Timer()
    S = TriSurface.read(fn)
    GD.message("Read surface with %d vertices, %d edges, %d triangles in %s seconds" % (S.ncoords(),S.nedges(),S.nelems(),t.seconds()))
    return S


def readSelection(select=True,draw=True,multi=True):
    """Read a Surface (or list) from asked file name(s).

    If select is True (default), this becomes the current selection.
    If select and draw are True (default), the selection is drawn.
    """
    types = [ 'Surface Files (*.gts *.stl *.off *.neu *.smesh)', 'All Files (*)' ]
    fn = askFilename(GD.cfg['workdir'],types,exist=True,multi=multi)
    if not multi:
        fn = [ fn ]
    if fn:
        chdir(fn[0])
        names = map(utils.projectName,fn)
        GD.gui.setBusy()
        surfaces = map(read_Surface,fn)
        for i,S in enumerate(surfaces):
            S.setProp(i)
        GD.gui.setBusy(False)
        export(dict(zip(names,surfaces)))
        if select:
            GD.message("Set selection to %s" % str(names))
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
        GD.message("Surface %s has %d vertices, %s edges and %d faces" %
                   (s,S.ncoords(),S.nedges(),S.nelems()))

def printType():
    for s in selection.names:
        S = named(s)
        if S.isClosedManifold():
            GD.message("Surface %s is a closed manifold" % s)
        elif S.isManifold():
            GD.message("Surface %s is an open manifold" % s)
        else:
            GD.message("Surface %s is not a manifold" % s)

def printArea():
    for s in selection.names:
        S = named(s)
        GD.message("Surface %s has area %s" % (s,S.area()))

def printVolume():
    for s in selection.names:
        S = named(s)
        GD.message("Surface %s has volume %s" % (s,S.volume()))


def printStats():
    for s in selection.names:
        S = named(s)
        GD.message("Statistics for surface %s" % s)
        print S.stats()

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
    print "Converted in %s seconds" % t.seconds()
    print surfaces.keys()
    export(surfaces)

    if not suffix:
        formex_menu.selection.clear()
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


## def convert_stl_to_off():
##     """Converts an stl to off format without reading it into pyFormex."""
##     fn = askFilename(GD.cfg['workdir'],"STL files (*.stl)",exist=True)
##     if fn:     
##         return surface.stl_to_off(fn,sanitize=False)


## def sanitize_stl_to_off():
##     """Sanitizes an stl to off format without reading it into pyFormex."""
##     fn = askFilename(GD.cfg['workdir'],"STL files (*.stl)",exist=True)
##     if fn:     
##         return surface.stl_to_off(fn,sanitize=True)




def write_surface(types=['surface','gts','stl','off','neu','smesh']):
    F = selection.check(single=True)
    if F:
        if type(types) == str:
            types = [ types ]
        types = map(utils.fileDescription,types)
        fn = askFilename(GD.cfg['workdir'],types,exist=False)
        if fn:
            GD.message("Exporting surface model to %s" % fn)
            GD.gui.setBusy()
            F.write(fn)   
            GD.gui.setBusy(False)

#
# Operations with surface type, border, ...
#
def showBorder():
    S = selection.check(single=True)
    if S:
        print S.nEdgeConnected()
        print S.borderEdges()
        F = S.border()
        if F.nelems() > 0:
            draw(F,color='red',linewidth=3)
            export({'border':F})
        else:
            warning("The surface %s does not have a border" % selection[0])

def checkBorder():
    S = selection.check(single=True)
    if S:
        S.checkBorder()


def fillBorder():
    S = selection.check(single=True)
    if S:
        options = ["Cancel","Existing points","New points"]
        res = ask("Which method ?",options)
        if res == options[1]: 
            S.fillBorder(0)
        elif res == options[2]: 
            S.fillBorder(1)


def fillHoles():
    """Fill the holes in the selected surface."""
    S = selection.check(single=True)
    if S:
        border_elems = S.edges[S.borderEdges()]
        if border_elems.size != 0:
            # partition borders
            print border_elems
            border_elems = partitionSegmentedCurve(border_elems)
            print border_elems
            
            # draw borders in new viewport
            R = GD.canvas.camera.getRot()
            P = GD.canvas.camera.perspective
            layout(2)
            viewport(1)
            GD.canvas.camera.rot = R
            toolbar.setPerspective(P)
            for i,elems in enumerate(border_elems):
                draw(Formex(S.coords[elems]))
            zoomAll()
            # pick borders for which the hole must be filled
            info("PICK HOLES WHICH HAVE TO BE FILLED.")
            picked = pick(mode='actor')
            layout(1)
            # fill the holes
            triangles = empty((0,3,),dtype=int)
            if picked.has_key(-1):
                for i in picked[-1]:
                    triangles = row_stack([triangles,fillHole(S.coords,border_elems[int(i)])])
                T = TriSurface(S.coords,triangles)
                S.append(T)
                draw(T,color='red',bbox=None)
            else:
                warning("No borders were picked.")
        else:
            warning("The surface %s does not have a border." % selection[0])


# Selectable values for display/histogram
# Each key is a description of a result
# Each value consist of a tuple
#  - function to calculate the values
#  - domain to display: True to display on edges, False to display on elements

SelectableStatsValues = {
    'Facet Area': (TriSurface.facetArea,False),
    'Aspect ratio': (TriSurface.aspectRatio,False),
    'Smallest altitude': (TriSurface.smallestAltitude,False),
    'Longest edge': (TriSurface.longestEdge,False),
    'Shortest edge': (TriSurface.shortestEdge,False),
    'Number of node adjacent elements': (TriSurface.nNodeAdjacent,False),
    'Number of edge adjacent elements': (TriSurface.nEdgeAdjacent,False),
    'Edge angle': (TriSurface.edgeAngles,True),
    'Number of connected elements': (TriSurface.nEdgeConnected,True),
    }


def showStatistics():
    S = selection.check(single=True)
    if S:
        dispmodes = ['On Domain','Histogram','Cumulative Histogram']
        keys = SelectableStatsValues.keys()
        keys.sort()
        res  = askItems([('Select Value',None,'select',keys),
                         ('Display Mode',None,'select',dispmodes)
                         ])
        GD.app.processEvents()
        if res:
            key = res['Select Value']
            func,onEdges = SelectableStatsValues[key]
            val = func(S)
            mode = res['Display Mode']
            if mode == 'On Domain':
                showSurfaceValue(S,key,val,onEdges)
            else: 
                showHistogram(key,val,cumulative= mode.startswith('Cumul'))

                                              
def showSurfaceValue(S,txt,val,onEdges):
    mi,ma = val.min(),val.max()
    dec = min(abs(mi),abs(ma))
    dec = int(log10(dec))
    dec = max(0,3-dec)
    # create a colorscale and draw the colorlegend
    CS = ColorScale([colors.blue,colors.yellow,colors.red],mi,ma,0.5*(mi+ma),1.)
    cval = array(map(CS.color,val))
    clear()
    if onEdges:
        F = Formex(S.coords[S.edges])
        draw(F,color=cval)#,linewidth=2)
    else:
        draw(S,color=cval)
    CL = ColorLegend(CS,100)
    CLA = decors.ColorLegend(CL,10,10,30,200,dec=dec) 
    GD.canvas.addDecoration(CLA)
    drawtext(txt,10,230,'hv18')


def colorByFront():
    S = selection.check(single=True)
    if S:
        res  = askItems([('front type',None,'select',['node','edge']),
                         ('number of colors',-1),
                         ('front width',1),
                         ('start at',0),
                         ('first prop',0),
                         ])
        GD.app.processEvents()
        if res:
            selection.remember()
            t = timer.Timer()
            ftype = res['front type']
            nwidth = res['front width']
            nsteps = nwidth * res['number of colors']
            startat = res['start at']
            firstprop = res['first prop']
            if ftype == 'node':
                p = S.walkNodeFront(nsteps=nsteps,startat=startat)
            else:
                p = S.walkEdgeFront(nsteps=nsteps,startat=startat)
            S.setProp(p/nwidth + firstprop)
            print "Colored in %s parts (%s seconds)" % (S.p.max()+1,t.seconds())
            selection.draw()


def partitionByConnection():
    S = selection.check(single=True)
    if S:
        selection.remember()
        t = timer.Timer()
        S.p = S.partitionByConnection()
        print "Partitioned in %s parts (%s seconds)" % (S.p.max()+1,t.seconds())
        selection.draw()


def partitionByAngle():
    S = selection.check(single=True)
    if S:
        res  = askItems([('angle',60.),('firstprop',1),('startat',0)])
        GD.app.processEvents()
        if res:
            selection.remember()
            t = timer.Timer()
            S.p = S.partitionByAngle(**res)
            print "Partitioned in %s parts (%s seconds)" % (S.p.max()+1,t.seconds())
            selection.draw()
         
 

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
        res = askItems([['scale',1.0]],
                       caption = 'Scaling Factor')
        if res:
            scale = float(res['scale'])
            selection.remember(True)
            for F in FL:
                F.scale(scale)
            selection.drawChanges()

            
def scale3Selection():
    """Scale the selection with 3 scale values."""
    FL = selection.check()
    if FL:
        res = askItems([['x-scale',1.0],['y-scale',1.0],['z-scale',1.0]],
                       caption = 'Scaling Factors')
        if res:
            scale = map(float,[res['%c-scale'%c] for c in 'xyz'])
            selection.remember(True)
            for F in FL:
                F.scale(scale)
            selection.drawChanges()


def translateSelection():
    """Translate the selection."""
    FL = selection.check()
    if FL:
        res = askItems([['direction',0],['distance','1.0']],
                       caption = 'Translation Parameters')
        if res:
            dir = int(res['direction'])
            dist = float(res['distance'])
            selection.remember(True)
            for F in FL:
                F.translate(dir,dist)
            selection.drawChanges()


def centerSelection():
    """Center the selection."""
    FL = selection.check()
    if FL:
        selection.remember(True)
        for F in FL:
            F.translate(-F.coords.center())
        selection.drawChanges()


def rotate(mode='global'):
    """Rotate the selection.

    mode is one of 'global','parallel','central','general'
    """
    FL = selection.check()
    if FL:
        if mode == 'global':
            res = askItems([['angle','90.0'],['axis',2]])
            if res:
                angle = float(res['angle'])
                axis = int(res['axis'])
                around = None
        elif mode == 'parallel':
            res = askItems([['angle','90.0'],['axis',2],['point','[0.0,0.0,0.0]']])
            if res:
                axis = int(res['axis'])
                angle = float(res['angle'])
                around = eval(res['point'])
        elif mode == 'central':
            res = askItems([['angle','90.0'],['axis','[0.0,0.0,0.0]']])
            if res:
                angle = float(res['angle'])
                axis = eval(res['axis'])
                around = None
        elif mode == 'general':
            res = askItems([['angle','90.0'],['axis','[0.0,0.0,0.0]'],['point','[0.0,0.0,0.0]']])
            if res:
                angle = float(res['angle'])
                axis = eval(res['axis'])
                around = eval(res['point'])
        if res:
            selection.remember(True)
            for F in FL:
                #print "ROTATE %s %s %s " % (angle,axis,around)
                F.rotate(angle,axis,around)
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
        for F in FL:
            F.coords.rollAxes()
        selection.drawChanges()


def clip_surface():
    """Clip the stl model."""
    if not check_surface():
        return
    itemlist = [['axis',0],['begin',0.0],['end',1.0],['nodes','any']]
    res = widgets.InputDialog(itemlist,'Clipping Parameters').getResult()
    if res:
        updateGUI()
        nodes,elems = PF['old_surface'] = PF['surface']
        F = Formex(nodes[elems])
        bb = F.bbox()
        GD.message("Original bbox: %s" % bb) 
        xmi = bb[0][0]
        xma = bb[1][0]
        dx = xma-xmi
        axis = int(res[0][1])
        xc1 = xmi + float(res[1][1]) * dx
        xc2 = xmi + float(res[2][1]) * dx
        nodid = res[3][1]
        #print nodid
        clear()
        draw(F,color='yellow')
        w = F.test(nodes='any',dir=axis,min=xc1,max=xc2)
        F = F.clip(w)
        draw(F,color='red')


def clipSelection():
    """Clip the selection.
    
    The coords list is not changed.
    """
    FL = selection.check()
    if FL:
        res = askItems([['axis',0],['begin',0.0],['end',1.0],['nodes','all','select',['all','any','none']]],caption='Clipping Parameters')
        if res:
            bb = bbox(FL)
            axis = int(res['axis'])
            xmi = bb[0][axis]
            xma = bb[1][axis]
            dx = xma-xmi
            xc1 = xmi + float(res['begin']) * dx
            xc2 = xmi + float(res['end']) * dx
            selection.changeValues([ F.clip(F.test(nodes=res['nodes'],dir=axis,min=xc1,max=xc2)) for F in FL ])
            selection.drawChanges()


def cutAtPlane():
    """Cut the selection with a plane."""
    FL = selection.check()
    res = askItems([['Point',(0.0,0.0,0.0)],
                    ['Normal',(1.0,0.0,0.0)],
                    ['New props',[1,2,2,3,4,5,6]],
                    ['Side','positive', 'radio', ['positive','negative','both']],
                    ['Tolerance',0.],
                    ],caption = 'Define the cutting plane')
    if res:
        P = res['Point']
        N = res['Normal']
        p = res['New props']
        side = res['Side']
        atol = res['Tolerance']
        selection.remember(True)
        if side == 'both':
            G = [F.toFormex().cutAtPlane(P,N,newprops=p,side=side,atol=atol) for F in FL]
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
            [F.cutAtPlane(P,N,newprops=p,side=side,atol=atol) for F in FL]
            selection.drawChanges()


def cutSelectionByPlanes():
    """Cut the selection with one or more planes, which are already created."""
    S = selection.check(single=True)
    if S:
        res1 = widgets.Selection(listAll(clas=Plane),
                                'Known %sobjects' % selection.object_type(),
                                mode='multi',sort=True).getResult()
        if res1:
            res2 = askItems([['Tolerance',0.],
                    ['Color by', 'side', 'radio', ['side', 'element type']], 
                    ['Side','both', 'radio', ['positive','negative','both']]],
                    caption = 'Cutting parameters')
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
                    Spos, Sneg = S.toFormex().cutAtPlane(p,n,newprops=newprops,side=side,atol=atol)
                elif side == 'positive':
                    Spos = S.toFormex().cutAtPlane(p,n,newprops=newprops,side=side,atol=atol)
                    Sneg = Formex()
                elif side == 'negative':
                    Sneg = S.toFormex().cutAtPlane(p,n,newprops=newprops,side=side,atol=atol)
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
    res = askItems([['Name','__auto__'],
                    ['Point',(0.0,0.0,0.0)],
                    ['Normal',(0.0,0.0,1.0)],
                    ],caption = 'Define the cutting plane')
    if res:
        name = res['Name']
        P = res['Point']
        N = res['Normal']
        G = [F.toFormex().intersectionLinesWithPlane(P,N) for F in FL]
        draw(G,color='red',linewidth=3)
        export(dict([('%s/%s' % (n,name),g) for n,g in zip(selection,G)])) 


def sliceIt():
    """Slice the surface to a sequence of cross sections."""
    S = selection.check(single=True)
    res = askItems([['Direction',0],
                    ['# slices',10],
                   ],caption = 'Define the slicing planes')
    if res:
        axis = res['Direction']
        nslices = res['# slices']
        bb = S.bbox()
        xmin,xmax = bb[:,axis]
        dx =  (xmax-xmin) / nslices
        x = arange(nslices+1) * dx
        N = unitVector(axis)
        print N
        P = [ bb[0]+N*s for s in x ]
        G = [S.toFormex().intersectionLinesWithPlane(Pi,N) for Pi in P]
        #[ G.setProp(i) for i,G in enumerate(G) ]
        G = Formex.concatenate(G)
        draw(G,color='red',linewidth=3)
        export({'%s/slices%s' % (selection[0],axis):G}) 


###################################################################
########### The following functions are in need of a make-over


def create_tetgen_volume():
    """Generate a volume tetraeder mesh inside an stl surface."""
    types = [ 'STL/OFF Files (*.stl *.off)', 'All Files (*)' ]
    fn = askFilename(GD.cfg['workdir'],types,exist=True,multi=False)
    if os.path.exists(fn):
        sta,out = utils.runCommand('tetgen -z %s' % fn)
        GD.message(out)


def flytru_stl():
    """Fly through the stl model."""
    global ctr
    Fc = Formex(array(ctr).reshape((-1,1,3)))
    path = connect([Fc,Fc],bias=[0,1])
    flyAlong(path)
    

def export_stl():
    """Export an stl model stored in Formex F in Abaqus .inp format."""
    global project,F
    if ack("Creating nodes and elements.\nFor a large model, this could take quite some time!"):
        GD.app.processEvents()
        GD.message("Creating nodes and elements.")
        nodes,elems = F.feModel()
        nnodes = nodes.shape[0]
        nelems = elems.shape[0]
        GD.message("There are %d unique nodes and %d triangle elements in the model." % (nnodes,nelems))
        stl_abq.abq_export(project+'.inp',nodes,elems,'S3',"Created by stl_examples.py")

def export_surface():       
    S = selection.check(single=True)
    if S:
        types = [ "Abaqus INP files (*.inp)" ]
        fn = askFilename(GD.cfg['workdir'],types,exist=False)
        if fn:
            print "Exporting surface model to %s" % fn
            updateGUI()
            S.refresh()
            nodes,elems = S.coords,S.elems
            surface_abq.abq_export(fn,nodes,elems,'S3',"Abaqus model generated by pyFormex from input file %s" % os.path.basename(fn))



def export_volume():
    if PF['volume'] is None:
        return
    types = [ "Abaqus INP files (*.inp)" ]
    fn = askFilename(GD.cfg['workdir'],types,exist=False)
    if fn:
        print "Exporting volume model to %s" % fn
        updateGUI()
        nodes,elems = PF['volume']
        stl_abq.abq_export(fn,nodes,elems,'C3D%d' % elems.shape[1],"Abaqus model generated by tetgen from surface in STL file %s.stl" % PF['project'])


def show_nodes():
    n = 0
    data = askItems({'node number':n})
    n = int(data['node number'])
    if n > 0:
        nodes,elems = PF['surface']
        print "Node %s = %s",(n,nodes[n])


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
    print "Selected %s of %s elements, leaving %s" % (ntrim,nelems,nkeep) 

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
    data = GD.cfg.get('stl/border',{'Number of trim rounds':1, 'Minimum number of border edges':1})
    GD.cfg['stl/border'] = askItems(data)
    GD.gui.update()
    n = int(data['Number of trim rounds'])
    nb = int(data['Minimum number of border edges'])
    print "Initial number of elements: %s" % elems.shape[0]
    for i in range(n):
        elems = trim_border(elems,nodes,nb)
        print "Number of elements after border removal: %s" % elems.shape[0]


def read_tetgen(surface=True, volume=True):
    """Read a tetgen model from files  fn.node, fn.ele, fn.smesh."""
    ftype = ''
    if surface:
        ftype += ' *.smesh'
    if volume:
        ftype += ' *.ele'
    fn = askFilename(GD.cfg['workdir'],"Tetgen files (%s)" % ftype,exist=True)
    nodes = elems =surf = None
    if fn:
        chdir(fn)
        project = utils.projectName(fn)
        set_project(project)
        nodes,nodenrs = tetgen.readNodes(project+'.node')
#        print "Read %d nodes" % nodes.shape[0]
        if volume:
            elems,elemnrs,elemattr = tetgen.readElems(project+'.ele')
            print "Read %d tetraeders" % elems.shape[0]
            PF['volume'] = (nodes,elems)
        if surface:
            surf = tetgen.readSurface(project+'.smesh')
            print "Read %d triangles" % surf.shape[0]
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
    GD.message("BBOX = %s" % F.bbox())
    clear()
    draw(F,color='random')
    PF['vol_model'] = F



def createGrid():
    res = askItems([('name','__auto__'),('nx',3),('ny',3),('b',1),('h',1)])
    if res:
        globals().update(res)
        #name = res['name']
        #nx = res['nx']
        #ny = res['ny']
        #b = 
        S = TriSurface(simple.rectangle(nx,ny,b,h,diag='d'))
        export({name:S})
        selection.set([name])
        selection.draw()


def createCube():
    res = askItems([('name','__auto__')])
    if res:
        name = res['name']
        S = Cube()
        export({name:S})
        selection.set([name])
        selection.draw()

def createSphere():
    res = askItems([('name','__auto__'),('grade',4),])
    if res:
        name = res['name']
        level = max(1,res['grade'])
        S = Sphere(level,verbose=True,filename=name+'.gts')
        export({name:S})
        selection.set([name])
        selection.draw()

_data = {}

def createCone():
    res = askItems([('name','__auto__'),
                    ('radius',1.),
                    ('height',1.),
                    ('angle',360.),
                    ('div_along_radius',6),
                    ('div_along_circ',12),
                    ('diagonals','up','select',['up','down']),
                    ])
    if res:
        name = res['name']
        F = simple.sector(r=res['radius'],t=res['angle'],nr=res['div_along_radius'],nt=res['div_along_circ'],h=res['height'],diag=res['diagonals'])
        export({name:TriSurface(F)})
        selection.set([name])
        selection.draw()


###################  Operations using gts library  ########################


def check():
    S = selection.check(single=True)
    if S:
        GD.message(S.check(verbose=True))


def split():
    S = selection.check(single=True)
    if S:
        GD.message(S.split(base=selection[0],verbose=True))


def coarsen():
    S = selection.check(single=True)
    if S:
        res = askItems([('min_edges',-1),
                        ('max_cost',-1),
                        ('mid_vertex',False),
                        ('length_cost',False),
                        ('max_fold',1.0),
                        ('volume_weight',0.5),
                        ('boundary_weight',0.5),
                        ('shape_weight',0.0),
                        ('progressive',False),
                        ('log',False),
                        ('verbose',False),
                        ])
        if res:
            selection.remember()
            if res['min_edges'] <= 0:
                res['min_edges'] = None
            if res['max_cost'] <= 0:
                res['max_cost'] = None
            S.coarsen(**res)
            selection.draw()


def refine():
    S = selection.check(single=True)
    if S:
        res = askItems([('max_edges',-1),
                        ('min_cost',-1),
                        ('log',False),
                        ('verbose',False),
                        ])
        if res:
            selection.remember()
            if res['max_edges'] <= 0:
                res['max_edges'] = None
            if res['min_cost'] <= 0:
                res['min_cost'] = None
            S.refine(**res)
            selection.draw()


def smooth():
    S = selection.check(single=True)
    if S:
        res = askItems([('lambda_value',0.5),
                        ('n_iterations',2),
                        ('fold_smoothing',None),
                        ('verbose',False),
                        ],'Laplacian Smoothing')
        if res:
            selection.remember()
            if res['fold_smoothing'] is not None:
                res['fold_smoothing'] = float(res['fold_smoothing'])
            S.smooth(**res)
            selection.draw()


def boolean():
    """Boolean operation on two surfaces.

    op is one of
    '+' : union,
    '-' : difference,
    '*' : interesection
    """
    ops = ['+ (Union)','- (Difference)','* (Intersection)']
    S = selection.check(single=False)
    if len(selection.names) != 2:
        warning("You must select exactly two triangulated surfaces!")
        return
    if S:
        res = askItems([('operation',None,'select',ops),
                        ('output intersection curve',False),
                        ('check self interesection',False),
                        ('verbose',False),
                        ],'Boolean Operation')
        if res:
            #selection.remember()
            newS = S[0].boolean(S[1],op=res['operation'].strip()[0],
                        inter=res['output intersection curve'],
                        check=res['check self interesection'],
                        verbose=res['verbose'])
            export({'__auto__':newS})
            #selection.draw()


################### dependent on gnuplot ####################



def showHistogram(txt,val,cumulative=False):
    if not utils.hasModule('gnuplot'):
        error("You do not have the Python Gnuplot module installed.\nI can not draw the requested plot.")
        return
        
    import Gnuplot

    y,x = histogram(val)
    #hist = column_stack([hist[1],hist[0]])
    if cumulative:
        y = y.cumsum()
    data = Gnuplot.Data(x,y,
                        title=txt,
                        with='histeps')  # boxes?
    g = Gnuplot.Gnuplot(persist=1)
    g.title('Histogram of %s' % txt)
    #g('set boxwidth 1')
    g.plot(data)


    
################### menu #################

def create_menu():
    """Create the Surface menu."""
    MenuData = [
        ("&Read Surface Files",readSelection),
        ("&Select Surface(s)",selection.ask),
        ("&Draw Selection",selection.draw),
        ("&Forget Selection",selection.forget),
        ("&Convert to Formex",toFormex),
        ("&Convert from Formex",fromFormex),
        ("&Write Surface Model",write_surface),
        ("---",None),
        ("&Create surface",
         [('&Plane Grid',createGrid),
          ('&Cube',createCube),
          ('&Sphere',createSphere),
          ('&Circle, Sector, Cone',createCone),
          ]),
        ("---",None),
        ("Print &Information",
         [('&Data Size',printSize),
          ('&Bounding Box',selection.printBbox),
          ('&Surface Type',printType),
          ('&Total Area',printArea),
          ('&Enclosed Volume',printVolume),
          ('&All Statistics',printStats),
          ]),
        ("&Set Property",selection.setProperty),
        ("&Shrink",toggle_shrink),
        ("Toggle &Annotations",
         [("&Names",selection.toggleNames,dict(checkable=True)),
          ("&Face Numbers",selection.toggleNumbers,dict(checkable=True)),
          ("&Edge Numbers",toggleEdgeNumbers,dict(checkable=True)),
          ("&Node Numbers",toggleNodeNumbers,dict(checkable=True)),
          ("&Normals",toggleNormals,dict(checkable=True)),
          ("&AvgNormals",toggleAvgNormals,dict(checkable=True)),
          ('&Toggle Bbox',selection.toggleBbox,dict(checkable=True)),
          ]),
        ("&Statistics",showStatistics),
        ("---",None),
        ("&Frontal Methods",
         [("&Color By Front",colorByFront),
          ("&Partition By Connection",partitionByConnection),
          ("&Partition By Angle",partitionByAngle),
          ]),
        ("&Border Line",showBorder),
        ("&Border Type",checkBorder),
        ("&Fill Border",fillBorder),
        ("&Fill Holes",fillHoles),
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
        ("&Cut",
         [("&Clip",clipSelection),
          ("&Cut(Trim) at Plane",cutAtPlane),
          ("&Cut by Planes",cutSelectionByPlanes),
          ("&Intersect With Plane",intersectWithPlane),
          ("&Slice",sliceIt),
           ]),
        ("&Undo Last Changes",selection.undoChanges),
        ('&GTS functions',
         [('&Check surface',check),
          ('&Split surface',split),
          ("&Coarsen surface",coarsen),
          ("&Refine surface",refine),
          ("&Smooth surface",smooth),
          ("&Boolean operation on two surfaces",boolean),
          ]),
        ("---",None),
#        ("&Show volume model",show_volume),
        #("&Print Nodal Coordinates",show_nodes),
        # ("&Convert STL file to OFF file",convert_stl_to_off),
        # ("&Sanitize STL file to OFF file",sanitize_stl_to_off),
#        ("&Trim border",trim_surface),
        ("&Create volume mesh",create_tetgen_volume),
#        ("&Read Tetgen Volume",read_tetgen_volume),
        ("&Export surface to Abaqus",export_surface),
#        ("&Export volume to Abaqus",export_volume),
        ("&Close Menu",close_menu),
        ]
    return widgets.Menu('Surface',items=MenuData,parent=GD.gui.menu,before='help')

    
def show_menu():
    """Show the Tools menu."""
    if not GD.gui.menu.item('Surface'):
        create_menu()


def close_menu():
    """Close the Tools menu."""
    m = GD.gui.menu.item('Surface')
    if m :
        m.remove()
    

if __name__ == "draw":
    # If executed as a pyformex script
    close_menu()
    show_menu()
    
elif __name__ == "__main__":
    print __doc__

# End
