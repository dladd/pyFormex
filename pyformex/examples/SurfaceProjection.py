#!/usr/bin/env pyformex
# $Id$

"""IntersectTriSurfaceWithLines.py

level = 'normal'
topics = ['surface']
techniques = ['transform','projection']
author: gianluca

.. Description

IntersectTriSurfaceWithLines
----------------------------

This example illustrates the use of intersectSurfaceWithLines.

"""
from plugins.isopar import *
from plugins.trisurface import *
from plugins.mesh import *
from connectivity import inverseIndex
from gui.widgets import simpleInputItem as I
from gui.imagecolor import *

clear()
smooth()
lights(True)
transparent(False)
view('iso')



image = None
scaled_image = None

#teapot
chdir(__file__)
T = TriSurface.read(pf.cfg['pyformexdir']+'/data/teapot.off')#file with horse stl
xmin,xmax = T.bbox()
T= T.translate(-T.center()).scale(1./(xmax[0]-xmin[0])).scale(4.).setProp(2)

draw(T)


##to convert image color to 8 bit (256 levels) use GIMP image-mode-indexed and save as png
filename = getcfg('datadir')+'/benedict_6.jpg'

def selectFile():
    """Select an image file."""
    global filename
    filename = askFilename(filename,filter=utils.fileDescription('img'),multi=False,exist=True)
    if filename:
        currentDialog().updateData({'filename':filename})
        loadImage()

def loadImage():
    global image, scaled_image
    image = QtGui.QImage(filename)
    if image.isNull():
        warning("Could not load image '%s'" % filename)
        return None

    w,h = image.width(),image.height()
    print "size = %sx%s" % (w,h)

    diag = currentDialog()
    if diag:
        diag.updateData({'nx':w,'ny':h})

    maxsiz = 40000.
    if w*h > maxsiz:
        scale = sqrt(maxsiz/w/h)
        w = int(w*scale)
        h = int(h*scale)
    return w, h


#the projection is made only with few control points (px*py) while the picture is reconstructed many points (nx*ny).

px, py=5, 5#control points for projection of patches
kx, ky= 50,50#number of cells in each patch

res = askItems([
    ('filename',filename,{'buttons':[('Select File',selectFile)]}),
    ('px',px,{'text':'patch x'}),
    ('kx',kx,{'text':'pixel in patch x'}), 
    ('py',py,{'text':'patch y'}),
    ('ky',ky,{'text':'pixel in patch y'}), 
    ])

if not res:
    exit()

globals().update(res)



nx, ny=px*kx, py*ky#pixels
print 'the picture is reconstructed with %d x %d pixels'%(nx, ny)

F = Formex(mpattern('123')).replic2(nx,ny).centered()
if image is None:
    print "Loading image"
    wpic, hpic=loadImage()

if image is None:
    exit()

# Create the colors
color,colortable = image2glcolor(image.scaled(nx,ny))
print "Converting image to color array"


# Create a 2D grid of nx*ny elements
##trick: to speed up the projection, the grid is patched and then refined with the isop quad8!
def makePatches8(px=3., py=2.):
    """it makes patches (quad regions) in xy of 8 nods each, suitable for the isop transformation with type quad8. The patches together are in the region (0.,0.) (1.,1.).
    From this patches, a grid can be generated using the makeGrid8"""
    #px*py is the nr of patches
    Pat=Formex( mpattern('123') )
    Fp=Pat.replic2(n1=px,n2=py)   
    cp=concatenate( [ Fp[:, 0], Fp[:, 1],Fp[:, 2],Fp[:, 3],    (Fp[:, 0]+Fp[:, 1])*0.5,  (Fp[:, 1]+Fp[:, 2])*0.5,  (Fp[:, 2]+Fp[:,3])*0.5, (Fp[:, 3]+Fp[:, 0])*0.5] , axis=1).reshape(-1, 8, 3)
    cp=[ Coords(cpi) for cpi in cp]
    #drawNumbers(Formex(cp[0]))
    #draw(Fp, linewidth=2)
    mcp=Formex(cp, eltype='Quad8').centered().toMesh().renumber()
    sc=mcp.sizes()
    return mcp.scale([ 1./sc[0], 1./sc[1], 1.] )

def makeGrid8(mcp, px=3., py=2., kx=30., ky=20.):
    #px*py is the nr of patches
    #kx*ky is the nr of quads in 1 patch
    Pat=Formex( mpattern('123') )
    Sp=Pat.replic2(n1=kx,n2=ky).scale([1./kx, 1./ky, 1.])
    #draw(Sp)
    Sm=Sp.toMesh().renumber()
    cp=mcp.toFormex()
    cp0=Coords(elements.Quad8().vertices )
    #drawNumbers(cp0)
    #[drawNumbers(cp[i]) for i in range(px*py) ]
    fullGrid=concatenate([ concatenate([mesh.Mesh.isopar(Sm,'quad8',cp[i+int(px)*j], cp0).toFormex()[:].reshape(ky, kx,4, 3) for i in range(px)], 1) for j in range(py) ], 0).reshape(-1, 4, 3)
    #drawNumbers(Formex(fullGrid))
    return Formex(fullGrid, eltype='Quad4')


mH=makePatches8(px, py)
try:
    hpic, wpic
    ratioYX=float(hpic)/wpic
    mH=mH.scale(ratioYX, 1)#Keep original proportions
except: pass
#draw(mH)
#draw(makeGrid8(mH, px, py, kx, ky) )

mH0=mH.copy().translate([-0.5, 0.1, 2.])
mH1=mH.copy().rotate(-30., 0).scale(0.5).translate([0., -.7, -2.])
def getEdgesMesh(m):
    return Mesh(m.coords,m.getEdges(unique=True) )

draw(makeGrid8(mH0, px, py, kx, ky),color=color,colormap=colortable)
draw(makeGrid8(mH1, px, py, kx, ky),color=color,colormap=colortable)
zoomAll()
sleep(1)
de0=draw(getEdgesMesh(mH0).translate([0., 0., 0.001]), color='white', linewidth=2)
de1=draw(getEdgesMesh(mH1).translate([0., 0., 0.001]), color='white', linewidth=2)
zoomAll()
sleep(1)

def intersectSurfaceWithSegments2(s1, segm, atol=1.e-5, max1xperline=True):
    """it takes a TriSurface ts and a set of segments (-1,2,3) and intersect the segments with the TriSurface.
    It returns the points of intersections and, for each point, the indices of the intersected segment and triangle. If max1xperline is True, only 1 intersection per line is returned (in order to remove multiple intersections due to the tolerance) together with the index of line and triangle (at the moment the selection of one intersection among the others is random: it does not take into account the distances). If some segments do not intersect the surface, their indices are also returned."""
    p, il, it=trisurface.intersectSurfaceWithLines(s1, segm[:, 0], normalize(segm[:, 1]-segm[:, 0]))
    win= length(p-segm[:, 0][il])+ length(p-segm[:, 1][il])< length(segm[:, 1][il]-segm[:, 0][il])+atol
    px, ilx, itx=p[win], il[win], it[win]
    if max1xperline:
        ip= inverseIndex(ilx.reshape(-1, 1))
        sp=sort(ip, axis=1)[:, -1]
        w= where(sp>-1)[0]
        sw=sp[w]
        return px[sw], w, itx[sw], delete( arange(len(segm)),  w)
    else:return px, ilx, itx




x= column_stack([mH0.coords, mH1.coords]).reshape(-1, 2, 3)
print 'intersecting %d lines with %d triangles'%(len(x), T. nelems())
dx=draw(Formex(x), linewidth=1, color='gray')

pts, il, it, mil=intersectSurfaceWithSegments2(T, x, max1xperline=True)
print '---%d missing---%d segments of the original %d do not intersect' %(len(mil),  len(x)-len(pts), len(x)  )
if len(x)-len(pts)!=0:raise ValueError,"some of the lines do not intersect the surface"
dp=draw(Formex(pts), marksize=6, color='white')


zoomAll()
sleep(1)

undraw(dp)
undraw(dx)
undraw(de0)
undraw(de1)
G=makeGrid8(mH._set_coords(Coords(pts)), px, py, kx, ky)


zoomAll()
draw(G.translate([0., 0., 0.01]),color=color,colormap=colortable)
draw(G.translate([0., 0., -0.01]),color=color,colormap=colortable)
zoomAll()
sleep(2)
pf.canvas.camera.setRotation(200, 0)
zoomAll()



