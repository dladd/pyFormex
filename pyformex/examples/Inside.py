# $Id$
"""Inside

This example shows how to find out if points are inside a closed surface.

"""
_status = 'checked'
_level = 'normal'
_topics = ['surface']
_techniques = ['inside']

from gui.draw import *
import simple
import timer

filename = os.path.join(getcfg('datadir'),'horse.off')


def selectSurfaceFile(fn):
    fn = askFilename(fn,filter=utils.fileDescription('surface'))
    return fn

def create():
    """Create a closed surface and a set of points."""

    res = askItems(
        [ _G('Surface', [
            _I('surface','file',choices=['file','sphere']),
            _I('filename',filename,text='Image file',itemtype='button',func=selectSurfaceFile),
            _I('grade',8),
            _I('refine',0),
            ]),
          _G('Points', [
              _I('points','grid',choices=['grid','random']),
              _I('npts',[30,30,30],itemtype='ivector'),
              _I('scale',[1.,1.,1.],itemptype='point'),
              _I('trl',[0.,0.,0.],itemptype='point'),
              ]),
          ],
        enablers = [
            ( 'surface','file','filename', ),
            ( 'surface','sphere','grade', ),
            ],
        )

    if not res:
        return

    globals().update(res)
    nx,ny,nz = npts

    # Create surface
    if surface == 'file':
        S = TriSurface.read(filename)
    elif surface == 'sphere':
        S = simple.sphere(ndiv=grade)

    if refine > S.nedges():
        S = S.refine(refine)
    
    draw(S, color='red')

    if not S.isClosedManifold():
        warning("This is not a closed manifold surface. Try another.")
        return
    
    # Create points

    if points == 'grid':
        P = simple.regularGrid([-1.,-1.,-1.],[1., 1., 1.],[nx-1,ny-1,nz-1])
    else:
        P = Coords(random.rand(nx*ny*nz*3))

    sc = array(scale)
    print scale
    siz = array(S.sizes())
    print siz
    print sc*siz
    P = Formex(P.reshape(-1, 3)).resized(sc*siz).centered()

    draw(P, marksize=1, color='black')
    
    #P = Coords(P).translate([1.5,0.5,0.5])
    
    ## print "Testing %s points against %s faces" % (P.shape[0],F.nelems())
    ## clear()
    ## draw(F, color='red')
    ## bb = bboxIntersection(F,P)
    ## drawBbox(bb,color=array(red),linewidth=2)
    ## zoomAll()

    ## t = timer.Timer()
    ## ind = F.inside(P)
    ## print "gtsinside: found %s points in %s seconds" % (len(ind),t.seconds())

    ## if len(ind) > 0:
    ##     draw(P[ind],color=green,marksize=5,ontop=True,nolight=True,bbox=None)


def doitgts(S,n,m,rand=False):
    clear()

    ## if rand:
    ##     P = Coords(random.rand(n*n*n*3))
    ## else:
    ##     P = simple.regularGrid([-1.,-1.,-1.],[1., 1., 1.],[n-1,n-1,n-1])
    ##     P = Coords(P).translate([1.5,0.5,0.5])

    ## P = P.reshape(-1, 3)
    ## F = S.refine(m)
    
    ## print "Testing %s points against %s faces" % (P.shape[0],F.nelems())
    ## clear()
    ## draw(F, color='red')
    ## draw(P, marksize=1, color='black')
    ## bb = bboxIntersection(F,P)
    ## drawBbox(bb,color=array(red),linewidth=2)
    ## zoomAll()

    ## t = timer.Timer()
    ## ind = F.inside(P)
    ## print "gtsinside: found %s points in %s seconds" % (len(ind),t.seconds())

    ## if len(ind) > 0:
    ##     draw(P[ind],color=green,marksize=5,ontop=True,nolight=True,bbox=None)



def run():
    resetAll()
    clear()
    smooth()

    chdir(__file__)

    create()

    ## doitgts(S,30,10000)
    ## return
    
    ## for m in [ 2000, 5000, 10000, 20000 ]:

    ##     for n in [  20, 30, 40 ]:

    ##         doitgts(S,n,m)

    ## return

    ## res = askItems([_I('size',20,min=1,max=100,text='Size of point grid'),
    ##                 _I('refine',0,text='Number of surface refinements'),
    ##                 ])
    ## if not res:
    ##     exit()
    ## n = res['size']
    ## m = res['refine']
    ## doit(n,m)
    

if __name__ == 'draw':
    run()
# End


