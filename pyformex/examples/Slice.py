#!/usr/bin/env pyformex --gui
# $Id$

from plugins.surface import TriSurface


def askSlices(bb):
    res = askItems([['Direction',0],
                    ['# slices',25],
                    ['total rot',70.],
                   ],caption = 'Define the slicing planes')
    if res:
        axis = res['Direction']
        nslices = res['# slices']
        totalrot = res['total rot']
        xmin,xmax = bb[:,axis]
        dx =  (xmax-xmin) / nslices
        print "Distance between slices: %s" % dx
        x = arange(nslices+1) * dx
        N = unitVector(axis)
        P = [ bb[0]+N*s for s in x ]
        return P,N,totalrot
    else:
        return [],[]

smooth()
setView('horse',[20,20,0])
chdir('/home/bene/prj/pyformex/stl')
S = TriSurface.read('horse-upright.gts')
bb = S.bbox()

t = -0.3
bb[0] = (1.0-t)*bb[0] + t*bb[1]
draw(S,bbox=bb,view='front')

P,n,t = askSlices(S.bbox())
a = t/len(P)

F = S.toFormex()
G = []
old = seterr(all='ignore')
for i,p in enumerate(P):
    F1 = F.cutAtPlane(p,-n)
    F1.setProp(i)
    G = [ g.rot(a,around=p) for g in G ] 
    G.append(F1)
    F = F.cutAtPlane(p,n)
    clear()
    draw(F,bbox=None)
    draw(G,bbox=None)

seterr(**old)
    
x = GD.canvas.width()/2
y = GD.canvas.height() - 30
drawtext("No animals got hurt during the making of this movie!",x,y,font='tr24',adjust='center')

# End
