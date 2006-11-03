#!/usr/bin/env pyformex
# $Id$
#
"""Sphere_stl"""

clear()
top = 0.
bot = -90.
r = 1.
n = 8
m = 12  # initial divisions

# Create points
dy = float(top-bot) / n
F = [ Formex(zeros((m+1,1,3))) ]
for i in range(n):
    dx = 360./(m+i)
    f = Formex([[[j*dx,(i+1)*dy,0]] for j in range(m+i+1)])
    F.append(f)

# Create Lines
message("Line model")
G = [[],[],[]]
for i,f in enumerate(F[1:]):
    G[0].append(connect([f,f],bias=[0,1]))
    G[1].append(connect([F[i],f],bias=[0,0]))
    if i > 0:
        G[2].append(connect([F[i],f],bias=[0,1]))
G = map(Formex.concatenate,G)
for i,f in enumerate(G):
    f.setProp(i)
G = Formex.concatenate(G)

clear()
draw(G)
#print G.bbox()
L = G.translate([0,bot,r]).spherical()
clear()
draw(L)

# Create Triangles
message("Surface model")
G = [[],[]]
for i,f in enumerate(F[1:]):
    G[0].append(connect([F[i],f,f],bias=[0,1,0]))
    if i > 0:
        G[1].append(connect([F[i],F[i],f],bias=[0,1,1]))
G = map(Formex.concatenate,G)
for i,f in enumerate(G):
    f.setProp(i)
G = Formex.concatenate(G)

clear()
draw(G)

flat()
GD.canvas.update()
T = G.translate([0,bot,r]).spherical()
clear()
draw(T)

T += T.reflect()
clear()
draw(T)

if ack('Export this model in STL format?'):
    fn = askFilename(GD.cfg['workdir'],"Stl files (*.stl)",exist=False)
    if fn:
        from plugins import stl
        f = file(fn,'w')
        stl.write_ascii(T.f,f)
        f.close()

