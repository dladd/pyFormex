#!/usr/bin/env pyformex --gui
# $Id$
"""Grid"""
import gui.actors

res = askItems([('nx',4),('ny',3),('nz',2),('Grid type','','select',['Box','Plane']),('alpha',0.3)])

if not res:
    exit()
    
nx = (res['nx'],res['ny'],res['nz'])
gridtype = res['Grid type']
alpha = res['alpha']

if gridtype == 'Box':
    GA = actors.GridActor(nx=nx,linewidth=0.2,alpha=alpha)
else:
    GA = actors.CoordPlaneActor(nx=nx,linewidth=0.2,alpha=alpha)

smooth()
GD.canvas.addActor(GA)
GD.canvas.setBbox(GA.bbox())
zoomAll()
GD.canvas.update()

