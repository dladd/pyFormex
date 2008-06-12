#!/usr/bin/env pyformex --gui
# $Id$
##
## This file is part of pyFormex 0.7.1 Release Sat May 24 13:26:21 2008
## pyFormex is a Python implementation of Formex algebra
## Website: http://pyformex.berlios.de/
## Copyright (C) Benedict Verhegghe (benedict.verhegghe@ugent.be) 
##
## This program is distributed under the GNU General Public License
## version 2 or later (see file COPYING for details)
##

def superEgg(size,n,e,u,v):
    """Return a parametric superegg surface

    u and v are the parameter values and should be withinn the ranges
    -pi < u <pi, -pi/2 < v < pi/2

    Returns a nu*nv*3 array of coordinates.
    """
    def c(o,m):
        c = cos(o)
        return sign(c)*abs(c)**m
    def s(o,m):
        c = sin(o)
        return sign(c)*abs(c)**m

    u = u.reshape(-1)
    v = v.reshape(-1,1)
    cvn = c(v,n)
    x = zeros((v.shape[0],u.shape[0],3))
    x[...,0] = size[0] * cvn * c(u,e)
    x[...,1] = size[1] * cvn * s(u,e)
    x[...,2] = size[2] * s(v,n)
    return Coords(x)

def rng(o,r,n):
    return arange(n+1) * r/n + o

        
if __name__ == "draw":

    reset()
    smoothwire()
    lights(False)
    transparent(False)
    view('iso')

    size = [1.,1.,1.]
    north_south = 1.
    east_west = 1.
    grid = [24,16]
    half = False

    while True:
        res = askItems([('size',size),
                        ('north_south',north_south),
                        ('east_west',east_west),
                        ('grid',grid),
                        ('half',half),
                        ],caption="SuperEgg parameters")
        if not res:
            break;

        globals().update(res)

        u = rng(-pi,2*pi,grid[0])
        if half:
            v = rng(0,pi/2,grid[1]/2)
        else:
            v = rng(-pi/2,pi,grid[1])

        x = superEgg(size,north_south,east_west,u,v)
        F = Formex(x)

        FS = [ i.points() for i in F.split() ]
        H = Formex.concatenate([connect([i,i,j,j],bias=[0,1,1,0]) for i,j in zip(FS[1:],FS[:-1])])

        clear()
        draw(H,color='gold',view=None,bbox=None)
#        zoomAll()
          
        # Break from endless loop if an input timeout is active !
        if widgets.input_timeout >= 0:
            break

    exit()


# End
