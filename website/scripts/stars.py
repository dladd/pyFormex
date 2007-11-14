#!/usr/bin/env pyformex
from numpy import random
nstars = 100 # number of stars
npoints = 7 # number of points in the star
noise = 0.3 # relative amplitude of noise
displ = nstars*0.6 # relative displacement
def star(n,noise=0.,prop=0):
    m = n/2
    f = Formex([[[0,1]]]).rosette(n,m*360./n).data()
    if noise != 0.:
        f = f + noise * random.random(f.shape)
    P = Formex(concatenate([f,f[:1]]))
    return Formex.connect([P,P],bias=[0,1]).setProp(prop)
Stars = Formex.concatenate( [ star(npoints,noise,i).translate(displ*random.random((3,))) for i in range(nstars) ])
clear()
draw(Stars)
