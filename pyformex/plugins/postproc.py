# $Id$

"""Postprocessing functions

Postprocessing means collecting a geometrical model and computed values
from a numerical simulation, and render the values on the domain.
"""

from numpy import *
from formex import *
from gui.draw import *


DB = None

def setDB(db):
    global DB
    DB = db


def niceNumber(f,approx=floor):
    """Returns a nice number close to but not smaller than f."""
    n = int(approx(log10(f)))
    m = int(str(f)[0])
    return m*10**n


def frameScale(nframes=10,cycle='up',shape='linear'):
    """Return a sequence of scale values between -1 and +1.

    nframes is the number of steps between 0 and |1| values.

    cycle determines how subsequent cycles occur:
      'up' : ramping up
      'updown': ramping up and down
      'revert': ramping up and down then reverse up and down

    shape determines the shape of the amplitude curve:
      'linear': linear scaling
      'sine': sinusoidal scaling
    """
    s = arange(nframes+1)
    if cycle in [ 'updown', 'revert' ]:
        s = concatenate([s, fliplr(s[:-1].reshape((1,-1)))[0]])
    if cycle in [ 'revert' ]: 
        s = concatenate([s, -fliplr(s[:-1].reshape((1,-1)))[0]])
    return s.astype(float)/nframes


#############################################################
# Do something with the data
# These function should be moved to a more general postprocessor
#

def showModel(nodes=True,elems=True):
    if nodes:
        Fn = Formex(DB.nodes)
        draw(Fn)
    if elems:
        Fe = [ Formex(DB.nodes[elems],i+1) for i,elems in enumerate(DB.elems.itervalues()) ]
        draw(Fe)
    zoomAll()

# End
