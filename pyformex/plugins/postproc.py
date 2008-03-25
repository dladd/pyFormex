# $Id$

from numpy import *

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

# End
