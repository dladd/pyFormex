#!/usr/bin/env python
# $Id$
##
## This file is part of pyFormex 0.4.2 Release Mon Feb 26 08:57:40 2007
## pyFormex is a python implementation of Formex algebra
## Homepage: http://pyformex.berlios.de/
## Distributed under the GNU General Public License, see file COPYING
## Copyright (C) Benedict Verhegghe except where stated otherwise 
##
"""A timer class."""

from datetime import datetime

class Timer(object):
    """A class for measuring elapsed time."""

    def __init__(self):
        """Create and start a timer."""
        self.reset()

    def reset(self):
        """Start the timer."""
        self.start = datetime.now()

    def read(self):
        """Read the timer.

        This returns the elapsed time since the last reset or then creation
        of the timer as a datetime.timedelta object.
        """
        now = datetime.now()
        return now - self.start
    
    def seconds(self):
        """Return the timer readings in seconds."""
        e = self.read()
        return e.days*24*3600 + e.seconds + int(round(e.microseconds / 1000000.))
        

if __name__ == "__main__":

    import time
    
    t = Timer()
    time.sleep(5)
    r = t.read()
    print r.days,r.seconds,r.microseconds
    print t.seconds()

# End
